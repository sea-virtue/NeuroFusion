import sys
import time
import types
import unittest
from dataclasses import replace
from unittest.mock import patch
from neurofusion.adapters.robotis import (RobotisGripper,GripperConfig,TABLES,
    GripperSample,GripperReceipt,GripperCurrentReceipt,confirmed_execution_observation)


class FakePort:
    def __init__(self, name): self.closed = False
    def openPort(self): return True
    def setBaudRate(self, baud): return True
    def closePort(self): self.closed = True


class FakePacket:
    def __init__(self, model=35073):
        self.model = model
        self.log = []
        self.fail_goal = False
        self.fail_zero_current = False
        self.short_read = False
        self.reads = []
        t = TABLES[model]
        self.values = {6:25,11:5,t.min_position:0,t.max_position:1150,
            t.current_limit:820,t.velocity_limit:100,t.acceleration_limit:255,
            t.min_voltage:150,t.max_voltage:300,t.status_return:2,
            t.present_position:503,t.present_velocity:0,t.present_current:0,
            t.voltage:243,t.temperature:41,t.moving:0,t.torque:0,
            t.hardware_error:0,t.goal_position:503,t.goal_current:820,
            t.profile_velocity:0,t.profile_acceleration:0}
    def ping(self, port, device): return self.model,0,0
    def getTxRxResult(self, value): return f'transport {value}'
    def getRxPacketError(self, value): return f'device {value}'
    def readTxRx(self, port, device, address, size):
        self.reads.append((address,size))
        data = bytearray(size)
        # Legacy RN block has reserved bytes between velocity and current.
        for reg,width in ((596,4),(600,4),(604,2),(606,4),(610,1),(611,4),
                          (615,4),(621,2),(623,2),(625,1)):
            if address <= reg and reg+width <= address+size:
                value = self.values[reg] & ((1 << (width*8))-1)
                data[reg-address:reg-address+width] = value.to_bytes(width,'little')
        return list(data[:-1] if self.short_read else data),0,0
    def writeTxRx(self, port, device, address, size, data):
        self.log.append((address,bytes(data)))
        if self.fail_goal and address == TABLES[self.model].goal_position:
            return -1001,0
        if address != 596 or size != 10:
            raise AssertionError('Unexpected legacy RN position profile block')
        for reg,offset,width in ((596,0,4),(600,4,4),(604,8,2)):
            self.values[reg] = int.from_bytes(bytes(data[offset:offset+width]),'little')
        return 0,0
    def __getattr__(self, name):
        if name.startswith('read'):
            return lambda port,device,address:(self.values[address],0,0)
        if name.startswith('write'):
            def write(port, device, address, value):
                self.log.append((address,value))
                if self.fail_goal and address == TABLES[self.model].goal_position:
                    return -1001,0
                if self.fail_zero_current and address == TABLES[self.model].goal_current and value == 0:
                    return -1001,0
                self.values[address] = value
                return 0,0
            return write
        raise AttributeError(name)


class RobotisTests(unittest.TestCase):
    def setUp(self):
        ceiling = patch('neurofusion.adapters.robotis.load_position_ceiling',return_value=None)
        ceiling.start()
        self.addCleanup(ceiling.stop)
        # Unit tests exercise logic behind the real-device incident interlock.
        self.hold = patch('neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None)
        self.hold.start()
        self.addCleanup(self.hold.stop)

    def session(self, packet=None, **kwargs):
        packet = packet or FakePacket()
        # Existing low-current commissioning cases remain explicit; the production
        # default using the hardware maximum is tested separately below.
        kwargs.setdefault('use_hardware_current_limit',False)
        sdk = types.SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        return patch.dict(sys.modules,{'dynamixel_sdk':sdk}),RobotisGripper(GripperConfig(**kwargs)),packet

    def proposal(self, closure, normalized=False, timestamp=None):
        return types.SimpleNamespace(robotis=(closure,),normalized=normalized,
                                     time_ns=time.monotonic_ns() if timestamp is None else timestamp)

    def test_read_only_defaults_do_not_write_and_decode_both_versions(self):
        for model in TABLES:
            mock,device,packet = self.session(FakePacket(model))
            with mock,device:
                sample = device.read()
                self.assertAlmostEqual(sample.closure,503/1150)
                self.assertAlmostEqual(sample.voltage_v,24.3)
                with self.assertRaises(RuntimeError): device.arm()
            self.assertEqual(packet.log,[])
            self.assertTrue(device._port.closed)

    def test_arm_preloads_position_and_low_limits_before_enabling_then_restores_off(self):
        mock,device,packet = self.session(allow_motion=True)
        t = TABLES[35073]
        with mock,device:
            device.arm()
            payload = (503).to_bytes(4,'little')+(5).to_bytes(4,'little')+(49).to_bytes(2,'little')
            self.assertEqual(packet.log[:4],[(t.profile_acceleration,1),
                                           (t.goal_position,payload),(t.torque,1),
                                           (t.goal_position,503)])
            self.assertGreater(packet.values[t.profile_velocity],0)
            self.assertLessEqual(packet.values[t.goal_current]*t.current_unit_a,.2)
        self.assertEqual(packet.values[t.torque],0)
        self.assertEqual(packet.values[t.goal_current],820)
        self.assertEqual(packet.values[t.profile_velocity],0)

    def test_rna_preload_uses_its_own_addresses_not_the_legacy_contiguous_block(self):
        mock,device,packet = self.session(FakePacket(35074),allow_motion=True)
        t = TABLES[35074]
        with mock,device:
            device.arm()
            self.assertEqual([address for address,value in packet.log],
                [t.profile_acceleration,t.goal_current,t.profile_velocity,t.goal_position,t.torque,t.goal_position])
            self.assertTrue(all(isinstance(value,int) for _,value in packet.log))

    def test_low_latency_is_configured_before_ping_and_failure_keeps_reads_available(self):
        for failure in (False,True):
            with self.subTest(failure=failure):
                events = []
                class LowLatencyPort(FakePort):
                    def __init__(self,name):
                        super().__init__(name)
                        def configure(enabled):
                            events.append(('low_latency',enabled))
                            if failure:
                                raise ValueError('driver refused')
                        self.ser = types.SimpleNamespace(set_low_latency_mode=configure)
                class OrderedPacket(FakePacket):
                    def ping(self,port,device):
                        events.append(('ping',device))
                        return super().ping(port,device)
                packet = OrderedPacket()
                sdk = types.SimpleNamespace(PortHandler=LowLatencyPort,PacketHandler=lambda protocol:packet)
                with patch.dict(sys.modules,{'dynamixel_sdk':sdk}),RobotisGripper(
                        GripperConfig(allow_motion=True)) as device:
                    self.assertEqual(device.read().position_raw,503)
                    self.assertEqual(events[:2],[('low_latency',True),('ping',1)])
                    self.assertEqual(device.usb_low_latency_enabled,not failure)
                    if failure:
                        with self.assertRaisesRegex(RuntimeError,'low-latency setup failed'):
                            device.arm()
                        self.assertEqual(packet.log,[])
                    else:
                        self.assertIsNone(device.usb_low_latency_error)

    def test_position_arm_refreshes_active_goal_before_feedback_can_follow_stale_trajectory(self):
        class StaleTrajectoryPacket(FakePacket):
            needs_refresh = False
            def __getattr__(self,name):
                operation = super().__getattr__(name)
                if not name.startswith('write'):
                    return operation
                def write(port,device,address,value):
                    result = operation(port,device,address,value)
                    if address == 562 and value == 1:
                        self.needs_refresh = True
                    elif address == 596 and self.values[562]:
                        self.needs_refresh = False
                    return result
                return write
            def readTxRx(self,port,device,address,size):
                if self.values[562] and self.needs_refresh:
                    self.values[611] = 450
                return super().readTxRx(port,device,address,size)
        mock,device,packet = self.session(StaleTrajectoryPacket(),allow_motion=True)
        with mock,device:
            armed = device.arm()
            self.assertEqual(armed.position_raw,503)
            self.assertFalse(packet.needs_refresh)
        self.assertEqual(packet.values[562],0)

    def test_failed_active_position_refresh_still_disables_and_restores(self):
        class FailedRefreshPacket(FakePacket):
            def __getattr__(self,name):
                operation = super().__getattr__(name)
                if name != 'write4ByteTxRx':
                    return operation
                def write(port,device,address,value):
                    if address == 596 and self.values[562]:
                        self.log.append((address,value))
                        return -1001,0
                    return operation(port,device,address,value)
                return write
        mock,device,packet = self.session(FailedRefreshPacket(),allow_motion=True)
        with self.assertRaisesRegex(RuntimeError,'transport'):
            with mock,device:
                device.arm()
        self.assertEqual(packet.values[562],0)
        self.assertEqual(packet.values[604],820)
        self.assertEqual(packet.values[600],0)
        self.assertTrue(device._port.closed)

    def test_position_uses_read_device_limit_and_matching_feedback_threshold(self):
        for model,limit in ((35073,820),(35073,500),(35074,820)):
            packet = FakePacket(model)
            t = TABLES[model]
            packet.values[t.current_limit] = limit
            mock,device,packet = self.session(packet,allow_motion=True,
                use_hardware_current_limit=True)
            with mock,device:
                self.assertAlmostEqual(device.config.current_cap_a,limit*t.current_unit_a)
                device.arm()
                self.assertEqual(packet.values[t.goal_current],limit)
                sample = replace(device.read(),current_a=device.config.current_cap_a-.01)
                device._check_motion_sample(sample)
                with self.assertRaisesRegex(RuntimeError,'current exceeds'):
                    device._check_motion_sample(replace(sample,current_a=device.config.current_cap_a+.04))
                self.assertFalse(any(address == t.current_limit for address,value in packet.log))

    def test_current_mode_does_not_turn_maximum_position_ceiling_into_current_target(self):
        packet = FakePacket()
        packet.values[11] = 0
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current',
            use_hardware_current_limit=True)
        with mock,device:
            self.assertEqual(device.config.current_cap_a,.2)
            device.arm()
            self.assertEqual(packet.values[604],0)
            with self.assertRaises(ValueError): device.command_current(3.)

    def test_block_feedback_handles_signed_values_gaps_and_short_reads(self):
        mock,device,packet = self.session()
        t = TABLES[35073]
        packet.values[t.present_velocity] = (1 << 32)-3
        packet.values[t.present_current] = (1 << 16)-2
        with mock,device:
            sample = device.read()
            self.assertEqual(sample.velocity_raw,-3)
            self.assertAlmostEqual(sample.current_a,-.00804)
            self.assertEqual(packet.reads,[(596,30)])
            packet.short_read = True
            with self.assertRaisesRegex(RuntimeError,'Short register read'):
                device.read()

    def test_current_arm_zeros_old_target_without_changing_mode_or_position_profile(self):
        for model in TABLES:
            packet = FakePacket(model)
            packet.values[11] = 0
            mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
            t = TABLES[model]
            with mock,device:
                sample = device.arm()
                self.assertEqual(sample.operating_mode,0)
                self.assertEqual(packet.log,[(t.goal_current,0),(t.torque,1)])
            self.assertEqual(packet.values[t.torque],0)
            self.assertEqual(packet.values[t.goal_current],0)

    def test_mode_mismatch_and_position_current_confusion_never_send_targets(self):
        packet = FakePacket()
        packet.values[11] = 0
        mock,device,packet = self.session(packet,allow_motion=True)
        with mock,device:
            with self.assertRaises(RuntimeError): device.arm()
        self.assertEqual(packet.log,[])
        mock,device,packet = self.session(allow_motion=True)
        with mock,device:
            device.arm()
            count = len(packet.log)
            with self.assertRaises(RuntimeError): device.command_current(.05)
            self.assertEqual(len(packet.log),count)

    def test_signed_current_is_bounded_quantized_and_not_a_position_receipt(self):
        packet = FakePacket()
        packet.values[11] = 0
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
        with mock,device:
            device.arm()
            for target in (.05,-.05,0.):
                receipt = device.command_current(target)
                self.assertIsInstance(receipt,GripperCurrentReceipt)
                self.assertLessEqual(abs(receipt.target_current_a),abs(target))
                self.assertAlmostEqual(receipt.target_current_a,receipt.target_raw*.00402)
                self.assertEqual(packet.values[604],receipt.target_raw & 0xFFFF)
            count = len(packet.log)
            for target in (float('nan'),float('inf'),.201,-.201,.001):
                with self.assertRaises(ValueError): device.command_current(target)
            with self.assertRaises(RuntimeError): device.command(self.proposal(.44))
            with self.assertRaises(RuntimeError): device.wait(receipt)
            self.assertEqual(len(packet.log),count)

    def test_current_hardware_limit_and_measured_session_range_are_enforced(self):
        packet = FakePacket()
        packet.values[11] = 0
        packet.values[30] = 10
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
        with mock,device:
            device.arm()
            count = len(packet.log)
            with self.assertRaises(ValueError): device.command_current(.1)
            packet.values[611] = 550
            with self.assertRaises(RuntimeError): device.command_current(.02)
            self.assertEqual(len(packet.log),count)

    def test_current_trial_sets_one_goal_then_disables_on_callback_failure(self):
        packet = FakePacket()
        packet.values[11] = 0
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
        with mock,device:
            device.arm()
            def failed(sample):
                raise RuntimeError('feedback consumer failed')
            with self.assertRaisesRegex(RuntimeError,'feedback consumer failed'):
                device.hold_current(.05,.01,failed)
            self.assertEqual(packet.values[562],0)
            self.assertEqual(packet.values[604],0)
            self.assertEqual(sum(1 for address,value in packet.log if address == 604 and value != 0),1)

    def test_current_trial_without_arm_never_disables_another_controller(self):
        packet = FakePacket()
        packet.values[11] = 0
        packet.values[562] = 1
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
        with mock,device:
            with self.assertRaisesRegex(RuntimeError,'not been armed'):
                device.hold_current(.05,.01)
        self.assertEqual(packet.log,[])

    def test_excess_current_immediately_after_enable_disables_both_modes(self):
        for mode in ('position','current'):
            packet = FakePacket()
            packet.values[11] = 5 if mode == 'position' else 0
            mock,device,packet = self.session(packet,allow_motion=True,control_mode=mode)
            with self.assertRaisesRegex(RuntimeError,'current exceeds'):
                with mock,device:
                    initial = device.read()
                    after = replace(initial,torque_enabled=True,current_a=.5)
                    with patch.object(device,'read',side_effect=[initial,after]):
                        device.arm()
            self.assertEqual(packet.values[562],0)

    def test_even_a_short_current_trial_reads_feedback_before_shutting_down(self):
        packet = FakePacket()
        packet.values[11] = 0
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
        with mock,device:
            device.arm()
            samples = []
            receipt,last = device.hold_current(.05,1e-12,samples.append)
            self.assertEqual(len(samples),1)
            self.assertEqual(last,samples[-1])
            self.assertEqual(packet.values[562],0)

    def test_zero_current_write_failure_still_attempts_torque_off(self):
        packet = FakePacket()
        packet.values[11] = 0
        mock,device,packet = self.session(packet,allow_motion=True,control_mode='current')
        with self.assertRaises(RuntimeError):
            with mock,device:
                device.arm()
                packet.fail_zero_current = True
                device.disarm()
        self.assertEqual(packet.values[562],0)

    def test_mode_change_is_explicit_eeprom_write_requires_torque_off_and_interlock(self):
        mock,device,packet = self.session(allow_motion=True,control_mode='current')
        with mock,device:
            packet.values[562] = 1
            with self.assertRaises(RuntimeError): device.set_mode('current')
            self.assertEqual(packet.log,[])
            packet.values[562] = 0
            with patch('neurofusion.adapters.robotis.motion_inhibit_reason',return_value='incident'):
                with self.assertRaises(RuntimeError): device.set_mode('current')
            self.assertEqual(packet.log,[])
            device.set_mode('current')
            self.assertEqual(packet.log,[(11,0)])
            device.arm()

    def test_current_feedback_cannot_be_reported_as_executed_position(self):
        sample = GripperSample(30,503,503/1150,0,0.,24.3,42,False,True,0,503,
                               operating_mode=0)
        with self.assertRaises(ValueError):
            confirmed_execution_observation(GripperReceipt(10,20,503,503/1150),sample,None)
        with self.assertRaises(ValueError):
            confirmed_execution_observation(GripperCurrentReceipt(10,20,10,.0402),sample,None)

    def test_normalized_stale_nonfinite_and_large_goals_never_reach_write(self):
        mock,device,packet = self.session(allow_motion=True)
        with mock,device:
            device.arm()
            count = len(packet.log)
            for proposal in (self.proposal(.43,True),self.proposal(.43,timestamp=0),
                             self.proposal(float('nan')),self.proposal(1.1),self.proposal(.9)):
                with self.assertRaises(ValueError): device.command(proposal)
                self.assertEqual(len(packet.log),count)

    def test_ack_does_not_imply_measured_arrival_timeout_turns_torque_off(self):
        mock,device,packet = self.session(allow_motion=True,timeout_s=.01)
        with mock,device:
            device.arm()
            receipt = device.command(self.proposal(device.closure(483)))
            self.assertEqual(receipt.target_raw,483)
            self.assertEqual(device.read().position_raw,503)
            with self.assertRaises(TimeoutError): device.wait(receipt)
            self.assertEqual(packet.values[TABLES[35073].torque],0)

    def test_write_failure_triggers_context_cleanup(self):
        mock,device,packet = self.session(allow_motion=True)
        with self.assertRaises(RuntimeError):
            with mock,device:
                device.arm()
                packet.fail_goal = True
                device.command(self.proposal(device.closure(483)))
        self.assertEqual(packet.values[TABLES[35073].torque],0)
        self.assertTrue(device._port.closed)

    def test_hardware_fault_prevents_motion_and_does_not_clear_errors(self):
        mock,device,packet = self.session(allow_motion=True)
        packet.values[TABLES[35073].hardware_error] = 4
        with mock,device:
            with self.assertRaises(RuntimeError): device.arm()
        self.assertEqual(packet.log,[])

    def test_unresolved_incident_blocks_even_explicit_motion_before_any_write(self):
        mock,device,packet = self.session(allow_motion=True)
        with mock,device,patch('neurofusion.adapters.robotis.motion_inhibit_reason',return_value='uncommanded opening'):
            with self.assertRaisesRegex(RuntimeError,'Physical motion inhibited'):
                device.arm()
        self.assertEqual(packet.log,[])

    def test_session_range_prevents_multiple_small_steps_becoming_large_motion(self):
        mock,device,packet = self.session(allow_motion=True)
        with mock,device:
            device.arm()
            packet.values[TABLES[35073].present_position] = 483
            count = len(packet.log)
            with self.assertRaises(ValueError): device.command(self.proposal(device.closure(463)))
            self.assertEqual(len(packet.log),count)

    def test_measured_feedback_preserves_arm_fields_and_requires_fresh_arrival(self):
        import torch
        state = types.SimpleNamespace(robot=torch.arange(16,dtype=torch.float32)[None],
                                      executed=torch.zeros(1,8),time_ns=1)
        sample = GripperSample(30,503,503/1150,0,0.,24.3,42,False,True,0,503)
        observation = sample.observation(state)
        self.assertTrue(torch.equal(observation.value[:,:15],state.robot[:,:15]))
        self.assertAlmostEqual(observation.value[0,15].item(),503/1150)
        receipt = GripperReceipt(10,20,503,503/1150)
        self.assertEqual(confirmed_execution_observation(receipt,sample,state).kind,'gripper_executed')
        for invalid in (replace(sample,host_ns=19),replace(sample,position_raw=480),
                        replace(sample,hardware_error=4)):
            with self.assertRaises(ValueError): confirmed_execution_observation(receipt,invalid,state)


if __name__ == '__main__': unittest.main()
