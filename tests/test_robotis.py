import sys
import time
import types
import unittest
from dataclasses import replace
from unittest.mock import patch
from neurofusion.adapters.robotis import (RobotisGripper,GripperConfig,TABLES,
    GripperSample,GripperReceipt,confirmed_execution_observation)


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
    def __getattr__(self, name):
        if name.startswith('read'):
            return lambda port,device,address:(self.values[address],0,0)
        if name.startswith('write'):
            def write(port, device, address, value):
                self.log.append((address,value))
                if self.fail_goal and address == TABLES[self.model].goal_position:
                    return -1001,0
                self.values[address] = value
                return 0,0
            return write
        raise AttributeError(name)


class RobotisTests(unittest.TestCase):
    def setUp(self):
        # Unit tests exercise logic behind the real-device incident interlock.
        self.hold = patch('neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None)
        self.hold.start()
        self.addCleanup(self.hold.stop)

    def session(self, packet=None, **kwargs):
        packet = packet or FakePacket()
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
            self.assertEqual(packet.log[3:5],[(t.goal_position,503),(t.torque,1)])
            self.assertGreater(packet.values[t.profile_velocity],0)
            self.assertLessEqual(packet.values[t.goal_current]*t.current_unit_a,.2)
        self.assertEqual(packet.values[t.torque],0)
        self.assertEqual(packet.values[t.goal_current],820)
        self.assertEqual(packet.values[t.profile_velocity],0)

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
