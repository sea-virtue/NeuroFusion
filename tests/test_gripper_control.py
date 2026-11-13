import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from neurofusion.adapters.robotis import RobotisGripper,PositionBoundary
from neurofusion.gripper_control import run
from test_robotis import FakePacket,FakePort


class GripperControlTests(unittest.TestCase):
    def setUp(self):
        ceiling = patch('neurofusion.adapters.robotis.load_position_ceiling',return_value=None)
        ceiling.start()
        self.addCleanup(ceiling.stop)

    def session(self,packet):
        sdk = types.SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        return patch.dict(sys.modules,{'dynamixel_sdk':sdk})

    def test_status_is_read_only(self):
        packet = FakePacket()
        with tempfile.TemporaryDirectory() as root,self.session(packet),contextlib.redirect_stdout(io.StringIO()):
            report = run(output=Path(root)/'status')
        self.assertTrue(report['passed'])
        self.assertFalse(report['final']['torque_enabled'])
        self.assertEqual(packet.log,[])

    def test_operator_ceiling_rejects_close_before_torque_or_any_write(self):
        packet = FakePacket()
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(510,510)),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'tightening boundary'):
                run('close',steps=12,output=Path(root)/'blocked')
        self.assertEqual(packet.log,[])

    def test_close_open_and_absolute_move_use_measured_direction_and_disable(self):
        for action,kwargs,target in (('close',{'steps':12},515),('open',{'steps':12},491),
                                     ('move',{'position':510},510)):
            with self.subTest(action=action):
                packet = FakePacket()
                def arrive(device,receipt,on_sample):
                    self.assertEqual(receipt.target_raw,target)
                    packet.values[611] = target
                    sample = device.read()
                    on_sample(sample)
                    return sample
                with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                        'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None),patch.object(
                        RobotisGripper,'wait',arrive),contextlib.redirect_stdout(io.StringIO()):
                    output = Path(root)/action
                    report = run(action,output=output,**kwargs)
                    saved = json.loads((output/'report.json').read_text())
                self.assertTrue(saved['passed'])
                self.assertEqual(report['target_raw'],target)
                self.assertEqual(report['reached']['position_raw'],target)
                self.assertEqual(report['profile_acceleration_raw'],255)
                self.assertEqual(report['profile_velocity_raw'],100)
                self.assertEqual(packet.values[606],0)
                self.assertEqual(packet.values[600],0)
                self.assertFalse(report['final']['torque_enabled'])
                self.assertEqual(packet.values[562],0)
                self.assertEqual(packet.values[604],820)

    def test_out_of_range_target_is_rejected_before_enabling(self):
        for action,kwargs in (('close',{'steps':12}),('move',{'position':1151})):
            with self.subTest(action=action):
                packet = FakePacket()
                packet.values[611] = 1149
                with tempfile.TemporaryDirectory() as root,self.session(packet),contextlib.redirect_stdout(io.StringIO()):
                    output = Path(root)/'invalid'
                    with self.assertRaises(ValueError):
                        run(action,output=output,**kwargs)
                    self.assertFalse(json.loads((output/'report.json').read_text())['passed'])
                self.assertEqual(packet.log,[])

    def test_wait_failure_exits_with_torque_off(self):
        packet = FakePacket()
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None),patch.object(
                RobotisGripper,'wait',side_effect=TimeoutError('no arrival')),contextlib.redirect_stdout(io.StringIO()):
            output = Path(root)/'timeout'
            with self.assertRaises(TimeoutError):
                run('close',output=output)
            self.assertFalse(json.loads((output/'report.json').read_text())['passed'])
        self.assertEqual(packet.values[562],0)
        self.assertEqual(packet.values[604],820)

    def test_explicit_low_motion_settings_preserve_original_profile_option(self):
        packet=FakePacket()
        def arrive(device,receipt,on_sample):
            packet.values[611]=receipt.target_raw
            return device.read()
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None),patch.object(
                RobotisGripper,'wait',arrive),contextlib.redirect_stdout(io.StringIO()):
            report=run('open',steps=1,acceleration='low',speed='low',output=Path(root)/'low')
        self.assertEqual(report['profile_acceleration_raw'],1)
        self.assertEqual(report['profile_velocity_raw'],5)
        self.assertEqual(packet.values[606],0)

    def test_invalid_steps_and_hold_never_open_port(self):
        for kwargs in ({'steps':-1},{'steps':0},{'steps':True},{'steps':1.5},{'hold_seconds':float('nan')},{'hold_seconds':31},
                       {'speed':'invalid'},{'acceleration':'invalid'}):
            with self.subTest(kwargs=kwargs),patch.object(RobotisGripper,'__enter__') as enter:
                with self.assertRaises(ValueError):
                    run('close',**kwargs)
                enter.assert_not_called()

    def test_large_manual_moves_use_full_hardware_span_and_preserve_position_ceiling(self):
        for action,kwargs,target in (('close',{'steps':100},500),('open',{'steps':100},300),
                                     ('move',{'position':50},50)):
            packet=FakePacket();packet.values[611]=400
            def arrive(device,receipt,on_sample):
                self.assertEqual(receipt.target_raw,target)
                packet.values[611]=target
                sample=device.read()
                device._check_motion_sample(sample)
                return sample
            with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                    'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None),patch(
                    'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(575,562)),patch.object(
                    RobotisGripper,'wait',arrive),contextlib.redirect_stdout(io.StringIO()):
                report=run(action,output=Path(root)/action,**kwargs)
            self.assertTrue(report['passed'])
            self.assertEqual(report['manual_travel_limits_raw'],dict(max_step_raw=1150,session_radius_raw=1150))
            self.assertEqual(packet.values[562],0)
        # Full-range manual configuration must still reject crossing the cap.
        packet=FakePacket()
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(575,562)),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'tightening boundary'):
                run('close',steps=100,output=Path(root)/'blocked')
        self.assertEqual(packet.log,[])

    def test_twenty_count_move_survives_one_count_change_before_command_check(self):
        packet=FakePacket()
        original_command=RobotisGripper.command
        def shifted_command(device,proposal):
            packet.values[611]=502  # Armed sample was 503; target is 523.
            return original_command(device,proposal)
        def arrive(device,receipt,on_sample):
            packet.values[611]=receipt.target_raw
            return device.read()
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None),patch.object(
                RobotisGripper,'command',shifted_command),patch.object(
                RobotisGripper,'wait',arrive),contextlib.redirect_stdout(io.StringIO()):
            report=run('close',steps=20,output=Path(root)/'jitter')
        self.assertTrue(report['passed'])
        self.assertEqual(report['receipt']['target_raw'],523)
        self.assertEqual(packet.values[562],0)


if __name__ == '__main__':
    unittest.main()
