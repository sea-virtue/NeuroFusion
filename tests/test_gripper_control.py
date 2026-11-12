import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from neurofusion.adapters.robotis import RobotisGripper
from neurofusion.gripper_control import run
from test_robotis import FakePacket,FakePort


class GripperControlTests(unittest.TestCase):
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
                self.assertFalse(report['final']['torque_enabled'])
                self.assertEqual(packet.values[562],0)
                self.assertEqual(packet.values[604],820)

    def test_out_of_range_target_is_rejected_before_enabling(self):
        for action,kwargs in (('close',{'steps':12}),('move',{'position':1100})):
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

    def test_oversized_relative_command_and_invalid_hold_never_open_port(self):
        for kwargs in ({'steps':21},{'steps':0},{'hold_seconds':float('nan')},{'hold_seconds':31}):
            with self.subTest(kwargs=kwargs),patch.object(RobotisGripper,'__enter__') as enter:
                with self.assertRaises(ValueError):
                    run('close',**kwargs)
                enter.assert_not_called()


if __name__ == '__main__':
    unittest.main()
