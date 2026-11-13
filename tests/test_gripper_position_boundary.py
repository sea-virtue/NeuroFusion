from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from neurofusion.adapters.robotis import GripperConfig, RobotisGripper, load_position_ceiling,PositionBoundary
from test_robotis import FakePacket, FakePort


class PositionBoundaryTests(unittest.TestCase):
    def test_direct_profile_preserves_both_target_and_feedback_boundaries(self):
        packet=FakePacket()
        sdk=SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with patch.dict('sys.modules',{'dynamixel_sdk':sdk}),patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(516,510)),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with RobotisGripper(GripperConfig(allow_motion=True,position_profile='direct')) as device:
                device.arm()
                before=list(packet.log)
                with self.assertRaises(ValueError):
                    device.command(SimpleNamespace(time_ns=time.monotonic_ns(),robotis=(511/1150,),normalized=False))
                self.assertEqual(packet.log,before)
                packet.values[611]=517
                with self.assertRaisesRegex(RuntimeError,'tightening boundary'):
                    device.command(SimpleNamespace(time_ns=time.monotonic_ns(),robotis=(510/1150,),normalized=False))
            self.assertEqual(packet.values[562],0)

    def test_interrupted_sdk_read_does_not_leave_torque_enabled(self):
        class BusyPacket(FakePacket):
            def __getattr__(self,name):
                method=super().__getattr__(name)
                if name.startswith('write'):
                    def write(port,*args):
                        if getattr(port,'is_using',False):return -1000,0
                        return method(port,*args)
                    return write
                return method
        packet=BusyPacket()
        sdk=SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with patch.dict('sys.modules',{'dynamixel_sdk':sdk}), patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=None), patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with self.assertRaises(KeyboardInterrupt):
                with RobotisGripper(GripperConfig(allow_motion=True)) as device:
                    device.arm()
                    device._port.is_using=True
                    raise KeyboardInterrupt
            self.assertEqual(packet.values[562],0)
            self.assertTrue(device._port.closed)

    def test_load_validates_device_and_preserves_hardware_scale(self):
        saved = dict(schema_version=1, model_number=35073, device_id=1, port='/dev/ttyUSB0',
                     min_raw=0, hardware_max_raw=1150, max_position_raw=510)
        limits = dict(min_raw=0, max_raw=1150)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'limit.json'
            path.write_text(json.dumps(saved))
            self.assertEqual(load_position_ceiling(GripperConfig(),35073,limits,path),PositionBoundary(510,510))
            path.write_text(json.dumps(saved|dict(command_margin_raw=3)))
            self.assertEqual(load_position_ceiling(GripperConfig(),35073,limits,path),PositionBoundary(510,507))
            for change in (dict(max_position_raw=True),dict(max_position_raw=1151),dict(model_number=35074),dict(device_id=2),dict(command_margin_raw=-1),dict(command_margin_raw=True)):
                path.write_text(json.dumps(saved|change))
                with self.assertRaises(ValueError):
                    load_position_ceiling(GripperConfig(),35073,limits,path)

    def test_command_and_low_level_writes_cannot_exceed_ceiling(self):
        packet = FakePacket()
        sdk = SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with patch.dict('sys.modules',{'dynamixel_sdk':sdk}), patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(510,510)), patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with RobotisGripper(GripperConfig(allow_motion=True)) as device:
                device.arm()
                before=list(packet.log)
                for operation in (lambda: device.command(SimpleNamespace(time_ns=time.monotonic_ns(),
                                        robotis=(511/1150,),normalized=False)),
                                  lambda: device._write(596,4,511),
                                  lambda: device._write_bytes(596,(511).to_bytes(4,'little')+b'\0'*6)):
                    with self.assertRaises(ValueError):operation()
                self.assertEqual(packet.log,before)
                self.assertEqual(device.raw_position(1.),1150)
                receipt=device.command(SimpleNamespace(time_ns=time.monotonic_ns(),robotis=(510/1150,),normalized=False))
                self.assertEqual(receipt.target_raw,510)
                packet.values[611]=511
                with self.assertRaises(RuntimeError):device._check_motion_sample(device.read())
            self.assertEqual(packet.values[562],0)

    def test_current_mode_cannot_bypass_position_ceiling(self):
        packet=FakePacket();packet.values[11]=0
        sdk=SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with patch.dict('sys.modules',{'dynamixel_sdk':sdk}), patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(510,510)), patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with RobotisGripper(GripperConfig(allow_motion=True,control_mode='current')) as device:
                with self.assertRaises(RuntimeError):device.arm()
            self.assertEqual(packet.log,[])

    def test_command_margin_keeps_physical_boundary_separate(self):
        packet=FakePacket()
        sdk=SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with patch.dict('sys.modules',{'dynamixel_sdk':sdk}), patch(
                'neurofusion.adapters.robotis.load_position_ceiling',return_value=PositionBoundary(510,507)), patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with RobotisGripper(GripperConfig(allow_motion=True)) as device:
                device.arm()
                with self.assertRaises(ValueError):device._check_position_target(508)
                device._check_position_target(507)
                packet.values[611]=509
                device._check_motion_sample(device.read())
                packet.values[611]=511
                with self.assertRaises(RuntimeError):device._check_motion_sample(device.read())
