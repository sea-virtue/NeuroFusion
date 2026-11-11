import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from neurofusion.gripper_current_check import run
from test_robotis import FakePacket,FakePort


class CurrentCheckTests(unittest.TestCase):
    def session(self, packet):
        sdk = types.SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        return patch.dict(sys.modules,{'dynamixel_sdk':sdk})

    def test_default_only_reads_existing_position_mode_and_never_changes_eeprom(self):
        packet = FakePacket()
        with tempfile.TemporaryDirectory() as root,self.session(packet):
            report = run(Path(root)/'read')
        self.assertTrue(report['passed'])
        self.assertEqual(report['initial']['operating_mode'],5)
        self.assertEqual(packet.log,[])

    def test_current_exercise_preserves_the_incident_interlock(self):
        packet = FakePacket()
        packet.values[11] = 0
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value='incident pending'):
            path = Path(root)/'blocked'
            with self.assertRaisesRegex(RuntimeError,'Physical motion inhibited'):
                run(path,exercise=True,operator_ready=True)
            self.assertFalse(json.loads((path/'report.json').read_text())['passed'])
        self.assertEqual(packet.log,[])

    def test_exercise_requires_operator_and_existing_current_mode(self):
        packet = FakePacket()
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with self.assertRaises(ValueError):
                run(Path(root)/'not-ready',exercise=True)
            with self.assertRaises(RuntimeError):
                run(Path(root)/'wrong-mode',exercise=True,operator_ready=True)
        self.assertEqual(packet.log,[])

    def test_one_signed_target_is_logged_with_no_position_execution_claim(self):
        packet = FakePacket()
        packet.values[11] = 0
        with tempfile.TemporaryDirectory() as root,self.session(packet),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            report = run(Path(root)/'current',exercise=True,operator_ready=True,
                         current_a=-.05,seconds=.001)
        self.assertTrue(report['passed'])
        self.assertEqual(report['receipt']['target_raw'],-12)
        self.assertAlmostEqual(report['receipt']['target_current_a'],-.04824)
        self.assertEqual(packet.values[562],0)
        self.assertEqual(packet.values[604],0)
        self.assertNotIn('confirmed_movements',report)
        self.assertEqual(report['trained_policy_commands_sent'],0)


if __name__ == '__main__':
    unittest.main()
