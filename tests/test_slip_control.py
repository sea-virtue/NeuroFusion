import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from neurofusion.control.slip import (SlipConfig,SlipController,SlipPrediction,
                                     GripperFeedback,GripperCommand)
from neurofusion.control.session import Episode,bc_rows
from neurofusion.control.__main__ import replay,demo_rows


BASE = 1_000_000_000


class SlipTests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(SlipConfig(grip_closure=.44,grip_settle_seconds=0.),**kwargs)

    def prediction(self, t, task='quiet', direction='down', seq=None):
        return SlipPrediction(t if seq is None else seq,BASE+t*1_000_000,BASE+t*1_000_000,
                              task,direction,.99,.99)

    def tick(self, controller, t, task='quiet', direction='down', closure=.44, prediction=None):
        now = BASE+t*1_000_000
        return controller.step(now,GripperFeedback(now,closure),
                               prediction or self.prediction(t,task,direction))

    def test_snn_onset_starts_timer_not_initial_opening(self):
        ctl = SlipController(self.config())
        self.tick(ctl,0)
        self.tick(ctl,20)
        first = self.tick(ctl,40,'slide',closure=.438)
        self.assertEqual(first['phase'],'OPENING')
        self.assertEqual(self.tick(ctl,80,'slide',closure=.438)['phase'],'SLIDING')
        for t in range(100,540,20): self.tick(ctl,t,'slide',closure=.438)
        self.assertEqual(ctl.phase,'SLIDING')
        decision = self.tick(ctl,540,'slide',closure=.438)
        self.assertEqual(decision['phase'],'BRAKING')
        self.assertEqual(decision['onset_available_ns'],BASE+40_000_000)
        self.assertGreater(decision['command']['closure'],.438)
        self.assertIsNone(decision['distance_m'])

    def test_repeated_prediction_cannot_confirm_motion(self):
        ctl = SlipController(self.config())
        self.tick(ctl,0)
        p = self.prediction(20,'slide')
        for t in (20,40,60,80): self.tick(ctl,t,prediction=p)
        self.assertEqual(ctl.phase,'OPENING')
        self.assertIsNone(ctl.onset_ns)

    def test_release_is_bounded_and_no_slide_times_out(self):
        ctl = SlipController(self.config(release_timeout=.2))
        values = [self.tick(ctl,t)['command'] for t in range(0,220,20)]
        self.assertTrue(all(v is None or v['closure'] >= .42 for v in values))
        self.assertEqual(ctl.reason,'no_slide_timeout')
        self.assertNotEqual(ctl.phase,'FINISHED')

    def test_wrong_direction_brakes_without_reopening(self):
        ctl = SlipController(self.config())
        self.tick(ctl,0)
        self.tick(ctl,20,'slide','up',closure=.438)
        self.assertEqual(ctl.phase,'BRAKING')
        self.assertEqual(ctl.reason,'unexpected_slide_direction')
        self.tick(ctl,40,'slide',closure=.44)
        self.assertEqual(ctl.phase,'FAULT')

    def test_stale_prediction_brakes_and_stale_feedback_never_sends(self):
        ctl = SlipController(self.config())
        self.tick(ctl,0)
        for t in range(20,180,20): self.tick(ctl,t,prediction=self.prediction(0))
        self.assertEqual(ctl.reason,'stale_or_missing_snn')
        other = SlipController(self.config())
        result = other.step(BASE+200_000_000,GripperFeedback(BASE,.44),self.prediction(200))
        self.assertEqual(result['phase'],'FAULT')
        self.assertIsNone(result['command'])

    def test_future_and_reversed_ticks_are_rejected(self):
        ctl = SlipController(self.config())
        with self.assertRaises(ValueError): self.tick(ctl,0,prediction=self.prediction(20))
        ctl = SlipController(self.config())
        self.tick(ctl,20)
        with self.assertRaises(ValueError): self.tick(ctl,20)

    def test_heartbeat_gap_brakes(self):
        ctl = SlipController(self.config())
        self.tick(ctl,0)
        self.tick(ctl,120,closure=.438)
        self.assertEqual(ctl.reason,'control_tick_gap')

    def test_invalid_config_and_probabilities(self):
        for changes in ({'slide_seconds':0},{'close_delay_seconds':.5},{'grip_closure':float('nan')},
                        {'expected_direction':'image_down'},{'max_open_delta':.6}):
            with self.assertRaises(ValueError): self.config(**changes)
        with self.assertRaises(ValueError): replace(self.prediction(0),task_confidence=float('nan'))

    def test_compensation_advances_braking(self):
        ctl = SlipController(self.config(close_delay_seconds=.1,confirm_seconds=0))
        self.tick(ctl,0)
        self.tick(ctl,20,'slide',closure=.438)
        for t in range(40,420,20): self.tick(ctl,t,'slide',closure=.438)
        self.assertEqual(self.tick(ctl,420,'slide',closure=.438)['phase'],'BRAKING')

    def test_shadow_fixture_is_not_bc_data(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'trial'
            cfg = SlipConfig(grip_closure=.44)
            self.assertEqual(replay(cfg,demo_rows(cfg),output,synthetic=True),'FINISHED')
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertTrue(manifest['synthetic'])
            self.assertEqual(manifest['commands_sent'],0)
            with self.assertRaises(ValueError): list(bc_rows(output))

    def test_failed_sink_records_error_not_successful_label(self):
        with tempfile.TemporaryDirectory() as root:
            episode = Episode(Path(root)/'trial',self.config(),synthetic=False)
            sink = SimpleNamespace(command=lambda cmd: (_ for _ in ()).throw(RuntimeError('write failed')))
            with self.assertRaises(RuntimeError):
                episode.tick(SlipController(self.config()),BASE,GripperFeedback(BASE,.44),self.prediction(0),sink)
            episode.close('failed')
            row = json.loads((episode.path/'trace.jsonl').read_text())
            self.assertIsNone(row['sent_command'])
            self.assertIn('error',row)

    def test_command_has_only_one_physical_output(self):
        cmd = GripperCommand(BASE,.44)
        self.assertEqual(cmd.robotis,(.44,))
        self.assertFalse(cmd.normalized)
        self.assertFalse(hasattr(cmd,'franka'))

    def test_command_reaches_existing_robotis_adapter_without_arm_fields(self):
        from test_robotis import FakePort,FakePacket
        from neurofusion.adapters.robotis import RobotisGripper,GripperConfig
        packet = FakePacket()
        sdk = SimpleNamespace(PortHandler=FakePort,PacketHandler=lambda protocol:packet)
        with patch.dict(sys.modules,{'dynamixel_sdk':sdk}),patch(
                'neurofusion.adapters.robotis.motion_inhibit_reason',return_value=None):
            with RobotisGripper(GripperConfig(allow_motion=True)) as device:
                device.arm()
                command = GripperCommand(time.monotonic_ns(),device.closure(500))
                receipt = device.command(command)
                self.assertEqual(receipt.target_raw,500)
                self.assertFalse(hasattr(command,'franka'))

    def test_stdin_stall_does_not_block_stream_watchdog(self):
        from neurofusion.control.__main__ import stream
        import threading
        finished = threading.Event()
        class Input:
            def __iter__(self):
                now = time.monotonic_ns()
                yield json.dumps({'sequence':1,'available_ns':now,'captured_ns':now,
                                  'task':'quiet','direction':'down','task_confidence':.99,'direction_confidence':.99})
                finished.wait(2.)
        class Device:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self):
                from neurofusion.adapters.robotis import GripperSample
                return GripperSample(time.monotonic_ns(),506,.44,0,0.,24.3,40,False,False,0,506)
            def _healthy(self,sample): pass
        with tempfile.TemporaryDirectory() as root,patch('sys.stdin',Input()),patch(
                'neurofusion.adapters.robotis.RobotisGripper',return_value=Device()):
            try:
                begin = time.monotonic()
                phase = stream(self.config(max_prediction_age=.03),Path(root)/'trial')
                self.assertEqual(phase,'FAULT')
                self.assertLess(time.monotonic()-begin,1.)
            finally: finished.set()

    def test_control_import_does_not_load_torch_or_arm_sdks(self):
        script = ('import sys; import neurofusion.control; '
                  'assert "torch" not in sys.modules; '
                  'assert "dynamixel_sdk" not in sys.modules')
        subprocess.run([sys.executable,'-c',script],check=True)

    def test_recognizer_uses_current_checkpoint_head(self):
        import torch
        from neurofusion.models.gelneuro_unified_snn import GelNeuroUnifiedSNN
        from neurofusion.control.snn import SNNRecognizer
        model = GelNeuroUnifiedSNN(1000)
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.copy_(torch.tensor([0.,10.,0.,0.,0.,10.,0.,0.]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'model.pt'
            torch.save({'model':model.state_dict(),'bin_us':1000,'input_gain':.25,
                        'classes':('quiet','slide','rotate','escape_wriggle'),
                        'directions':('up','down','left','right')},path)
            recognizer = SNNRecognizer(path,expected_direction='down')
            p = recognizer.predict(torch.empty(0,4),0,1000,time.monotonic_ns())
            self.assertEqual((p.task,p.direction),('slide','down'))
            with self.assertRaises(ValueError): recognizer.predict(torch.empty(0,4),2000,3000,time.monotonic_ns())


if __name__ == '__main__': unittest.main()
