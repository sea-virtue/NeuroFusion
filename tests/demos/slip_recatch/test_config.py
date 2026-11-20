from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from neurofusion.demos.slip_recatch.config import CountSlipConfig, parse_slip_config
from neurofusion.demos.slip_recatch.slip import SlipConfig, SlipController, SlipPrediction, GripperFeedback
from neurofusion.demos.slip_recatch.runtime import replay


class CountConfigTests(unittest.TestCase):
    def test_changing_only_travel_updates_timeout_and_session_radius(self):
        cfg=CountSlipConfig()
        self.assertEqual(cfg.release_timeout,23)
        self.assertEqual(cfg.gripper_session_radius_raw,45)
        larger=replace(cfg,max_open_counts=80)
        self.assertEqual(larger.release_timeout,43)
        self.assertEqual(larger.gripper_session_radius_raw,85)
        partial=replace(cfg,max_open_counts=41,open_step_counts=3)
        self.assertEqual(partial.release_timeout,10)

    def test_resolution_uses_hardware_scale_and_offset(self):
        cfg=CountSlipConfig().resolve({'min_raw':100,'max_raw':2100},700)
        self.assertEqual(cfg.grip_closure,.3)
        self.assertEqual(cfg.max_open_delta,.02)
        self.assertEqual(cfg.open_step,1/2000)
        self.assertEqual(cfg.max_tracking_delta,16/2000)
        self.assertEqual(cfg.arrival_tolerance,3/2000)
        self.assertEqual(cfg.slide_threshold,.6)
        self.assertEqual(cfg.max_prediction_age,.15)
        self.assertEqual(cfg.grip_settle_seconds,.06)

    def test_fixed_reference_and_explicit_override(self):
        limits={'min_raw':0,'max_raw':1150}
        cfg=CountSlipConfig(grip_reference=213)
        self.assertEqual(cfg.resolve(limits,197).grip_closure,213/1150)
        self.assertEqual(cfg.resolve(limits,197,grip_from_current=True).grip_closure,197/1150)
        with self.assertRaisesRegex(ValueError,'fixed encoder count'):
            CountSlipConfig().uses_current_position(False)

    def test_invalid_human_parameters_are_rejected(self):
        for change in ({'max_open_counts':True},{'max_open_counts':40.5},
                       {'max_open_counts':0},{'open_step_counts':17},
                       {'open_step_interval_s':float('nan')},
                       {'extra_observation_s':-1},{'max_prediction_age_ms':float('inf')},
                       {'slide_threshold_percent':101},{'slide_threshold_percent':0},
                       {'grip_reference':True},{'grip_reference':'197'},
                       {'session_margin_counts':-1},{'slide_confirm_frames':0}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                CountSlipConfig(**change)
        with self.assertRaisesRegex(ValueError,'Unknown count configuration field'):
            parse_slip_config({'format':'counts-v1','max_open_count':40})
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            parse_slip_config({'format':'counts-v2'})

    def test_near_lower_limit_and_bad_hardware_limits_are_rejected(self):
        for limits,start in (({'min_raw':0,'max_raw':1150},40),
                             ({'min_raw':100,'max_raw':2100},130),
                             ({'min_raw':0,'max_raw':0},197),
                             ({'min_raw':0,'max_raw':1150},1200)):
            with self.subTest(limits=limits,start=start),self.assertRaises(ValueError):
                CountSlipConfig().resolve(limits,start)

    def test_legacy_fractional_snapshot_remains_loadable(self):
        original=SlipConfig(grip_closure=213/1150)
        self.assertEqual(parse_slip_config(asdict(original)),original)

    def test_full_travel_then_timeout_and_slide_both_reclose(self):
        # Ideal feedback checks state transitions, not physical motor dynamics.
        cfg=CountSlipConfig().resolve({'min_raw':0,'max_raw':1150},197)
        for onset in (None,21.):
            with self.subTest(slide_at=onset):
                ctl=SlipController(cfg)
                closure=cfg.grip_closure
                targets=[]
                first_brake=None
                reached_min=None
                for i in range(2501):
                    t=i/100; now=1_000_000_000+i*10_000_000
                    sliding=onset is not None and t>=onset
                    p=SlipPrediction(i,now,now,'slide' if sliding else 'quiet','up',.99,.99,
                                     slide_probability=.9 if sliding else .01)
                    decision=ctl.step(now,GripperFeedback(now,closure),p)
                    if decision['command']:
                        closure=decision['command']['closure']
                        raw=round(closure*1150)
                        targets.append(raw)
                        if raw==157 and reached_min is None: reached_min=t
                    if decision['phase']=='BRAKING' and first_brake is None: first_brake=t
                    if ctl.phase in ctl.terminal: break
                self.assertEqual(min(targets),157)
                self.assertEqual(targets[-1],197)
                self.assertGreater(first_brake,reached_min)
                if onset is None:
                    self.assertEqual(ctl.reason,'no_slide_timeout')
                    self.assertGreaterEqual(first_brake,23)
                else:
                    self.assertEqual(ctl.phase,'FINISHED')
                    self.assertEqual(first_brake,onset)
                    self.assertIsNone(ctl.reason)

    def test_count_replay_requires_real_scale_and_keeps_first_feedback(self):
        cfg=CountSlipConfig()
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'replay'
            with self.assertRaisesRegex(ValueError,'gripper-limits'):
                replay(cfg,[],output)
            self.assertFalse(output.exists())
            rows=[{'now_ns':1_000_000_000,
                   'feedback':{'host_ns':1_000_000_000,'closure':197/1150},'snn':None}]
            replay(cfg,iter(rows),output,gripper_limits={'min_raw':0,'max_raw':1150})
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['config']['grip_closure'],197/1150)
            self.assertEqual(len((output/'trace.jsonl').read_text().splitlines()),1)


if __name__=='__main__': unittest.main()
