from io import StringIO
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from neurofusion.adapters.robotis import GripperSample
from neurofusion.demos.balloon_grip.__main__ import gripper_config_for_run, verify_startup_hold
from neurofusion.demos.balloon_grip.controller import Config, ContactGuard, GripOffsetRegulator, Regulator
from neurofusion.demos.balloon_grip.pressure import PressureReading, pressure_target
from neurofusion.demos.balloon_grip.tactile import ContactImage, RecentEvents


class PressureTargetTest(unittest.TestCase):
    def setUp(self):
        self.context = dict(now_ns=200_000_000, clock_us=110_000, sight_ns=180_000_000,
                            deadband=.045, lower=74, upper=80)

    def test_requires_sight_agreement_with_the_pressure_direction(self):
        release = PressureReading(100_000, 195_000_000, {}, 'close')
        press = PressureReading(100_000, 195_000_000, {}, 'open')
        self.assertEqual(pressure_target(release, ratio=.9, **self.context), ('close', 80))
        self.assertEqual(pressure_target(press, ratio=1.1, **self.context), ('open', 74))
        self.assertEqual(pressure_target(release, ratio=1.1, **self.context), ('sight_disagrees', None))
        self.assertEqual(pressure_target(press, ratio=.9, **self.context), ('sight_disagrees', None))

    def test_ambiguous_or_stale_readings_hold(self):
        uncertain = PressureReading(100_000, 195_000_000, {}, None)
        stale = PressureReading(50_000, 195_000_000, {}, 'close')
        self.assertEqual(pressure_target(uncertain, ratio=.9, **self.context), ('hold', None))
        self.assertEqual(pressure_target(stale, ratio=.9, **self.context), ('pressure_stale', None))
        fresh = PressureReading(100_000, 195_000_000, {}, 'close')
        context = {**self.context, 'sight_ns': 60_000_000}
        self.assertEqual(pressure_target(fresh, ratio=.9, **context), ('sight_stale', None))

    def test_pressure_uses_slip_recatch_position_drive_and_narrow_radius(self):
        args = SimpleNamespace(port='/dev/ttyUSB0', execute=True, contact_head=Path('contact.npz'),
                               pressure_checkpoint=Path('pressure.pt'), manual_current_limit=False)
        config = Config()
        pressure = gripper_config_for_run(args, config, radius=3)
        self.assertTrue(pressure.use_hardware_current_limit)
        self.assertEqual(pressure.position_profile, 'direct')
        self.assertEqual(pressure.session_radius_raw, 3)

        args.pressure_checkpoint = None
        sparsh = gripper_config_for_run(args, config, radius=4)
        self.assertFalse(sparsh.use_hardware_current_limit)
        self.assertEqual(sparsh.position_profile, 'bounded')
        self.assertEqual(sparsh.current_cap_a, .20)


class RegulationTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.regulator = Regulator(self.config, 2.0, 100, 150)

    def test_repeated_fast_pull_and_rebound(self):
        measured = goal = 125
        closes = opens = 0
        fps = 18.0

        # Six oscillations at 1.5 Hz, with limited encoder travel per frame
        # and contact depending on both stretch and jaw position.
        for tick in range(72):
            frame_s = tick / fps
            disturbance = -.33 * math.sin(2 * math.pi * 1.5 * frame_s)
            measured += min(1, max(-1, goal - measured))
            ratio = 1 + disturbance + (measured - 125) * .03
            decision = self.regulator.decide(frame_s, ratio, measured, goal)
            self.assertNotEqual(decision.reason, 'contact_lost')
            if decision.target_raw is None:
                continue
            goal = decision.target_raw
            self.regulator.last_action_s = frame_s
            self.assertLessEqual(abs(goal - measured), self.config.tracking_limit_counts)
            if decision.reason == 'close':
                self.assertGreater(goal, measured)
                closes += 1
            if decision.reason == 'open':
                self.assertLess(goal, measured)
                opens += 1
        self.assertGreaterEqual(closes, 6)
        self.assertGreaterEqual(opens, 6)

    def test_falling_contact_can_close_before_crossing_deadband(self):
        self.regulator.decide(1, 1.05, 125)
        decision = self.regulator.decide(1.05, .94, 125)
        self.assertEqual(decision.reason, 'close')
        self.assertEqual(decision.target_raw, 126)
        self.assertLess(decision.projected_ratio, .88)

    def test_rebound_cancels_pending_closing_while_contact_is_still_low(self):
        self.regulator.decide(1, .60, 125)
        decision = self.regulator.decide(1.05, .70, 125, goal=126)
        self.assertEqual(decision.reason, 'cancel_pending_step')
        self.assertEqual(decision.target_raw, 125)

    def test_rebound_can_open_before_crossing_deadband(self):
        self.regulator.decide(1, .93, 125)
        decision = self.regulator.decide(1.05, 1.06, 125, goal=127)
        self.assertEqual(decision.reason, 'open')
        self.assertEqual(decision.target_raw, 123)

    def test_long_gap_does_not_project_from_old_sample(self):
        self.regulator.decide(1, .76, 125)
        decision = self.regulator.decide(1.5, 1.0, 125)
        self.assertEqual(decision.reason, 'hold')
        self.assertEqual(decision.projected_ratio, 1.0)

    def test_contact_loss_and_limits(self):
        cases = (
            (.1, 125, None, 'contact_lost', None),
            (.8, 150, None, 'travel_limit', None),
            (1.4, 100, None, 'travel_limit', None),
            (.8, 125, 129, 'tracking_limit', None),
            (1.4, 125, 129, 'open', 123),
        )
        for ratio, measured, goal, reason, target in cases:
            with self.subTest(reason=reason, ratio=ratio, goal=goal):
                self.regulator.last_frame = None
                decision = self.regulator.decide(1, ratio, measured, goal)
                self.assertEqual((decision.reason, decision.target_raw), (reason, target))

    def test_rate_limit(self):
        self.regulator.last_action_s = 2
        self.assertEqual(self.regulator.decide(2.02, .8, 125).reason, 'rate_limit')
        self.assertEqual(self.regulator.decide(2.05, 1.4, 125).reason, 'open')


class GripOffsetTest(unittest.TestCase):
    def setUp(self):
        self.regulator = GripOffsetRegulator(Config(), 5.0, 100, 75, 115)

    def test_closes_on_loss_of_compression_and_opens_on_recovery_below_setpoint(self):
        measured = 100
        for tick in range(6):
            t = tick * .06
            decision = self.regulator.decide(t, .73, measured, measured)
            if decision.target_raw is not None:
                self.assertEqual(decision.reason, 'close')
                self.regulator.last_action_s = t
                measured = decision.target_raw
        self.assertEqual(measured, 103)
        recovered = self.regulator.decide(.40, .84, measured, measured)
        self.assertEqual(recovered.reason, 'open')
        self.assertLess(recovered.target_raw, measured)
        self.assertEqual(self.regulator.decide(.50, .1, 104, 104).reason, 'contact_lost')

    def test_does_not_accumulate_goals_beyond_tracking_or_trial_limits(self):
        decision = self.regulator.decide(1.0, .4, 99, 103)
        self.assertEqual(decision.reason, 'tracking_limit')
        decision = self.regulator.decide(1.2, 2.0, 95, 95)
        self.assertEqual(decision.target_raw, 96)
        self.assertIsNone(self.regulator.decide(1.4, 2.0, 94, 94).target_raw)

    def test_hysteresis_keeps_the_goal_near_a_rounding_boundary(self):
        for tick, ratio in enumerate((.735, .739, .735, .739)):
            decision = self.regulator.decide(tick, ratio, 103, 103)
            self.assertEqual(decision.reason, 'hold')

    def test_event_activity_extends_projection_in_the_existing_direction(self):
        plain = self.regulator
        assisted = GripOffsetRegulator(Config(), 5.0, 100, 75, 115)
        plain.decide(1., 1., 100)
        assisted.decide(1., 1., 100)
        p = plain.decide(1.06, .95, 100)
        a = assisted.decide(1.06, .95, 100, motion=True)
        self.assertLess(a.projected_ratio, p.projected_ratio)
        self.assertEqual(a.reason, 'close')


class FastGripTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'demos' / 'balloon-grip' / 'configs' / 'fast-close.json'
        config = Config(**json.loads(path.read_text()))
        self.regulator = GripOffsetRegulator(config, 5.0, 46, 40, 54)

    def test_five_count_close_and_limited_travel(self):
        c = self.regulator
        c.decide(1., 1., 46, 46)
        decision = c.decide(1.055, .938, 46, 46)
        self.assertEqual((decision.reason, decision.target_raw), ('close', 51))
        self.assertEqual(c.decide(1.12, .84, 51, 51).target_raw, 54)
        self.assertIsNone(c.decide(1.19, .81, 54, 54).target_raw)

    def test_weak_optical_contact_assists_closing_before_loss(self):
        c = self.regulator
        c.decide(1., 1., 46, 46, guard_ratio=1.)
        decision = c.decide(1.06, .97, 46, 46, guard_ratio=.75)
        self.assertEqual((decision.reason, decision.target_raw), ('close', 51))
        self.assertEqual(c.decide(1.12, .95, 51, 51, guard_ratio=.1).reason, 'contact_lost')

    def test_near_grip_jitter_holds_and_recovery_allows_opening(self):
        c = self.regulator
        c.decide(1., .833, 48, 48)
        self.assertEqual(c.decide(1.06, .856, 54, 54).reason, 'hold')
        restored = c.decide(1.12, .95, 54, 54)
        self.assertEqual((restored.reason, restored.target_raw), ('open', 52))
        c.decide(1.18, .95, 52, 52)
        self.assertEqual(c.decide(1.24, .95, 50, 50).target_raw, 48)


class BurstCloseTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        path = root / 'demos' / 'balloon-grip' / 'configs' / 'fast-close-20.json'
        self.config = Config(**json.loads(path.read_text()))

    def test_recorded_drop_requests_twenty_count_close_before_contact_loss(self):
        guard = ContactGuard(self.config, 2.6907)
        controller = GripOffsetRegulator(self.config, 4.415, 50, 46, 72)
        guard.observe(4_676_000_000, 2.6907 * 1.004)
        controller.decide(4.676, .993, 50, 50, guard_ratio=1.004)
        ratio, burst, lost = guard.observe(4_728_000_000, 2.6907 * .680)
        self.assertTrue(burst)
        self.assertFalse(lost)
        self.assertEqual(guard.observe(4_728_000_000, 2.6907 * .680), (ratio, burst, lost))
        decision = controller.decide(4.728, .993, 50, 50, guard_ratio=ratio, burst=burst)
        self.assertEqual((decision.reason, decision.target_raw), ('close', 70))
        _, burst, lost = guard.observe(4_788_000_000, 2.6907 * .432)
        self.assertTrue(lost)
        self.assertFalse(burst)

    def test_optical_assist_stays_active_during_small_rebound(self):
        controller = GripOffsetRegulator(self.config, 4.415, 50, 46, 72)
        controller.decide(3.520, .957, 57, 58, guard_ratio=.790)
        decision = controller.decide(3.581, .964, 57, 58, guard_ratio=.814)
        self.assertNotEqual(decision.reason, 'open')
        recovered = controller.decide(3.650, 1.05, 58, 58, guard_ratio=.98)
        self.assertEqual(recovered.reason, 'open')


class StartupTest(unittest.TestCase):
    def test_startup_drift_is_rejected_before_any_control_command(self):
        origin = 140
        device = Mock()
        device.arm.return_value = GripperSample(
            1, origin, .12, 0, 0, 23.4, 42, False, True, 0, origin,
        )
        device.read.return_value = GripperSample(
            2, origin - 4, .12, -8, -.1, 23.4, 42, True, True, 0, origin,
        )
        with self.assertRaisesRegex(RuntimeError, 'drifted during startup'):
            verify_startup_hold(device, origin, StringIO())
        device.command.assert_not_called()


class SensorTest(unittest.TestCase):
    def test_contact_image_rejects_uniform_brightness_shift(self):
        unloaded = np.full((144, 192, 3), 100, dtype=np.uint8)
        sensor = ContactImage(unloaded)
        shifted = unloaded + 10
        self.assertEqual(sensor.score(shifted), 0)
        shifted[40:90, 50:90] += 20
        weak = sensor.score(shifted)
        shifted[40:90, 50:90] += 20
        self.assertGreater(weak, 0)
        self.assertGreater(sensor.score(shifted), weak)

    def test_event_window_ages_out_even_when_dvs_is_silent(self):
        events = RecentEvents(150)
        first = np.array([[0, 0, 0, p] for p in (0, 1, 0, 1)])
        second = np.array([[0, 0, 0, p] for p in (0, 1, 0)])
        empty = np.empty((0, 4))
        self.assertEqual(events.update(10, first), (4, 2))
        self.assertEqual(events.update(100, second), (7, 3))
        self.assertEqual(events.update(160, empty), (3, 1))
        self.assertEqual(events.update(250, empty), (0, 0))
        self.assertFalse(events.sites)

    def test_repeated_event_from_one_pixel_is_only_one_motion_site(self):
        events = RecentEvents(100)
        events.update(10, np.array([[7, 82, 0, 1]]*5))
        self.assertEqual(len(events.sites), 1)
        events.update(65, np.array([[7, 82, 0, 0], [19, 20, 0, 0]]))
        self.assertEqual(len(events.sites), 2)
        events.update(120, np.empty((0, 4)))
        self.assertEqual(len(events.sites), 2)
        events.update(170, np.empty((0, 4)))
        self.assertFalse(events.sites)


if __name__ == '__main__':
    unittest.main()
