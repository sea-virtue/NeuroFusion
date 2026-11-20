"""Regulate grip during repeated stretching and rebound of a weighted balloon."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    max_open_counts: int = 25
    max_close_counts: int = 15
    open_step_counts: int = 2
    close_step_counts: int = 1
    tracking_limit_counts: int = 4
    current_cap_a: float = 0.20
    min_command_interval_s: float = 0.035
    lookahead_s: float = 0.060
    max_projection_delta: float = 0.20
    startup_s: float = 2.0
    min_startup_score: float = 1.2
    deadband_fraction: float = 0.12
    loss_fraction: float = 0.20
    max_sensor_age_s: float = 0.25
    motion_window_s: float = 0.120
    gelneuro_min_sites: int = 3
    network_gain_counts: float = 16.0
    network_deadband_fraction: float = 0.045
    network_hysteresis_counts: float = 0.75
    network_max_offset_counts: int = 4
    network_max_close_offset_counts: int | None = None
    network_close_gain_counts: float | None = None
    network_recovery_fraction: float = 0.0
    network_guard_assist_fraction: float = 0.0
    network_guard_burst_drop_fraction: float = 0.0

    @property
    def network_close_limit(self):
        if self.network_max_close_offset_counts is None:
            return self.network_max_offset_counts
        return self.network_max_close_offset_counts

    def __post_init__(self):
        integer_fields = (
            'max_open_counts', 'max_close_counts', 'open_step_counts',
            'close_step_counts', 'tracking_limit_counts', 'gelneuro_min_sites',
            'network_max_offset_counts',
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        number_fields = (
            'current_cap_a', 'min_command_interval_s', 'lookahead_s',
            'max_projection_delta', 'startup_s', 'min_startup_score',
            'deadband_fraction', 'loss_fraction', 'max_sensor_age_s', 'motion_window_s',
            'network_gain_counts', 'network_deadband_fraction', 'network_hysteresis_counts',
        )
        for name in number_fields:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.network_close_limit) is not int or self.network_close_limit <= 0:
            raise ValueError('network_max_close_offset_counts must be a positive integer')
        gain = self.network_close_gain_counts
        if gain is not None and (type(gain) not in (int, float) or not math.isfinite(gain) or gain <= 0):
            raise ValueError('network_close_gain_counts must be finite and positive')
        for name in ('network_recovery_fraction', 'network_guard_assist_fraction',
                     'network_guard_burst_drop_fraction'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value < 1:
                raise ValueError(f'{name} must be in [0,1)')
        if max(self.open_step_counts, self.close_step_counts) > self.tracking_limit_counts:
            raise ValueError('Steps must fit inside the tracking limit')
        if not 0 < self.loss_fraction < 1 - self.deadband_fraction < 1:
            raise ValueError('Expected 0 < loss_fraction < 1 - deadband_fraction < 1')
        if self.network_deadband_fraction >= 1 - self.loss_fraction:
            raise ValueError('Invalid network deadband for configured contact loss')
        if self.current_cap_a > .30:
            raise ValueError('Balloon grip current cap must be at most 0.30 A')
        if self.lookahead_s > self.max_sensor_age_s or self.max_projection_delta >= 1:
            raise ValueError('Keep the projection within the sensor age and one baseline unit')


class ContactGuard:
    """Use each distinct optical frame once when looking for a rapid fall."""

    def __init__(self, config, baseline):
        self.config = config
        self.baseline = baseline
        self.previous = None
        self.result = None

    def observe(self, stamp, score):
        if self.previous is not None and stamp == self.previous[0]:
            return self.result
        c = self.config
        ratio = score / self.baseline
        lost = score <= max(c.min_startup_score, c.loss_fraction * self.baseline)
        burst = False
        if not lost and c.network_guard_burst_drop_fraction and self.previous is not None:
            old_stamp, old_ratio = self.previous
            burst = (0 < stamp - old_stamp <= c.max_sensor_age_s * 1e9
                     and ratio < c.network_guard_assist_fraction
                     and old_ratio - ratio >= c.network_guard_burst_drop_fraction)
        self.previous = stamp, ratio
        self.result = ratio, burst, lost
        return self.result


@dataclass(frozen=True)
class Decision:
    reason: str
    projected_ratio: float
    target_raw: int | None = None


class Regulator:
    """Project brief contact changes and issue bounded position steps."""

    def __init__(self, config, setpoint, lower, upper):
        self.config = config
        self.setpoint = setpoint
        self.lower = lower
        self.upper = upper
        self.last_action_s = -math.inf
        self.last_frame = None

    def project(self, frame_s, ratio, motion=False):
        c = self.config
        delta = 0.0
        if self.last_frame is not None:
            previous_s, previous_ratio = self.last_frame
            elapsed = frame_s - previous_s
            if 0 < elapsed <= c.max_sensor_age_s:
                horizon = min(c.max_sensor_age_s, c.lookahead_s * (1.5 if motion else 1.0))
                estimate = (ratio - previous_ratio) * horizon / elapsed
                delta = min(c.max_projection_delta, max(-c.max_projection_delta, estimate))
        self.last_frame = frame_s, ratio
        return ratio + delta, delta

    def decide(self, frame_s, ratio, measured, goal=None):
        c = self.config
        projected, delta = self.project(frame_s, ratio)
        low = 1 - c.deadband_fraction
        high = 1 + c.deadband_fraction
        reversal = c.deadband_fraction / 2

        if ratio <= c.loss_fraction:
            return Decision('contact_lost', projected)
        if ratio > 1 + 2 * c.deadband_fraction or (
            ratio >= low and projected > high and delta >= -reversal
        ):
            step, direction = -c.open_step_counts, 'open'
        elif ratio <= high and projected < low and delta <= reversal:
            step, direction = c.close_step_counts, 'close'
        elif (
            goal is not None
            and (goal - measured) * delta > 0
            and abs(delta) > reversal
            and self.lower <= measured <= self.upper
        ):
            step, direction = 0, 'cancel_pending_step'
        else:
            return Decision('hold', projected)

        if frame_s - self.last_action_s < c.min_command_interval_s:
            return Decision('rate_limit', projected)
        if step == 0:
            return Decision(direction, projected, measured)
        previous = measured if goal is None else goal
        # Discard the pending step in the opposite direction on phase reversal.
        base = previous if (previous - measured) * step >= 0 else measured
        target = min(self.upper, max(self.lower, base + step))
        if target == previous or (target - measured) * step <= 0:
            return Decision('travel_limit', projected)
        if abs(target - measured) > c.tracking_limit_counts:
            return Decision('tracking_limit', projected)
        return Decision(direction, projected, target)


class GripOffsetRegulator(Regulator):
    """Return towards the initial jaw position as contact recovers."""

    def __init__(self, config, setpoint, origin, lower, upper):
        lower = max(lower, origin - config.network_max_offset_counts)
        upper = min(upper, origin + config.network_close_limit)
        super().__init__(config, setpoint, lower, upper)
        self.origin = origin
        self.weakest_ratio = 1.0
        self.guard_assisting = False

    def decide(self, frame_s, ratio, measured, goal=None, *, motion=False, guard_ratio=None, burst=False):
        c = self.config
        if guard_ratio is not None:
            threshold = c.network_guard_assist_fraction
            if self.guard_assisting and c.network_guard_burst_drop_fraction:
                threshold += 0.10
            self.guard_assisting = guard_ratio < threshold
            if self.guard_assisting:
                ratio = min(ratio, guard_ratio)
        projected, _ = self.project(frame_s, ratio, motion)
        if ratio <= c.loss_fraction:
            return Decision('contact_lost', projected)
        error = 1 - projected
        correction = math.copysign(max(0.0, abs(error) - c.network_deadband_fraction), error)
        gain = c.network_gain_counts
        if correction > 0 and c.network_close_gain_counts is not None:
            gain = c.network_close_gain_counts
        requested = self.origin + gain * correction
        if burst:
            requested = max(requested, measured + c.close_step_counts)
        bounded = min(self.upper, max(self.lower, requested))
        desired = round(bounded)
        previous = measured if goal is None else goal
        if previous <= self.origin or ratio >= 1:
            self.weakest_ratio = ratio
        else:
            self.weakest_ratio = min(self.weakest_ratio, ratio)
        if desired < previous and previous > self.origin and ratio < 1 - c.network_deadband_fraction:
            if ratio - self.weakest_ratio < c.network_recovery_fraction:
                return Decision('hold', projected)
        if desired == previous or abs(bounded - previous) < c.network_hysteresis_counts:
            return Decision('hold', projected)
        if frame_s - self.last_action_s < c.min_command_interval_s:
            return Decision('rate_limit', projected)

        if (previous - measured) * (desired - previous) < 0:
            # Replace a pending goal on rebound from the actual position.
            base = measured
        else:
            base = previous
        step = min(c.close_step_counts, max(-c.open_step_counts, desired - base))
        target = base + step
        if burst and target > measured + c.tracking_limit_counts:
            target = measured + c.tracking_limit_counts
        if (target - previous) * (desired - previous) <= 0:
            return Decision('hold', projected)
        if not self.lower <= target <= self.upper or abs(target - measured) > c.tracking_limit_counts:
            return Decision('tracking_limit', projected)
        return Decision('close' if target > previous else 'open', projected, target)
