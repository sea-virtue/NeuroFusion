"""Human-edited encoder counts resolved against measured hardware limits."""
from dataclasses import dataclass
import math

from .slip import SlipConfig


@dataclass(frozen=True)
class CountSlipConfig:
    format: str = 'counts-v1'
    grip_reference: str | int = 'current'
    max_open_counts: int = 40
    open_step_counts: int = 1
    open_step_interval_s: float = .5
    extra_observation_s: float = 3.
    slide_threshold_percent: float = 60.
    expected_direction: str = 'up'
    # Advanced options; ordinary changes need only the fields above.
    tracking_limit_counts: int = 16
    arrival_tolerance_counts: float = 3.
    session_margin_counts: int = 5
    grip_settle_ms: float = 60.
    brake_timeout_s: float = .8
    max_prediction_age_ms: float = 150.
    max_feedback_age_ms: float = 150.
    max_tick_gap_ms: float = 100.
    slide_confirm_frames: int = 1

    def __post_init__(self):
        if self.format != 'counts-v1':
            raise ValueError('Unsupported slip configuration format')
        if self.grip_reference != 'current' and (
                type(self.grip_reference) is not int or self.grip_reference <= 0):
            raise ValueError('grip_reference must be "current" or a positive encoder count')
        for name in ('max_open_counts','open_step_counts','tracking_limit_counts','slide_confirm_frames'):
            value = getattr(self,name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.session_margin_counts) is not int or self.session_margin_counts < 0:
            raise ValueError('session_margin_counts must be a nonnegative integer')
        for name in ('open_step_interval_s','extra_observation_s','slide_threshold_percent',
                     'arrival_tolerance_counts','grip_settle_ms','brake_timeout_s',
                     'max_prediction_age_ms','max_feedback_age_ms','max_tick_gap_ms'):
            value = getattr(self,name)
            zero_allowed = name in ('extra_observation_s','grip_settle_ms')
            if (type(value) not in (int,float) or not math.isfinite(value)
                    or (value < 0 if zero_allowed else value <= 0)):
                raise ValueError(f'Invalid {name}')
        if self.slide_threshold_percent > 100:
            raise ValueError('slide_threshold_percent must be in (0,100]')
        if self.expected_direction not in ('up','down','left','right'):
            raise ValueError('Invalid expected_direction')
        if self.open_step_counts > min(self.max_open_counts,self.tracking_limit_counts):
            raise ValueError('open_step_counts exceeds release or tracking limit')

    @property
    def slide_seconds(self):
        return 0.

    @property
    def slide_threshold(self):
        return self.slide_threshold_percent / 100.

    @property
    def max_prediction_age(self):
        return self.max_prediction_age_ms / 1000.

    @property
    def release_timeout(self):
        steps = (self.max_open_counts+self.open_step_counts-1)//self.open_step_counts
        return steps*self.open_step_interval_s+self.extra_observation_s

    @property
    def gripper_session_radius_raw(self):
        return self.max_open_counts+self.session_margin_counts

    def uses_current_position(self, override=None):
        if override is None:
            return self.grip_reference == 'current'
        if not override and self.grip_reference == 'current':
            raise ValueError('Set grip_reference to a fixed encoder count before using --no-grip-from-current')
        return override

    def resolve(self, limits, startup_position_raw, *, grip_from_current=None):
        """Resolve once before arming; fractions use actual register limits."""
        low,high = limits['min_raw'],limits['max_raw']
        if high <= low or not low <= startup_position_raw <= high:
            raise ValueError('Invalid measured gripper position or hardware limits')
        use_current = self.uses_current_position(grip_from_current)
        reference = startup_position_raw if use_current else self.grip_reference
        if not low < reference <= high:
            raise ValueError('grip_reference is outside the hardware position limits')
        if reference-self.max_open_counts <= low:
            raise ValueError('Requested opening range reaches the hardware lower limit; reduce max_open_counts')
        span = high-low
        return SlipConfig(
            grip_closure=(reference-low)/span, expected_direction=self.expected_direction,
            confidence=.6, slide_probability_threshold=self.slide_threshold,
            slide_confirm_frames=self.slide_confirm_frames,
            max_prediction_age=self.max_prediction_age,
            max_feedback_age=self.max_feedback_age_ms/1000., max_tick_gap=self.max_tick_gap_ms/1000.,
            grip_settle_seconds=self.grip_settle_ms/1000.,
            release_timeout=self.release_timeout, brake_timeout=self.brake_timeout_s,
            max_open_delta=self.max_open_counts/span,
            max_tracking_delta=self.tracking_limit_counts/span,
            arrival_tolerance=self.arrival_tolerance_counts/span,
            open_step=self.open_step_counts/span, open_step_interval=self.open_step_interval_s,
            gripper_session_radius_raw=self.gripper_session_radius_raw)

    def describe(self, *, grip_from_current=None):
        reference = '启动实测位置' if self.uses_current_position(grip_from_current) else str(self.grip_reference)
        return (f'松爪配置：基准 {reference}；最多 {self.max_open_counts} 个计数；'
                f'每 {self.open_step_interval_s:g} 秒松 {self.open_step_counts} 个计数；'
                f'超时 {self.release_timeout:g} 秒；会话范围 ±{self.gripper_session_radius_raw} 个计数。')


def parse_slip_config(settings):
    """Keep legacy fractional configs and recorded session snapshots loadable."""
    if not isinstance(settings,dict):
        raise ValueError('Slip configuration must be a JSON object')
    if 'format' not in settings:
        return SlipConfig(**settings)
    if settings['format'] != 'counts-v1':
        raise ValueError(f'Unsupported slip configuration format: {settings["format"]!r}')
    try:
        return CountSlipConfig(**settings)
    except TypeError as exc:
        raise ValueError(f'Unknown count configuration field: {exc}') from exc
