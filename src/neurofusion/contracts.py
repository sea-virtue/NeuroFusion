"""Clock and message boundaries; no implicit device/host clock conversion."""
from dataclasses import dataclass
from enum import IntEnum

Vector = tuple[float, ...]
ACTION_NAMES = ('dx_m', 'dy_m', 'dz_m', 'dyaw_rad', 'dgripper_width_m')


class Kind(IntEnum):
    # Equal availability time: queued context/state/observations precede control.
    CONTEXT = 0
    ROBOT_STATE = 1
    EXECUTED_ACTION = 2
    FRAME = 3
    EVENT = 4
    CONTROL = 5


@dataclass(frozen=True)
class Stamp:
    # One host monotonic clock domain; relative nanoseconds in the synthetic demo.
    available_ns: int
    received_ns: int | None = None
    device_us: int | None = None
    clock_id: str = 'host_monotonic'

    def __post_init__(self):
        if type(self.available_ns) is not int or self.available_ns < 0:
            raise ValueError('available_ns must be a nonnegative integer')
        if self.clock_id != 'host_monotonic':
            raise ValueError('Map scheduling into host_monotonic explicitly first')
        if self.received_ns is not None and not 0 <= self.received_ns <= self.available_ns:
            raise ValueError('received_ns must be no later than available_ns')
        if self.device_us is not None and self.device_us < 0:
            raise ValueError('device_us must be nonnegative')


@dataclass(frozen=True)
class ImageSample:
    stamp: Stamp
    image: object
    source: str  # scene_rgb or gelsight; image layout/calibration to be defined.


@dataclass(frozen=True)
class EventPacket:
    stamp: Stamp
    events: object
    device_start_us: int | None = None
    device_end_us: int | None = None


@dataclass(frozen=True)
class Message:
    kind: Kind
    stamp: Stamp
    feature: Vector = ()

    def __post_init__(self):
        if not isinstance(self.kind, Kind):
            raise TypeError('kind must be a Kind')
        object.__setattr__(self, 'feature', tuple(self.feature))


@dataclass(frozen=True)
class Snapshot:
    memory: Vector
    context: Vector
    robot: Vector
    executed_action: Vector
    time_ns: int
    revision: int = 0
    last_frame_ns: int | None = None
    last_event_ns: int | None = None


@dataclass(frozen=True)
class UpdateTiming:
    now_ns: int
    elapsed_s: float
    frame_age_s: float | None
    event_age_s: float | None


@dataclass(frozen=True)
class ActionProposal:
    time_ns: int
    memory_revision: int
    values: Vector
    # A proposal is NOT confirmation that a robot received or executed an action.
    source: str = 'unified_action_head'
