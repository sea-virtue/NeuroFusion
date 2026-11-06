"""Versioned on-disk format and operator-supplied trial conditions."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math

FORMAT_VERSION = 1
EVENT_DTYPE = "<i8"
EVENT_COLUMNS = ["x", "y", "timestamp_us", "polarity"]
TASKS = ("background", "static_contact", "slide", "start_stop", "reverse",
         "press_release", "rotate", "mixed", "diagnostic")
DIRECTIONS = ("none", "up", "down", "left", "right")


@dataclass
class NativeFrame:
    """One unmodified camera-produced JPEG, before host decoding."""
    payload: bytes
    width: int
    height: int


@dataclass
class TrialLabels:
    task: str = "diagnostic"
    object_id: str = "unrecorded"
    direction: str = "none"
    speed_level: str = "unmeasured"
    contact_level: str = "unmeasured"
    repetition: int = 1
    material: str = "unrecorded"
    direction_convention: str = "unrecorded; define physical directions before training"
    mounting_notes: str = "unrecorded; surfaces need not observe the same contact patch"
    notes: str = ""

    def validate(self):
        if self.task not in TASKS or self.direction not in DIRECTIONS:
            raise ValueError("Unknown task/direction")
        if self.repetition < 1:
            raise ValueError("repetition must be >= 1")
        if self.task == "slide" and self.direction == "none":
            raise ValueError("slide requires an operator-defined direction")


@dataclass
class CaptureConfig:
    root: str = "data/raw/tactile"
    output: str | None = None
    seconds: float = 8.0
    lead_in: float = 2.0
    tail_out: float = 2.0
    gelsight: bool = True
    camera: int = 0
    poll_ms: float = 5.0
    frame_queue: int = 8
    event_queue: int = 256
    startup_timeout: float = 45.0
    labels: TrialLabels = field(default_factory=TrialLabels)

    def validate(self):
        self.labels.validate()
        for name in ("seconds", "lead_in", "tail_out", "poll_ms", "startup_timeout"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.seconds <= 0 or self.poll_ms <= 0 or self.startup_timeout <= 0:
            raise ValueError("seconds, poll_ms and startup_timeout must be positive")
        if min(self.lead_in, self.tail_out) < 0 or self.lead_in + self.tail_out >= self.seconds:
            raise ValueError("lead-in + tail-out must be shorter than the trial")
        if min(self.frame_queue, self.event_queue) < 1 or self.camera < 0:
            raise ValueError("queues must be >= 1 and camera >= 0")

    def to_dict(self):
        return asdict(self)


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("File reference escapes the recording directory")
    return path


def instruction_schedule(config):
    """Operator prompts only; their timestamps are not verified motion labels."""
    task = config.labels.task
    if task == "background":
        return [(0.0, "NO CONTACT / HOLD STILL")]
    if task in ("static_contact", "diagnostic"):
        return [(0.0, "HOLD STILL")]
    start, end = config.lead_in, config.seconds - config.tail_out
    span = end - start
    prompts = [(0.0, "HOLD STILL")]
    if task == "start_stop":
        prompts += [(start, "MOVE"), (start + span / 3, "STOP MOTION"), (start + 2 * span / 3, "MOVE AGAIN")]
    elif task == "reverse":
        prompts += [(start, "MOVE IN INITIAL DIRECTION"), (start + span / 2, "REVERSE DIRECTION")]
    elif task == "press_release":
        prompts += [(start, "PRESS WITHOUT SLIDING"), (start + span / 4, "HOLD CONTACT"), (start + 3 * span / 4, "RELEASE CONTACT")]
    else:
        prompts += [(start, f"DO {task.upper()} / {config.labels.direction.upper()}")]
    return prompts + [(end, "STOP MOTION / HOLD STILL")]
