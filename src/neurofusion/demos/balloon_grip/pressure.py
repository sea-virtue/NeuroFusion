"""Run the balloon pressure head on the same raw DVS windows as slip-recatch."""

from collections import deque
from dataclasses import dataclass

from ..slip_recatch.live import EventWindows


@dataclass(frozen=True)
class PressureReading:
    end_us: int
    captured_ns: int
    probabilities: dict[str, float]
    direction: str | None


def pressure_target(reading, *, now_ns, clock_us, sight_ns, ratio, deadband, lower, upper):
    if reading is None or reading.direction is None:
        return 'hold', None
    if not (0 <= now_ns - reading.captured_ns <= 50_000_000 and 0 <= clock_us - reading.end_us <= 50_000):
        return 'pressure_stale', None
    if not 0 <= now_ns - sight_ns <= 120_000_000:
        return 'sight_stale', None
    if reading.direction == 'open' and ratio >= 1 + deadband:
        return 'open', lower
    if reading.direction == 'close' and ratio <= 1 - deadband:
        return 'close', upper
    return 'sight_disagrees', None


class PressureModel:
    classes = ('quiet', 'press', 'release')

    def __init__(self, checkpoint, confidence, release_confidence):
        import torch
        from ...models.gelneuro_unified_snn import GelNeuroUnifiedSNN

        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('kind') != 'unified-iaf-pressure' or tuple(saved['classes']) != self.classes:
            raise ValueError('Expected a GelNeuro quiet/press/release checkpoint')
        self.model = GelNeuroUnifiedSNN.from_checkpoint(saved).eval()
        self.bin_us = self.model.bin_us
        self.windows = EventWindows(self.bin_us, self.bin_us)
        self.confidence = confidence
        self.release_confidence = release_confidence
        self.recent_counts = deque(maxlen=3)
        self.reset()

    def reset(self):
        self.state = None
        self.recent_counts.clear()
        self.previous = self.latest = None

    def update(self, received_ns, events, watermark_us):
        self.windows.push(events, received_ns)
        if watermark_us is not None:
            self.windows.advance_watermark(watermark_us, received_ns)
        for values, begin, end, captured in self.windows.ready():
            yield self.predict(values, begin, end, captured)

    def predict(self, events, begin, end, captured):
        import torch
        from ...models.snn_adapter import count_bins

        with torch.inference_mode():
            counts = count_bins([events], begin, end, self.bin_us, 'cpu')
            task, _, self.state = self.model(counts, self.state)
            quiet, press, release = task[0, -1].softmax(-1).tolist()
        self.recent_counts.append(len(events))
        direction = None
        if sum(self.recent_counts) >= 3:
            if press >= self.confidence and press - max(quiet, release) >= .15:
                direction = 'open'
            elif release >= self.release_confidence and release > max(quiet, press):
                direction = 'close'
        confirmed = direction if direction == self.previous else None
        self.previous = direction
        self.latest = PressureReading(end, captured, dict(quiet=quiet, press=press, release=release), confirmed)
        return self.latest
