from typing import Protocol
from ..contracts import EventPacket, ImageSample, Snapshot, UpdateTiming, Vector


class ImageEncoder(Protocol):
    def encode(self, sample: ImageSample) -> Vector: ...


class EventEncoder(Protocol):
    def encode(self, packet: EventPacket) -> Vector: ...


class FrameCorrection(Protocol):
    def correct(self, memory: Vector, feature: Vector, context: Vector) -> Vector: ...


class GatedEventUpdate(Protocol):
    def update(self, memory: Vector, feature: Vector, context: Vector,
               robot: Vector, timing: UpdateTiming) -> Vector: ...


class ActionTimePropagation(Protocol):
    def propagate(self, memory: Vector, robot: Vector, executed_action: Vector,
                  timing: UpdateTiming) -> Vector: ...


class UnifiedActionHead(Protocol):
    def predict(self, state: Snapshot) -> Vector: ...
