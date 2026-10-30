from typing import Iterable, Protocol
from ..contracts import ActionProposal, EventPacket, ImageSample, Message


class CameraSource(Protocol):
    def samples(self) -> Iterable[ImageSample]: ...


class EventSource(Protocol):
    def packets(self) -> Iterable[EventPacket]: ...


class RobotFeedbackSource(Protocol):
    def observations(self) -> Iterable[Message]: ...


class ActionSink(Protocol):
    def publish(self, action: ActionProposal) -> None: ...


class ShadowSink:
    """Collect model proposals only; never acknowledge or send robot commands."""
    def __init__(self):
        self.proposals: list[ActionProposal] = []

    def publish(self, action: ActionProposal):
        self.proposals.append(action)
