"""Single-thread reference sequencer, NOT a real-time scheduler.

Workers must finish encoding before publishing a feature's availability time.
Live workers, synchronization, overload recovery and deadline enforcement are pending.
"""
from dataclasses import replace
import heapq
import threading

from ..config import Config, vector
from ..contracts import ActionProposal, Kind, Message, Snapshot, UpdateTiming
from ..models.interfaces import ActionTimePropagation, FrameCorrection, GatedEventUpdate, UnifiedActionHead
from ..adapters.interfaces import ActionSink


class Engine:
    def __init__(self, config: Config, frame: FrameCorrection, event: GatedEventUpdate,
                 propagation: ActionTimePropagation, head: UnifiedActionHead,
                 sink: ActionSink, trace=None):
        self.config = config
        self._frame, self._event = frame, event
        self._propagation, self._head, self._sink = propagation, head, sink
        self._trace = trace or (lambda entry: None)
        self._owner = threading.get_ident()
        self._queue = []
        self._sequence = 0
        self._running = False
        self._processed_ns = -1
        self._state = Snapshot((0.,) * config.memory_dim, (0.,) * config.context_dim,
                               (0.,) * config.robot_dim, (0.,) * config.action_dim, 0)

    @property
    def state(self):
        return self._state  # Frozen snapshot; tuples cannot mutate the writer's state.

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError('Only the owning thread may submit or update state')

    def submit(self, message: Message):
        self._check_owner()
        if self._running:
            raise RuntimeError('Components cannot reenter or publish into a running update')
        if message.stamp.available_ns < self._processed_ns:
            raise ValueError('Late availability time: do not rewind state or sort by device clocks')
        dimensions = {Kind.CONTEXT: self.config.context_dim, Kind.ROBOT_STATE: self.config.robot_dim,
                      Kind.EXECUTED_ACTION: self.config.action_dim, Kind.FRAME: self.config.frame_feature_dim,
                      Kind.EVENT: self.config.event_feature_dim, Kind.CONTROL: 0}
        checked = replace(message, feature=vector(message.feature, dimensions[message.kind], message.kind.name))
        if len(self._queue) >= self.config.max_pending_messages:
            raise OverflowError('Input queue full; live coalescing/drop policy is not implemented')
        heapq.heappush(self._queue, (checked.stamp.available_ns, int(checked.kind), self._sequence, checked))
        self._sequence += 1

    def run_until(self, until_ns: int | None = None):
        self._check_owner()
        if self._running:
            raise RuntimeError('State writer is not reentrant')
        self._running = True
        try:
            while self._queue and (until_ns is None or self._queue[0][0] <= until_ns):
                _, _, _, message = heapq.heappop(self._queue)
                self._process(message)
        finally:
            self._running = False

    def _process(self, message):
        before = self._state
        now = message.stamp.available_ns
        timing = UpdateTiming(now, (now - before.time_ns) / 1e9,
                              None if before.last_frame_ns is None else (now - before.last_frame_ns) / 1e9,
                              None if before.last_event_ns is None else (now - before.last_event_ns) / 1e9)
        h = vector(self._propagation.propagate(before.memory, before.robot, before.executed_action, timing),
                   self.config.memory_dim, 'propagated memory')
        state = replace(before, memory=h, time_ns=now, revision=before.revision + 1)
        kind = message.kind
        if kind == Kind.CONTEXT:
            state = replace(state, context=message.feature)
        elif kind == Kind.ROBOT_STATE:
            state = replace(state, robot=message.feature)
        elif kind == Kind.EXECUTED_ACTION:
            state = replace(state, executed_action=message.feature)
        elif kind == Kind.FRAME:
            h = self._frame.correct(state.memory, message.feature, state.context)
            state = replace(state, memory=vector(h, self.config.memory_dim, 'frame memory'), last_frame_ns=now)
        elif kind == Kind.EVENT:
            h = self._event.update(state.memory, message.feature, state.context, state.robot, timing)
            state = replace(state, memory=vector(h, self.config.memory_dim, 'event memory'), last_event_ns=now)
        # Compute and validate before publishing the new snapshot.
        action = None
        if kind == Kind.CONTROL:
            values = vector(self._head.predict(state), self.config.action_dim, 'action')
            action = ActionProposal(now, state.revision, values)
        self._state = state
        self._processed_ns = now
        self._trace({'kind': kind.name, 'available_ns': now, 'revision': state.revision,
                     'last_frame_ns': state.last_frame_ns, 'last_event_ns': state.last_event_ns,
                     'received_ns': message.stamp.received_ns, 'device_us': message.stamp.device_us,
                     'action': None if action is None else list(action.values)})
        if action is not None:
            self._sink.publish(action)
