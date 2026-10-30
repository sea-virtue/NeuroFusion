"""SYNTHETIC wiring demonstration. No learned model, hardware, or latency benchmark."""
from dataclasses import asdict
from .adapters.interfaces import ShadowSink
from .config import Config
from .contracts import ACTION_NAMES, Kind, Message, Stamp
from .runtime.clock import control_ticks
from .runtime.engine import Engine


class ToyFrameCorrection:
    def correct(self, memory, feature, context):
        return (feature[0], *memory[1:])


class ToyEventUpdate:
    def update(self, memory, feature, context, robot, timing):
        return (memory[0] + feature[0], *memory[1:])


class ToyPropagation:
    def propagate(self, memory, robot, executed_action, timing):
        return memory


class ToyActionHead:
    def predict(self, state):
        return (state.memory[0] * 0.0001, 0., 0., 0., 0.)


def run_demo(config: Config, include_events=True):
    trace = []
    sink = ShadowSink()
    engine = Engine(config, ToyFrameCorrection(), ToyEventUpdate(), ToyPropagation(),
                    ToyActionHead(), sink, trace.append)

    def feature(kind, time_ms, first, size):
        engine.submit(Message(kind, Stamp(time_ms * 1_000_000), (first,) + (0.,) * (size - 1)))

    feature(Kind.CONTEXT, 0, 1., config.context_dim)
    feature(Kind.ROBOT_STATE, 0, 0., config.robot_dim)
    feature(Kind.FRAME, 0, 1., config.frame_feature_dim)
    feature(Kind.FRAME, 40, 2., config.frame_feature_dim)
    if include_events:
        for ms, value in [(5, .25), (15, -.10), (25, .20), (35, -.05)]:
            feature(Kind.EVENT, ms, value, config.event_feature_dim)
    for tick in control_ticks(0, 40_000_000, config.control_hz):
        engine.submit(tick)
    engine.run_until()
    return {
        'synthetic': True,
        'scope': 'Logical single-writer wiring test; no model learning, real hardware, wall-clock rate or latency claim',
        'config': asdict(config),
        'action_names_provisional': list(ACTION_NAMES),
        'events_enabled': include_events,
        'trace': trace,
        'action_proposals': [asdict(a) for a in sink.proposals],
        'robot_commands_sent': 0,
    }
