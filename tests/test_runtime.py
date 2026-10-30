import threading
import unittest

from neurofusion.adapters.interfaces import ShadowSink
from neurofusion.config import Config
from neurofusion.contracts import Kind, Message, Stamp
from neurofusion.demo import run_demo, ToyActionHead, ToyEventUpdate, ToyFrameCorrection, ToyPropagation
from neurofusion.models.pending import RegressionActionHead
from neurofusion.runtime.engine import Engine


def fixture(config=None):
    config = config or Config()
    sink = ShadowSink()
    return Engine(config, ToyFrameCorrection(), ToyEventUpdate(), ToyPropagation(), ToyActionHead(), sink), sink


class RuntimeTests(unittest.TestCase):
    def test_event_changes_action_between_slow_frames(self):
        with_events = run_demo(Config())
        without = run_demo(Config(), include_events=False)
        # The next frame is at 40 ms; 20 ms action must already reflect events.
        a = next(a for a in with_events['action_proposals'] if a['time_ns'] == 20_000_000)
        b = next(a for a in without['action_proposals'] if a['time_ns'] == 20_000_000)
        self.assertNotEqual(a['values'], b['values'])
        self.assertEqual(with_events['robot_commands_sent'], 0)

    def test_same_time_queued_observation_precedes_control(self):
        config = Config()
        engine, sink = fixture(config)
        engine.submit(Message(Kind.CONTROL, Stamp(10)))
        engine.submit(Message(Kind.EVENT, Stamp(10), (1.,) * config.event_feature_dim))
        engine.run_until()
        self.assertGreater(sink.proposals[0].values[0], 0)

    def test_control_ticks_do_not_wait_for_frames(self):
        config = Config(control_hz=100)
        report = run_demo(config, include_events=False)
        self.assertEqual([a['time_ns'] for a in report['action_proposals']],
                         [0, 10_000_000, 20_000_000, 30_000_000, 40_000_000])

    def test_rejects_backdated_available_time(self):
        engine, _ = fixture()
        engine.submit(Message(Kind.CONTROL, Stamp(10)))
        engine.run_until()
        with self.assertRaises(ValueError):
            engine.submit(Message(Kind.CONTROL, Stamp(9)))

    def test_device_timestamp_does_not_reorder_host_availability(self):
        engine, _ = fixture()
        engine.submit(Message(Kind.CONTROL, Stamp(20, device_us=1)))
        engine.submit(Message(Kind.CONTROL, Stamp(10, device_us=999)))
        engine.run_until()
        self.assertEqual(engine.state.time_ns, 20)

    def test_rejects_invalid_shape_nan_and_queue_overflow(self):
        engine, _ = fixture(Config(max_pending_messages=1))
        with self.assertRaises(ValueError):
            engine.submit(Message(Kind.EVENT, Stamp(0), (1.,)))
        with self.assertRaises(ValueError):
            engine.submit(Message(Kind.EVENT, Stamp(0), (float('nan'),) * engine.config.event_feature_dim))
        engine.submit(Message(Kind.CONTROL, Stamp(0)))
        with self.assertRaises(OverflowError):
            engine.submit(Message(Kind.CONTROL, Stamp(1)))

    def test_cross_thread_writes_rejected(self):
        engine, _ = fixture()
        errors = []
        def other_thread():
            try:
                engine.submit(Message(Kind.CONTROL, Stamp(0)))
            except RuntimeError as error:
                errors.append(error)
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
        self.assertEqual(len(errors), 1)

    def test_learned_head_does_not_silently_run_as_fake_model(self):
        engine, _ = fixture()
        with self.assertRaises(NotImplementedError):
            RegressionActionHead().predict(engine.state)

    def test_clock_domains_and_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            Stamp(0, clock_id='device')
        with self.assertRaises(ValueError):
            Config(memory_dim=0)
        with self.assertRaises(ValueError):
            Config(control_hz=100.0)

    def test_repeatable_synthetic_trace(self):
        self.assertEqual(run_demo(Config()), run_demo(Config()))


if __name__ == '__main__':
    unittest.main()
