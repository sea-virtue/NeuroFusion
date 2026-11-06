import json
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from neurofusion.capture.reader import inspect_recording, iter_event_batches, iter_frames
from neurofusion.capture.schema import CaptureConfig, NativeFrame, TrialLabels
from neurofusion.capture.storage import AsyncWriter, EventStore, FrameStore, write_json


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, frames=False, status="complete"):
        write_json(self.root / "manifest.json", {"format_version": 1, "status": status,
                   "synthetic": True, "config": CaptureConfig(gelsight=frames).to_dict(), "sensors": {}})

    def events(self):
        values = np.array([[1, 2, 100, 1], [3, 4, 100, 0], [5, 6, 105, 1]], dtype=np.int64)
        store = EventStore(self.root)
        store.write((0, 1_000, 2_000, values))
        store.write((1, 3_000, 4_000, np.empty((0, 4), dtype=np.int64)))
        store.close()
        return values

    def test_event_roundtrip_preserves_device_and_host_times_and_empty_polls(self):
        self.manifest()
        original = self.events()
        batches = list(iter_event_batches(self.root))
        np.testing.assert_array_equal(original, batches[0][1])
        self.assertEqual(batches[0][0]["host_received_ns"], 2000)
        self.assertEqual(batches[1][1].shape, (0, 4))
        result = inspect_recording(self.root)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["events"], 3)
        self.assertEqual(result["event_poll_timing"]["frames_or_batches"], 2)

    def test_frame_roundtrip_is_pixel_exact(self):
        self.manifest(frames=True)
        self.events()
        original = np.random.default_rng(3).integers(0, 256, (37, 49, 3), dtype=np.uint8)
        frames = FrameStore(self.root)
        frames.write((0, 10, 20, original))
        frames.close()
        loaded = list(iter_frames(self.root))
        np.testing.assert_array_equal(original, loaded[0][1])
        self.assertTrue(inspect_recording(self.root)["passed"])

    def test_truncated_payload_is_not_accepted(self):
        self.manifest()
        self.events()
        with (self.root / "events/events.bin").open("ab") as stream:
            stream.write(b"x")
        self.assertFalse(inspect_recording(self.root)["passed"])

    def test_native_camera_jpeg_is_saved_byte_for_byte(self):
        from PIL import Image
        self.manifest(frames=True)
        self.events()
        buffer = BytesIO()
        Image.fromarray(np.zeros((37, 49, 3), dtype=np.uint8)).save(buffer, format="JPEG")
        payload = buffer.getvalue()
        frames = FrameStore(self.root)
        frames.write((0, 10, 20, NativeFrame(payload, 49, 37)))
        frames.close()
        self.assertEqual((self.root / "gelsight/frames/00000000.jpg").read_bytes(), payload)
        self.assertTrue(inspect_recording(self.root)["passed"])

    def test_index_cannot_hide_event_payload(self):
        self.manifest()
        self.events()
        (self.root / "events/batches.jsonl").write_text("", encoding="utf-8")
        self.assertFalse(inspect_recording(self.root)["passed"])

    def test_reversed_device_time_rejected_without_silent_sorting(self):
        store = EventStore(self.root)
        try:
            store.write((0, 1, 2, np.array([[1, 2, 100, 1]], dtype=np.int64)))
            with self.assertRaisesRegex(ValueError, "reversed/reset"):
                store.write((1, 3, 4, np.array([[1, 2, 99, 1]], dtype=np.int64)))
        finally:
            store.close()

    def test_static_background_zero_events_is_valid_but_warned(self):
        self.manifest()
        store = EventStore(self.root)
        store.write((0, 1, 2, np.empty((0, 4), dtype=np.int64)))
        store.close()
        result = inspect_recording(self.root)
        self.assertTrue(result["passed"], result)
        self.assertTrue(any("No events" in warning for warning in result["warnings"]))

    def test_failed_or_interrupted_session_is_not_a_complete_training_trial(self):
        self.events()
        for status in ("failed", "interrupted", "recording"):
            self.manifest(status=status)
            self.assertFalse(inspect_recording(self.root)["passed"])

    def test_queue_overflow_is_explicit_and_accepted_items_are_drained(self):
        entered, release = threading.Event(), threading.Event()

        class SlowStore:
            count = 0
            def write(self, item):
                entered.set()
                release.wait(2)
                self.count += 1
            def close(self):
                pass
            def stats(self):
                return {"count": self.count}

        writer = AsyncWriter(SlowStore(), 1)
        try:
            writer.submit("first")
            self.assertTrue(entered.wait(2))
            writer.submit("second")
            with self.assertRaisesRegex(RuntimeError, "queue full"):
                writer.submit("third")
        finally:
            release.set()
            writer.close()
        self.assertEqual(writer.stats()["count"], 2)

    def test_disk_error_propagates_out_of_background_writer(self):
        class BrokenStore:
            def write(self, item):
                raise OSError("disk full")
            def close(self):
                pass
        writer = AsyncWriter(BrokenStore(), 1)
        writer.submit("x")
        writer.thread.join(2)
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            writer.close()

    def test_slide_requires_direction_and_duration_leaves_motion_interval(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            CaptureConfig(labels=TrialLabels(task="slide")).validate()
        with self.assertRaisesRegex(ValueError, "shorter"):
            CaptureConfig(seconds=3).validate()


if __name__ == "__main__":
    unittest.main()
