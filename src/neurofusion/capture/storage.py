"""Lossless stores and bounded asynchronous writers, independent of sensor SDKs."""
import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from .schema import EVENT_COLUMNS, EVENT_DTYPE, NativeFrame


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def event_array(values, previous_timestamp=None):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 4 or values.dtype.kind not in "iu":
        raise ValueError("Events must be an integer N x 4 array [x,y,timestamp_us,polarity]")
    if values.size:
        if (values[:, :2] < 0).any() or (values[:, :2] >= 128).any():
            raise ValueError("Event coordinate outside 128x128")
        if (values[:, 2] < 0).any() or not np.isin(values[:, 3], [0, 1]).all():
            raise ValueError("Invalid event timestamp/polarity")
        times = values[:, 2]
        if (times[1:] < times[:-1]).any() or (previous_timestamp is not None and times[0] < previous_timestamp):
            raise ValueError("Event time reversed/reset; raw data will not be silently sorted")
        if (values > np.iinfo(np.int64).max).any():
            raise ValueError("Event integer cannot be represented by the file dtype")
    return np.asarray(values, dtype=EVENT_DTYPE, order="C")


class EventStore:
    def __init__(self, root):
        self.root = Path(root) / "events"
        self.root.mkdir()
        self.binary = (self.root / "events.bin").open("xb")
        self.index = (self.root / "batches.jsonl").open("x", encoding="utf-8")
        write_json(self.root / "format.json", {"dtype": EVENT_DTYPE, "columns": EVENT_COLUMNS,
                   "shape": ["N", 4], "bytes_per_event": 32, "sensor_size": [128, 128],
                   "time_unit": "us", "device_clock": "Speck stopwatch; not mapped to camera/host"})
        self.count = self.batches = 0
        self.last_timestamp = None

    def write(self, item):
        sequence, started, received, values = item
        values = event_array(values, self.last_timestamp)
        count = len(values)
        # Payload first, then index: an interrupted write is detected by inspect.
        self.binary.write(values.tobytes(order="C"))
        self.binary.flush()
        line(self.index, {"batch_id": sequence, "offset_events": self.count, "count": count,
             "host_read_started_ns": started, "host_received_ns": received,
             "device_first_us": int(values[0, 2]) if count else None,
             "device_last_us": int(values[-1, 2]) if count else None})
        if count:
            self.last_timestamp = int(values[-1, 2])
        self.count += count
        self.batches += 1

    def close(self):
        for stream in (self.binary, self.index):
            try:
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                stream.close()

    def stats(self):
        return {"events_written": self.count, "batches_written": self.batches,
                "event_bytes": self.count * 32}


class FrameStore:
    def __init__(self, root):
        self.root = Path(root) / "gelsight"
        (self.root / "frames").mkdir(parents=True)
        self.index = (self.root / "frames.jsonl").open("x", encoding="utf-8")
        self.count = self.bytes = 0

    def write(self, item):
        from PIL import Image
        sequence, started, received, rgb = item
        native = isinstance(rgb, NativeFrame)
        if not native and (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3):
            raise ValueError("GelSight requires a native JPEG or decoded uint8 RGB")
        suffix = "jpg" if native else "png"
        relative = f"gelsight/frames/{sequence:08d}.{suffix}"
        destination = self.root.parent / relative
        temporary = destination.with_suffix(f".{suffix}.part")
        # Native path writes camera-produced JPEG bytes exactly, with no decoding.
        if native:
            temporary.write_bytes(rgb.payload)
            shape = [rgb.height, rgb.width, 3]
        else:
            Image.fromarray(rgb).save(temporary, format="PNG", compress_level=1)
            shape = list(rgb.shape)
        temporary.replace(destination)
        size = destination.stat().st_size
        line(self.index, {"frame_id": sequence, "path": relative,
             "host_read_started_ns": started, "host_received_ns": received,
             "host_saved_ns": time.monotonic_ns(), "shape": shape,
             "bytes": size, "color": "RGB", "device_timestamp": None,
             "encoding": "camera_native_jpeg" if native else "lossless_png_from_decoded_rgb"})
        self.count += 1
        self.bytes += size

    def close(self):
        try:
            self.index.flush()
            os.fsync(self.index.fileno())
        finally:
            self.index.close()

    def stats(self):
        return {"frames_written": self.count, "frame_bytes": self.bytes}


class AsyncWriter:
    """A full queue fails the trial instead of silently discarding raw observations."""
    def __init__(self, store, capacity):
        self.store = store
        self.queue = queue.Queue(capacity)
        self.error = None
        self.high_water = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                self.store.write(item)
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.store.close()
            except BaseException as error:
                self.error = self.error or error

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Raw writer failed: {self.error}") from self.error

    def submit(self, item):
        self.check()
        try:
            self.queue.put_nowait(item)
            self.high_water = max(self.high_water, self.queue.qsize())
        except queue.Full:
            raise RuntimeError("Raw writer queue full: this trial is FAILED (not silently downsampled). "
                               "Try events-only or a camera-supported lower resolution on a later trial.") from None

    def close(self):
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                self.check()
        self.thread.join()
        self.check()

    def stats(self):
        return {**self.store.stats(), "queue_high_water": self.high_water,
                "queue_capacity": self.queue.maxsize}
