"""Streaming replay and integrity checks. These do not train a model."""
import json
from pathlib import Path

import numpy as np

from .schema import EVENT_DTYPE, FORMAT_VERSION, safe_path
from .storage import event_array, write_json


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, text in enumerate(stream, 1):
            try:
                yield json.loads(text)
            except Exception as error:
                raise ValueError(f"{path}:{number}: broken JSON index") from error


def iter_event_batches(root):
    """Yield (batch metadata, N x 4 int64 view); includes empty SDK polls."""
    root = Path(root)
    path = root / "events/events.bin"
    size = path.stat().st_size
    if size % 32:
        raise ValueError("Event payload ends in a partial event")
    data = np.memmap(path, dtype=EVENT_DTYPE, mode="r", shape=(size // 32, 4)) if size else np.empty((0, 4), dtype=EVENT_DTYPE)
    offset = 0
    for batch in rows(root / "events/batches.jsonl"):
        count = batch["count"]
        if not isinstance(count, int) or count < 0 or batch["offset_events"] != offset or offset + count > len(data):
            raise ValueError("Event batch index has a gap, overlap or truncated payload")
        yield batch, data[offset:offset + count]
        offset += count
    if offset != len(data):
        raise ValueError("Event payload has unindexed events (interrupted write)")


def iter_frames(root):
    """Yield (frame metadata, decoded RGB uint8), one full frame at a time."""
    from PIL import Image
    root = Path(root)
    for frame in rows(root / "gelsight/frames.jsonl"):
        with Image.open(safe_path(root, frame["path"])) as image:
            yield frame, np.array(image.convert("RGB"))


def _timing(times):
    if len(times) < 2:
        return {"frames_or_batches": len(times), "observed_hz": None, "max_gap_ms": None}
    gaps = np.diff(np.asarray(times, dtype=np.int64))
    if (gaps < 0).any():
        raise ValueError("Host receipt times reversed")
    span = (times[-1] - times[0]) / 1e9
    return {"frames_or_batches": len(times), "observed_hz": (len(times) - 1) / span if span else None,
            "median_gap_ms": float(np.median(gaps) / 1e6),
            "p95_gap_ms": float(np.percentile(gaps, 95) / 1e6),
            "max_gap_ms": float(gaps.max() / 1e6)}


def inspect_recording(root, save=False):
    root = Path(root)
    report = {"path": str(root.resolve()), "format_version": FORMAT_VERSION,
              "issues": [], "warnings": [], "events": 0, "frames": 0}
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported recording format")
        report["recording_status"] = manifest["status"]
        if manifest["status"] != "complete":
            report["issues"].append(f"Recording was {manifest['status']}; not a complete trial")
        if manifest.get("errors"):
            report["issues"].extend(manifest["errors"])
        report["labels"] = manifest["config"]["labels"]
        report["synthetic"] = manifest.get("synthetic", False)
        times = []
        previous_device = None
        last_received = None
        for sequence, (batch, values) in enumerate(iter_event_batches(root)):
            if batch["batch_id"] != sequence or batch["host_read_started_ns"] > batch["host_received_ns"]:
                raise ValueError("Bad event sequence/read timing")
            if last_received is not None and batch["host_read_started_ns"] < last_received:
                raise ValueError("Event read intervals overlap or reverse")
            last_received = batch["host_received_ns"]
            event_array(values, previous_device)
            if len(values):
                if batch["device_first_us"] != int(values[0, 2]) or batch["device_last_us"] != int(values[-1, 2]):
                    raise ValueError("Device timestamps disagree with index")
                report.setdefault("first_device_us", int(values[0, 2]))
                previous_device = int(values[-1, 2])
            elif batch["device_first_us"] is not None or batch["device_last_us"] is not None:
                raise ValueError("Empty event batch has device timestamps")
            report["events"] += len(values)
            times.append(batch["host_received_ns"])
        report["event_poll_timing"] = _timing(times)
        if not times:
            report["issues"].append("No SDK polls saved")
        report["last_device_us"] = previous_device
        if not report["events"]:
            report["warnings"].append("No events. This may be valid background; use a pressed/moving trial to verify activity.")
        elif report.get("first_device_us") == report["last_device_us"]:
            report["issues"].append("Device clock did not advance across events")
        if manifest["config"]["gelsight"]:
            times = []
            from PIL import Image
            indexed = set()
            for sequence, frame in enumerate(rows(root / "gelsight/frames.jsonl")):
                path = safe_path(root, frame["path"])
                if frame["frame_id"] != sequence or frame["host_read_started_ns"] > frame["host_received_ns"]:
                    raise ValueError("Bad frame sequence/read timing")
                if times and frame["host_read_started_ns"] < times[-1]:
                    raise ValueError("Frame read intervals overlap or reverse")
                with Image.open(path) as image:
                    if list((image.height, image.width, len(image.getbands()))) != frame["shape"] or image.mode != "RGB":
                        raise ValueError(f"Frame shape/color mismatch: {path.name}")
                    image.load()  # Full decode, not just a header check.
                if path.stat().st_size != frame["bytes"]:
                    raise ValueError(f"Frame size mismatch: {path.name}")
                if frame["host_saved_ns"] < frame["host_received_ns"]:
                    raise ValueError("Frame save precedes receipt")
                indexed.add(path.name)
                times.append(frame["host_received_ns"])
                report["frames"] += 1
                report["frame_shape"] = frame["shape"]
            files = {path.name for path in (root / "gelsight/frames").iterdir()}
            if files != indexed:
                raise ValueError("Unindexed, missing or unfinished frame files")
            report["frame_timing"] = _timing(times)
            report["frame_bytes"] = sum(path.stat().st_size for path in (root / "gelsight/frames").iterdir())
            if not times:
                report["issues"].append("GelSight enabled but no frame saved")
            report["warnings"].append("Camera host receipt timing is not exposure timing; hardware frame drops cannot be inferred from it alone.")
        for sensor, key, measured in (("events", "events_written", report["events"]), ("gelsight", "frames_written", report["frames"])):
            entry = manifest.get("sensors", {}).get(sensor, {})
            stats = entry.get("stats", {})
            if key in stats and stats[key] != measured:
                report["issues"].append(f"{sensor}: summary count differs from saved data")
            if entry.get("rejected_observations", 0):
                report["issues"].append(f"{sensor}: raw observations were rejected by the writer")
            if "observations_received" in entry and entry["observations_received"] != measured:
                report["issues"].append(f"{sensor}: received count differs from saved data")
            if sensor == "events" and "batches_written" in stats and stats["batches_written"] != report["event_poll_timing"]["frames_or_batches"]:
                report["issues"].append("Event batch summary differs from saved index")
        report["event_bytes"] = (root / "events/events.bin").stat().st_size
    except Exception as error:
        report["issues"].append(str(error))
    report["warnings"].append("Operator conditions/prompts are weak labels; no measured force, velocity, expert robot actions or precise inter-device synchronization.")
    report["passed"] = not report["issues"]
    if save:
        write_json(root / "quality.json", report)
    return report
