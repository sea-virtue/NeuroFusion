"""Separate sensor processes; each has a bounded disk-writing thread.

Only preview copies may be dropped. Raw queue overflow marks the recording failed.
"""
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import queue
import signal
import shutil
import time
import traceback
import uuid

import numpy as np

from .schema import FORMAT_VERSION, CaptureConfig, instruction_schedule
from .storage import AsyncWriter, EventStore, FrameStore, line, write_json


def _preview(channel, value):
    # Never slow raw acquisition for a GUI; dropping a preview is intentional.
    try:
        channel.put_nowait(value)
    except queue.Full:
        pass


def sensor_worker(sensor, config, root, stop, start, deadline, messages, previews):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    writer = None
    reads = observations = rejected = 0
    error = None
    last_status = last_preview = 0
    event_image = np.zeros((128, 128), dtype=np.int8)
    try:
        from neurofusion.adapters.live_devices import SpeckSource
        from .sources import NativeGelSightSource
        source = SpeckSource() if sensor == "events" else NativeGelSightSource(config.camera)
        with source as device:
            store = EventStore(root) if sensor == "events" else FrameStore(root)
            writer = AsyncWriter(store, config.event_queue if sensor == "events" else config.frame_queue)
            info = {"host_ready_ns": time.monotonic_ns()}
            if sensor == "gelsight":
                info.update(device.info)
            else:
                info.update({"sensor_size": [128, 128], "device": "Speck2fModuleDevKit"})
            messages.put({"sensor": sensor, "kind": "ready", "info": info})
            while not start.wait(0.05):
                if stop.is_set():
                    return
            # Speck starts its clock on opening. Drain startup-only buffered events
            # at the common host gate, then preserve every subsequent SDK batch.
            if sensor == "events":
                _, startup = device.read()
                info["startup_events_excluded"] = len(startup)
                messages.put({"sensor": sensor, "kind": "info", "info": info})
            while not stop.is_set() and time.monotonic_ns() < deadline.value:
                started = time.monotonic_ns()
                received, values = device.read()
                reads += 1
                observations += len(values) if sensor == "events" else 1
                try:
                    writer.submit((reads - 1, started, received, values))
                except Exception:
                    rejected += len(values) if sensor == "events" else 1
                    raise
                if sensor == "events":
                    if len(values):
                        event_image[values[:, 1], values[:, 0]] = np.where(values[:, 3], 1, -1)
                    if received - last_preview >= 50_000_000:
                        _preview(previews, (sensor, event_image.copy()))
                        event_image.fill(0)
                        last_preview = received
                    stop.wait(config.poll_ms / 1000)
                elif received - last_preview >= 150_000_000:
                    _preview(previews, (sensor, values))
                    last_preview = received
                if received - last_status >= 1_000_000_000:
                    writer.check()
                    messages.put({"sensor": sensor, "kind": "progress", "reads": reads,
                                  "observations_received": observations, "stats": writer.stats()})
                    last_status = received
            writer.close()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        messages.put({"sensor": sensor, "kind": "error", "error": error, "traceback": traceback.format_exc()})
        stop.set()
    finally:
        if writer is not None and writer.thread.is_alive():
            try:
                writer.close()
            except Exception as exc:
                error = error or str(exc)
                messages.put({"sensor": sensor, "kind": "error", "error": str(exc)})
                stop.set()
        messages.put({"sensor": sensor, "kind": "done", "error": error,
                      "reads": reads, "observations_received": observations,
                      "rejected_observations": rejected,
                      "stats": writer.stats() if writer else {}})


class CaptureSession:
    """Poll from a CLI or GUI. This controller never reads sensor SDKs itself."""
    def __init__(self, config: CaptureConfig):
        config.validate()
        self.config = config
        self.instructions = instruction_schedule(config)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}_{config.labels.task}_{uuid.uuid4().hex[:8]}"
        self.root = Path(config.output or Path(config.root) / name).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = {"format_version": FORMAT_VERSION, "status": "starting",
                         "created_utc": datetime.now(timezone.utc).isoformat(),
                         "host_monotonic_anchor_ns": time.monotonic_ns(), "host_wall_anchor_ns": time.time_ns(),
                         "config": config.to_dict(), "synthetic": False,
                         "disk_free_bytes_at_start": shutil.disk_usage(self.root).free,
                         "time_semantics": "host_received_ns is read completion on one host monotonic clock, NOT sensor exposure; event device us is separate/unmapped",
                         "image_semantics": "original camera-produced MJPEG frame bytes in individual JPEG files, full resolution; no host decode/recompression on acquisition path; not Bayer RAW",
                         "label_semantics": "operator trial conditions and timed prompts; no verified per-window motion/force ground truth",
                         "instruction_schedule_seconds": self.instructions,
                         "external_rgb": False, "robot_actions": False, "robot_commands_sent": 0,
                         "sensors": {}, "errors": []}
        write_json(self.root / "manifest.json", self.manifest)
        self.marks = (self.root / "annotations.jsonl").open("x", encoding="utf-8")
        self.context = mp.get_context("spawn")
        self.stop_event, self.start_event = self.context.Event(), self.context.Event()
        self.deadline = self.context.Value("q", 0)
        self.messages = self.context.Queue()
        self.previews = self.context.Queue(maxsize=2)
        self.processes = {}
        self.ready = set()
        self.done = set()
        self.started_ns = None
        self.stopping_at = None
        self.interrupted = False
        self.finished = False
        self.phase = "opening sensors"
        self.launched_at = time.monotonic()

    def start(self):
        try:
            for sensor in (["events", "gelsight"] if self.config.gelsight else ["events"]):
                process = self.context.Process(target=sensor_worker,
                    args=(sensor, self.config, str(self.root), self.stop_event, self.start_event,
                          self.deadline, self.messages, self.previews), name=f"capture-{sensor}")
                process.start()
                self.processes[sensor] = process
        except BaseException as error:
            self.manifest["errors"].append(f"Process launch failed: {error}")
            self.stop_event.set()
            self.stopping_at = time.monotonic()
        return self

    def mark(self, name, source="operator_button"):
        if self.finished:
            return
        line(self.marks, {"name": name, "host_marked_ns": time.monotonic_ns(),
                         "source": source, "verified_motion_ground_truth": False})

    def stop(self):
        if not self.finished:
            self.interrupted = True
            self.stop_event.set()
            self.stopping_at = self.stopping_at or time.monotonic()

    def _drain(self):
        updates = []
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            updates.append(message)
            sensor = message["sensor"]
            entry = self.manifest["sensors"].setdefault(sensor, {})
            kind = message["kind"]
            if kind in ("ready", "info"):
                entry.update(message["info"])
                self.ready.add(sensor)
            elif kind == "error":
                self.manifest["errors"].append(f"{sensor}: {message['error']}")
                entry["traceback"] = message.get("traceback")
                self.stop_event.set()
            else:
                entry.update({k: v for k, v in message.items() if k not in ("kind", "sensor")})
                if kind == "done":
                    self.done.add(sensor)
        return updates

    def poll(self):
        if self.finished:
            return []
        updates = self._drain()
        now = time.monotonic_ns()
        if len(self.ready) == len(self.processes) and self.processes and self.started_ns is None and not self.stop_event.is_set():
            self.started_ns = now
            self.deadline.value = now + round(self.config.seconds * 1e9)
            self.manifest.update(status="recording", host_recording_start_ns=now,
                                 host_planned_end_ns=self.deadline.value)
            write_json(self.root / "manifest.json", self.manifest)
            self.mark("recording_start", "host_controller")
            self.start_event.set()
        if self.started_ns is not None and not self.stop_event.is_set():
            elapsed = (now - self.started_ns) / 1e9
            phase = next(text for offset, text in reversed(self.instructions) if offset <= elapsed)
            if phase != self.phase:
                self.phase = phase
                self.mark(phase, "timed_instruction_not_measured_motion")
                updates.append({"kind": "phase", "text": phase})
            if now >= self.deadline.value:
                self.stop_event.set()
                self.mark("planned_duration_reached", "host_controller")
        elif self.started_ns is None and time.monotonic() - self.launched_at > self.config.startup_timeout:
            self.manifest["errors"].append("Sensor startup timeout")
            self.stop_event.set()
        for sensor, process in self.processes.items():
            if process.exitcode is not None and sensor not in self.done:
                # A process's queue feeder flushes before its normal exit.
                updates += self._drain()
                if sensor not in self.done:
                    self.manifest["errors"].append(f"{sensor}: process exited without final summary ({process.exitcode})")
                    self.done.add(sensor)
                    self.stop_event.set()
            if process.exitcode not in (None, 0) and not any(f"{sensor}: process exit" in e for e in self.manifest["errors"]):
                self.manifest["errors"].append(f"{sensor}: process exit {process.exitcode}")
                self.stop_event.set()
        if self.stop_event.is_set():
            self.stopping_at = self.stopping_at or time.monotonic()
            self.phase = "saving / releasing sensors"
            if time.monotonic() - self.stopping_at > 20:
                for sensor, process in self.processes.items():
                    if process.is_alive():
                        process.terminate()
                        self.manifest["errors"].append(f"{sensor}: forced termination after shutdown timeout; inspect partial data")
            if all(not p.is_alive() for p in self.processes.values()):
                self._finish()
        return updates

    def _finish(self):
        for process in self.processes.values():
            process.join(timeout=0)
        self._drain()
        self.mark("recording_closed", "host_controller")
        self.marks.close()
        self.manifest["host_closed_ns"] = time.monotonic_ns()
        self.manifest["status"] = "failed" if self.manifest["errors"] else "interrupted" if self.interrupted else "complete"
        self.manifest["duration_measured_s"] = ((self.manifest["host_closed_ns"] - self.started_ns) / 1e9 if self.started_ns else 0)
        self.manifest["duration_note"] = "includes final capture completion and writer drain; use per-observation host times for actual coverage"
        write_json(self.root / "manifest.json", self.manifest)
        self.messages.close()
        self.previews.close()
        self.finished = True
        self.phase = self.manifest["status"]

    def latest_previews(self):
        latest = {}
        if self.finished:
            return latest
        while True:
            try:
                sensor, value = self.previews.get_nowait()
                latest[sensor] = value
            except queue.Empty:
                return latest
