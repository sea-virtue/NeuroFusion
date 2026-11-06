"""Weakly labelled event windows grouped by original capture."""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from torch.utils.data import Dataset

from ..capture.reader import iter_event_batches
from ..models.events import voxelize
from ..models.gelneuro_snn import GelNeuroSNN


DIRECTIONS = ("up", "down", "left", "right")
WINDOW_US = 32_000
GUARD_NS = 550_000_000


@dataclass
class Trial:
    name: str
    labels: dict
    events: np.ndarray
    start_us: int
    end_us: int

    @property
    def direction(self):
        return DIRECTIONS.index(self.labels["direction"]) if self.labels["task"] == "slide" else -1


def read_trials(root, reviewed=None):
    """Read audited captures. Optional reviewed bounds use original device microseconds."""
    reviews = json.loads(Path(reviewed).read_text()) if reviewed else {}
    trials = []
    for folder in sorted(Path(root).iterdir()):
        if not (folder / "manifest.json").is_file():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        quality = json.loads((folder / "quality.json").read_text())
        if manifest["status"] != "complete" or not quality["passed"] or manifest.get("synthetic"):
            raise ValueError(f"Recording has not passed the real-data audit: {folder.name}")
        labels = manifest["config"]["labels"]
        task = labels["task"]
        if task == "slide" and labels["direction"] not in DIRECTIONS:
            raise ValueError(f"Unknown slide direction: {folder.name}")
        batches = [(meta, items) for meta, items in iter_event_batches(folder) if len(items)]
        if not batches:
            raise ValueError(f"Recording contains no device timestamps: {folder.name}")
        events = np.concatenate([items for _, items in batches])
        start_us, end_us = int(events[0, 2]), int(events[-1, 2]) + 1
        if folder.name in reviews:
            start_us, end_us = reviews[folder.name]
            if not int(events[0, 2]) <= start_us < end_us <= int(events[-1, 2]) + 1:
                raise ValueError(f"Reviewed bounds outside recording: {folder.name}")
        elif task == "slide":
            marks = [json.loads(line) for line in (folder / "annotations.jsonl").read_text().splitlines()]
            begin = next(mark["host_marked_ns"] for mark in marks if mark["name"].startswith("DO SLIDE"))
            stop = next(mark["host_marked_ns"] for mark in marks if mark["name"].startswith("STOP MOTION"))
            interior = [items for meta, items in batches
                        if begin + GUARD_NS <= meta["host_received_ns"] < stop - GUARD_NS]
            if not interior:
                raise ValueError(f"No events inside guarded prompt interval: {folder.name}")
            start_us, end_us = int(interior[0][0, 2]), int(interior[-1][-1, 2]) + 1
        if end_us - start_us < WINDOW_US:
            raise ValueError(f"Selected interval shorter than {WINDOW_US} us: {folder.name}")
        trials.append(Trial(folder.name, labels, events, start_us, end_us))
    if not trials:
        raise ValueError(f"No audited tactile recordings in {root}")
    unknown = set(reviews) - {trial.name for trial in trials}
    if unknown:
        raise ValueError(f"Reviewed recordings absent: {sorted(unknown)}")
    return trials


def split_trials(trials, fold):
    if fold not in range(3):
        raise ValueError("fold must be 0, 1 or 2")
    groups = defaultdict(list)
    replay = []
    for trial in trials:
        task = trial.labels["task"]
        if task == "slide":
            groups[(task, trial.labels["direction"], trial.labels["speed_level"])].append(trial)
        elif task in ("background", "static_contact"):
            groups[(task,)].append(trial)
        else:
            replay.append(trial)
    train, heldout = [], []
    for group in groups.values():
        group.sort(key=lambda trial: (trial.labels["repetition"], trial.name))
        for index, trial in enumerate(group):
            (heldout if index % 3 == fold else train).append(trial)
    if any({trial.direction for trial in subset} != {-1, 0, 1, 2, 3} for subset in (train, heldout)):
        raise ValueError("Both folds need quiet data and all four slide directions")
    return train, heldout, replay


def pack_window(events, start_us, end_us, bin_us):
    x0, y0, x1, y1 = GelNeuroSNN.roi
    x, y, time, polarity = events.T
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    index = (((time[keep] - start_us) // bin_us * 2 + polarity[keep]) * 120
             + y[keep] - y0) * 120 + x[keep] - x0
    steps = (end_us - start_us) // bin_us
    return np.bincount(index, minlength=steps * 2 * 120 * 120).astype(np.float32).reshape(steps, 2, 120, 120)


class Windows(Dataset):
    def __init__(self, trials, kind, bin_us, per_trial, seed=0, jitter=False, shuffle_time=False):
        self.trials, self.kind, self.bin_us = trials, kind, bin_us
        self.seed, self.jitter, self.shuffle_time = seed, jitter, shuffle_time
        self.per_trial = per_trial
        self.epoch = 0

    def __len__(self):
        return len(self.trials) * self.per_trial

    def __getitem__(self, index):
        trial_id, window_id = divmod(index, self.per_trial)
        trial = self.trials[trial_id]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        fraction = (window_id + (rng.random() if self.jitter else 0.5)) / self.per_trial
        start_us = trial.start_us + int(fraction * (trial.end_us - trial.start_us - WINDOW_US))
        end_us = start_us + WINDOW_US
        lo, hi = np.searchsorted(trial.events[:, 2], [start_us, end_us])
        events = trial.events[lo:hi]
        if self.shuffle_time:
            events = events.copy()
            events[:, 2] = rng.permutation(events[:, 2])
            events = events[np.argsort(events[:, 2], kind="stable")]
        if self.kind == "snn":
            value = pack_window(events, start_us, end_us, self.bin_us)
        else:
            value = voxelize([events], start_us, end_us)[0]
        return value, trial.direction, trial_id, end_us
