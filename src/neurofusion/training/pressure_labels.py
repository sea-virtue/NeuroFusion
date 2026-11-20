"""Pressure-phase labels for the shared GelNeuro binning and SNN trainer."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .gelneuro_data import read_trials
from .train_gelneuro_temporal import IGNORE, recording_bins


PRESSURE_CLASSES = ("quiet", "press", "release")
PRESSURE_INTERVALS = "pressure_intervals_us"


def pressure_trials(root, labels_path):
    review = json.loads(Path(labels_path).read_text())
    intervals_by_trial = review["trials"]
    trials = []
    for trial in read_trials(root):
        if trial.name not in intervals_by_trial:
            continue
        if trial.labels["task"] != "press_release":
            raise ValueError(f"Pressure intervals require a press_release trial: {trial.name}")
        intervals = intervals_by_trial[trial.name]
        if not intervals:
            raise ValueError(f"Empty pressure labels: {trial.name}")
        last = trial.start_us
        classes = set()
        for name, start, end in intervals:
            if name not in PRESSURE_CLASSES or type(start) is not int or type(end) is not int:
                raise ValueError(f"Invalid device-time pressure label: {trial.name}")
            if not last <= start < end <= trial.end_us:
                raise ValueError(f"Overlapping or out-of-range pressure intervals: {trial.name}")
            classes.add(name)
            last = end
        if classes != set(PRESSURE_CLASSES):
            raise ValueError(f"Each pressure trial needs quiet, press and release intervals: {trial.name}")
        trials.append(replace(trial, labels={**trial.labels, PRESSURE_INTERVALS: intervals}))
    if {t.name for t in trials} != set(intervals_by_trial) or len(trials) < 3:
        raise ValueError("Pressure review must refer to at least three audited original trials")
    return trials


def pressure_targets(trial, ends, bin_us):
    target = np.full(len(ends), IGNORE, dtype=np.int64)
    for name, start, end in trial.labels[PRESSURE_INTERVALS]:
        target[(ends - bin_us >= start) & (ends <= end)] = PRESSURE_CLASSES.index(name)
    if set(target[target >= 0]) != set(range(len(PRESSURE_CLASSES))):
        raise ValueError(f"No complete supervised bins: {trial.name} needs all three pressure classes")
    return torch.from_numpy(target)


def prepare_pressure(root, labels_path, bin_us):
    trials = pressure_trials(root, labels_path)
    sequences = {}
    for trial in trials:
        bins, _, ends = recording_bins(trial, bin_us=bin_us)
        target = pressure_targets(trial, ends, bin_us)
        sequences[trial.name] = bins, target, torch.full_like(target, IGNORE), ends
    return trials, sequences


def split_pressure(trials, fold):
    ordered = sorted(trials, key=lambda trial: (trial.labels["repetition"], trial.name))
    train, heldout = [], []
    for i, trial in enumerate(ordered):
        (heldout if i % 3 == fold else train).append(trial)
    return train, heldout
