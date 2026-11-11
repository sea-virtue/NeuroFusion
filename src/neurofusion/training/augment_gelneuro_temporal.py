"""Create training-only GelNeuro variants without rewriting captures."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .gelneuro_data import read_trials


TIME_SCALES = (0.90, 0.95, 1.00, 1.00, 1.05, 1.10)


def source_signature(trial):
    sha = hashlib.sha256()
    sha.update(trial.events.tobytes())
    sha.update(json.dumps([trial.start_us, trial.end_us, trial.labels],
                          sort_keys=True).encode())
    return sha.hexdigest()


def generate(trial, variant, rng, method, bin_us=100_000):
    from .train_gelneuro_temporal import recording_bins

    origin = int(trial.events[0, 2])
    last = int(trial.events[-1, 2])
    if method == "phase":
        offset = (variant + 1) * 10_000
        begin = origin + offset
        derived = replace(trial, events=trial.events[trial.events[:, 2] >= begin])
        bins, labels, ends = recording_bins(derived, bounds=(begin, last), bin_us=bin_us)
        meta = dict(phase_offset_us=offset, source_events=len(trial.events),
                    retained_events=len(derived.events), labelled_bins=int(labels.ne(-2).sum()))
        return bins.numpy(), labels.numpy(), ends, meta

    scale = TIME_SCALES[variant]
    dx, dy = rng.integers(-6, 7, size=2).tolist()
    drop = float(rng.uniform(.05, .15))
    events = trial.events.copy()
    events[:, 0] += dx
    events[:, 1] += dy

    def warp(timestamp):
        return origin + np.rint((timestamp - origin) * scale).astype(np.int64)

    events[:, 2] = warp(events[:, 2])
    x, y = events[:, :2].T
    inside = (x >= 0) & (x < 128) & (y >= 0) & (y < 128)
    keep = inside & (rng.random(len(events)) >= drop)
    derived = replace(trial, events=events[keep], start_us=int(warp(trial.start_us)),
                      end_us=int(warp(trial.end_us)))
    bins, labels, ends = recording_bins(derived, bounds=(origin, int(warp(last))), bin_us=bin_us)
    meta = dict(dx=dx, dy=dy, time_scale=scale, event_drop_probability=drop,
                source_events=len(events), retained_events=int(keep.sum()),
                labelled_bins=int(labels.ne(-2).sum()))
    return bins.numpy(), labels.numpy(), ends, meta


class AugmentedRecordings(Dataset):
    """Sample one version of each training recording per epoch."""
    def __init__(self, root, trials, originals, seed, bin_us=None):
        self.originals = originals
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if bin_us is not None and manifest["bin_us"] != bin_us:
            raise ValueError("Augmented bin width differs from model")
        if bool(manifest.get("unified", False)) != (len(originals[0]) == 4):
            raise ValueError("Augmented target layout differs from training task")
        self.variants = []
        self.seed = seed
        self.epoch = 0
        for trial in trials:
            entry = manifest["recordings"][trial.name]
            if entry["source_sha256"] != source_signature(trial):
                raise ValueError(f"Augmented data no longer matches source {trial.name}")
            self.variants.append([self.root / item["file"] for item in entry["variants"]])

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, index):
        choices = self.variants[index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        # Keep original recordings in the mix while preserving recording-level sampling.
        if rng.random() < .25 or not choices:
            return self.originals[index]
        with np.load(choices[int(rng.integers(len(choices)))], allow_pickle=False) as data:
            bins, labels = torch.from_numpy(data["bins"]), torch.from_numpy(data["labels"])
            if "directions" in data:
                return bins, labels, torch.from_numpy(data["directions"]), data["ends"]
            return bins, labels, data["ends"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/tactile")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=20261106, type=int)
    parser.add_argument("--method", choices=("mixed", "phase"), default="mixed")
    parser.add_argument("--unified", action="store_true")
    parser.add_argument("--bin-ms", type=int, choices=(50,100), default=100)
    args = parser.parse_args()
    trials = [t for t in read_trials(args.data) if t.labels["task"] in ("slide", "background", "static_contact")]
    if args.unified:
        from .train_gelneuro_unified import prepare, task_of, IGNORE
        trials, _ = prepare(args.data, args.bin_ms * 1000)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source=str(Path(args.data).resolve()), seed=args.seed, bin_us=args.bin_ms * 1000, unified=args.unified,
                    method=args.method, recordings={})
    skipped = []
    for index, trial in enumerate(trials):
        entries = []
        for variant in range(len(TIME_SCALES)):
            try:
                bins, labels, ends, meta = generate(
                    trial, variant, np.random.default_rng([args.seed, index, variant]), args.method, args.bin_ms * 1000)
            except ValueError as error:
                if not str(error).startswith("No complete supervised bins:"):
                    raise
                skipped.append(dict(trial=trial.name, variant=variant, reason=str(error)))
                continue
            name = f"{trial.name}-aug{variant}.npz"
            arrays = dict(bins=bins, labels=labels, ends=ends)
            if args.unified:
                valid = labels != IGNORE
                arrays["directions"] = labels if task_of(trial) == 1 else np.full_like(labels, IGNORE)
                arrays["labels"] = np.where(valid, task_of(trial), IGNORE)
            np.savez_compressed(args.out / name, **arrays)
            entries.append(dict(file=name, **meta))
        manifest["recordings"][trial.name] = dict(source_sha256=source_signature(trial),
                                                   source_labels=trial.labels, variants=entries)
    manifest["skipped"] = skipped
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(out=str(args.out), sources=len(trials),
                          variants=sum(len(r["variants"]) for r in manifest["recordings"].values()),
                          skipped=len(skipped))))


if __name__ == "__main__":
    main()
