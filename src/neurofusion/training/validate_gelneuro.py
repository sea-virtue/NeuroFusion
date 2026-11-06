"""Evaluate held-out GelNeuro recordings, temporal ablations and challenge traces."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import torch

from .gelneuro_data import DIRECTIONS, Windows, read_trials, split_trials
from .train_gelneuro import configure_device, load_model, loader


@torch.inference_mode()
def predict(model, dataset, device, batch_size):
    records, elapsed = [], []
    for values, labels, trials, ends in loader(dataset, batch_size, seed=0):
        start = time.perf_counter()
        logits, spikes = model(values.to(device))
        probabilities = logits[:, 0].sigmoid().cpu().tolist()
        directions = logits[:, 1:].argmax(1).cpu().tolist()
        activity = spikes.cpu().tolist() if spikes is not None else [None] * len(values)
        elapsed.append(1e3 * (time.perf_counter() - start))
        counts = values.flatten(1).ne(0).any(1).tolist()
        for index, (label, trial, end) in enumerate(zip(labels.tolist(), trials.tolist(), ends.tolist())):
            recording = dataset.trials[trial]
            records.append(dict(trial=recording.name, task=recording.labels["task"],
                                speed=recording.labels.get("speed_level"),
                                device_window_end_us=end, target_direction=label,
                                nonempty_input=counts[index], slide_probability=probabilities[index],
                                direction=DIRECTIONS[directions[index]], output_spikes=activity[index]))
    return records, dict(batch_size=batch_size, batches=len(elapsed),
                         p50_ms=float(np.percentile(elapsed, 50)),
                         p95_ms=float(np.percentile(elapsed, 95)))


def metrics(records):
    positive = [r for r in records if r["target_direction"] >= 0]
    quiet = [r for r in records if r["target_direction"] < 0]
    if not positive or not quiet:
        raise ValueError("Held-out evaluation needs slide and quiet windows")
    confusion = np.zeros((4, 4), dtype=np.int64)
    for row in positive:
        confusion[row["target_direction"], DIRECTIONS.index(row["direction"])] += 1
    denominator = confusion.sum(0) + confusion.sum(1)
    scores = 2 * confusion.diagonal() / denominator.clip(min=1)
    return dict(windows=len(records), slide_windows=len(positive), quiet_windows=len(quiet),
                direction_macro_f1=float(scores.mean()),
                direction_f1=dict(zip(DIRECTIONS, scores.tolist())),
                direction_confusion=confusion.tolist(),
                slide_recall=statistics.mean(r["slide_probability"] >= .5 for r in positive),
                quiet_false_positive=statistics.mean(r["slide_probability"] >= .5 for r in quiet),
                nonempty_slide_fraction=statistics.mean(r["nonempty_input"] for r in positive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", default="data/raw/tactile")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--replay-out", type=Path, help="CSV of scores on separate unlabelled challenge recordings")
    parser.add_argument("--replay-windows", type=int, default=160)
    args = parser.parse_args()
    if min(args.windows, args.replay_windows, args.batch_size, args.threads) < 1:
        parser.error("windows, batch size and threads must be positive")
    paths = [path.resolve() for path in (args.out, args.replay_out) if path]
    if len(paths) != len(set(paths)) or any(path.exists() for path in paths):
        parser.error("Output paths must be distinct new files")
    device = configure_device(args.device, args.threads)
    model, saved = load_model(args.checkpoint, device)
    model.eval()
    reviewed = None
    if saved["reviewed"] is not None:
        with tempfile.TemporaryDirectory() as temporary:
            reviewed = Path(temporary) / "reviews.json"
            reviewed.write_text(saved["reviewed"])
            trials = read_trials(args.data, reviewed)
    else:
        trials = read_trials(args.data)
    train, heldout, replay = split_trials(trials, saved["fold"])
    if ([t.name for t in train] != saved["train_trials"]
            or [t.name for t in heldout] != saved["heldout_trials"]):
        raise ValueError("Current recording split does not match checkpoint")
    options = (heldout, saved["kind"], saved["bin_us"], args.windows, saved["seed"])
    records, latency = predict(model, Windows(*options), device, args.batch_size)
    report = dict(checkpoint=str(args.checkpoint), fold=saved["fold"], kind=saved["kind"],
                  bin_us=saved["bin_us"], probe_of=saved.get("probe_of"),
                  timing_scope="Model + host-device transfer, excluding loading/binning and SDK; synchronous batch.",
                  supervision="Weak operator conditions; direction metrics conditional on slide-labelled windows.",
                  normal=metrics(records), latency=latency, predictions=records)
    if args.shuffled:
        shuffled, _ = predict(model, Windows(*options, shuffle_time=True), device, args.batch_size)
        report["shuffled"] = metrics(shuffled)
    if args.replay_out:
        windows = Windows(replay, saved["kind"], saved["bin_us"], args.replay_windows)
        trace, _ = predict(model, windows, device, args.batch_size)
        args.replay_out.parent.mkdir(parents=True, exist_ok=True)
        with args.replay_out.open("x", newline="", encoding="utf-8") as stream:
            fields = [field for field in trace[0] if field != "target_direction"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
