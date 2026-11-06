"""Train causal CNN/GRU and stateless CNN on fixed event bins."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..models.gelneuro_temporal_cnn import GelNeuroTemporalCNN
from .gelneuro_data import DIRECTIONS, read_trials, split_trials
from .train_gelneuro import configure_device
from .validate_gelneuro import metrics as basic_metrics


BIN_US = 100_000
IGNORE = -2


def recording_bins(trial, bounds=None, bin_us=BIN_US):
    events = trial.events
    origin, last = bounds if bounds is not None else (int(events[0, 2]), int(events[-1, 2]))
    # Keep short recordings in their original fold if a grid crosses both label edges.
    if bounds is None and (trial.end_us - origin) // bin_us <= (
            trial.start_us - origin + bin_us - 1) // bin_us:
        origin += (trial.start_us - origin) % bin_us
    size = (last + 1 - origin) // bin_us
    if size < 1:
        raise ValueError(f"Recording shorter than one bin: {trial.name}")
    step = (events[:, 2] - origin) // bin_us
    keep = (step >= 0) & (step < size)
    x, y, _, polarity = events[keep].T
    flat = ((step[keep] * 2 + polarity) * 32 + y // 4) * 32 + x // 4
    counts = np.bincount(flat, minlength=size * 2048).reshape(size, 2, 32, 32)
    starts = origin + np.arange(size) * bin_us
    ends = starts + bin_us
    labelled = (starts >= trial.start_us) & (ends <= trial.end_us)
    labels = np.where(labelled, trial.direction, IGNORE)
    if not labelled.any():
        raise ValueError(f"No complete supervised bins: {trial.name}")
    return (torch.from_numpy(np.log1p(counts.astype(np.float32))),
            torch.from_numpy(labels), ends)


def collate(batch):
    bins, labels, _ = zip(*batch)
    return pad_sequence(bins, batch_first=True), pad_sequence(labels, batch_first=True, padding_value=IGNORE)


def task_loss(logits, labels, positive_weight):
    valid = labels != IGNORE
    logits, labels = logits[valid], labels[valid]
    slide = labels >= 0
    activity = F.binary_cross_entropy_with_logits(logits[:, 0], slide.to(logits.dtype),
                                                  pos_weight=positive_weight)
    direction = F.cross_entropy(logits[:, 1:], labels.clamp_min(0), reduction="none")
    return activity + (direction * slide).sum() / slide.sum().clamp_min(1)


def score(rows):
    result = basic_metrics(rows)
    positives = [r for r in rows if r["target_direction"] >= 0]
    detected = [r for r in rows if r["slide_probability"] >= .5]
    correct = lambda r: r["direction"] == DIRECTIONS[r["target_direction"]]
    result["slide_precision"] = (sum(r["target_direction"] >= 0 for r in detected) / len(detected)
                                 if detected else None)
    result["direction_accuracy"] = sum(map(correct, positives)) / len(positives)
    result["joint_recall"] = sum(r["slide_probability"] >= .5 and correct(r) for r in positives) / len(positives)
    visible = [r for r in positives if r["nonempty_input"]]
    result["nonempty_direction_accuracy"] = sum(map(correct, visible)) / len(visible) if visible else None
    result["events_only_slide_recall"] = len(visible) / len(positives)
    result["events_only_quiet_false_positive"] = sum(
        r["nonempty_input"] for r in rows if r["target_direction"] < 0) / result["quiet_windows"]
    return result


@torch.inference_mode()
def evaluate(model, trials, sequences, device, reset_each_bin=False):
    rows = []
    for trial, (bins, labels, ends) in zip(trials, sequences):
        values = bins.to(device)
        logits, _ = model(values.unsqueeze(1) if reset_each_bin else values.unsqueeze(0))
        logits = logits.reshape(-1, 5)
        probabilities = logits[:, 0].sigmoid().tolist()
        directions = logits[:, 1:].argmax(1).tolist()
        active = bins.flatten(1).ne(0).any(1).tolist()
        for target, end, probability, direction, seen in zip(labels.tolist(), ends, probabilities, directions, active):
            if target == IGNORE:
                continue
            rows.append(dict(trial=trial.name, device_window_end_us=int(end), target_direction=target,
                             slide_probability=probability, direction=DIRECTIONS[direction], nonempty_input=seen))
    return rows


def run(args, device, fold, kind, train, heldout):
    torch.manual_seed(args.seed)
    model = GelNeuroTemporalCNN(recurrent=kind == "gru").to(device)
    training = [recording_bins(trial, bin_us=args.bin_ms * 1000) for trial in train]
    evaluation = [recording_bins(trial, bin_us=args.bin_ms * 1000) for trial in heldout]
    targets = torch.cat([y for _, y, _ in training])
    positive_weight = (targets.eq(-1).sum() / targets.ge(0).sum()).to(device)
    if args.enhanced:
        from .augment_gelneuro_temporal import AugmentedRecordings
        training = AugmentedRecordings(args.enhanced, train, training, args.seed)
    batches = DataLoader(training, batch_size=args.batch_size, collate_fn=collate, shuffle=True,
                         generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        if args.enhanced:
            training.epoch = epoch
        losses = []
        for bins, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(bins.to(device))
            loss = task_loss(logits, labels.to(device), positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        average = sum(losses) / len(losses)
        history.append(average)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(dict(fold=fold, kind=kind, epoch=epoch+1, loss=average)), flush=True)
    seconds = time.perf_counter() - started
    model.eval()
    rows = evaluate(model, heldout, evaluation, device)
    result = dict(fold=fold, kind=kind, bin_us=args.bin_ms * 1000, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed, seconds=seconds, positive_weight=float(positive_weight),
                  training_supervised_bins=int(targets.ne(IGNORE).sum()),
                  enhanced=str(args.enhanced) if args.enhanced else None,
                  train_trials=[t.name for t in train], heldout_trials=[t.name for t in heldout],
                  loss=history, metrics=score(rows), predictions=rows)
    if kind == "gru":
        result["reset_each_bin"] = score(evaluate(model, heldout, evaluation, device, reset_each_bin=True))
    prefix = args.out / f"fold{fold}-{kind}"
    with prefix.with_suffix(".pt").open("xb") as stream:
        torch.save(dict(kind=kind, bin_us=args.bin_ms * 1000, model=model.state_dict()), stream)
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(fold=fold, kind=kind, seconds=seconds, metrics=result["metrics"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/tactile")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--enhanced", type=Path, help="Training-only variants created by augment_gelneuro_temporal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bin-ms", choices=(50, 100, 150), type=int, default=100)
    parser.add_argument("--kind", choices=("both", "cnn", "gru"), default="both")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--fold", choices=range(3), type=int)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0:
        parser.error("Expected positive epochs, batch size and learning rate")
    if args.enhanced and args.bin_ms != 100:
        parser.error("--enhanced currently requires 100 ms bins")
    device = configure_device(args.device, 4)
    trials = read_trials(args.data)
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    kinds = ("cnn", "gru") if args.kind == "both" else (args.kind,)
    for fold in range(3) if args.fold is None else (args.fold,):
        train, heldout, _ = split_trials(trials, fold)
        for kind in kinds:
            results.append(run(args, device, fold, kind, train, heldout))
    summary = {}
    for kind in kinds:
        selected = [r for r in results if r["kind"] == kind]
        rows = [row for r in selected for row in r["predictions"]]
        summary[kind] = dict(pooled=score(rows), train_seconds=sum(r["seconds"] for r in selected),
                             folds=[dict(fold=r["fold"], metrics=r["metrics"],
                                         reset_each_bin=r.get("reset_each_bin")) for r in selected])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
