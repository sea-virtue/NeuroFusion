"""Train or benchmark GelNeuro features on grouped tactile recordings."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .gelneuro_data import DIRECTIONS, Windows, read_trials, split_trials
from ..models.events import PacketEncoder
from ..models.gelneuro_snn import GelNeuroSNN, SpikeReadout


class EventTasks(nn.Module):
    def __init__(self, kind, bin_us):
        super().__init__()
        self.encoder = GelNeuroSNN() if kind == "snn" else PacketEncoder()
        self.readout = SpikeReadout(bin_us) if kind == "snn" else nn.Identity()
        self.head = nn.Linear(64, 1 + len(DIRECTIONS))
        self.kind = kind

    def forward(self, values):
        if self.kind == "snn":
            spikes, _ = self.encoder(values)
            features = self.readout(spikes)
            activity = spikes.sum((1, 2, 3, 4))
        else:
            features = self.encoder(values)
            activity = None
        return self.head(features), activity


def task_loss(logits, direction, positive_weight):
    sliding = direction >= 0
    activity = F.binary_cross_entropy_with_logits(
        logits[:, 0], sliding.to(logits.dtype), pos_weight=positive_weight)
    directions = F.cross_entropy(logits[:, 1:], direction.clamp_min(0), reduction="none")
    return activity + (directions * sliding).sum() / sliding.sum().clamp_min(1)


def configure_device(name, threads):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable in this Python. Check torch.version.cuda or pass --device cpu.")
    if device.type == "cpu":
        torch.set_num_threads(threads)
    return device


def loader(data, batch_size, seed, shuffle=False):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def update_batch(model, batch, optimizer, weight, device, spike_penalty=0.0, spike_budget=64):
    values, labels, _, _ = batch
    optimizer.zero_grad(set_to_none=True)
    logits, activity = model(values.to(device))
    loss = task_loss(logits, labels.to(device), weight)
    if spike_penalty:
        loss = loss + spike_penalty * (activity / spike_budget - 1).relu().mean()
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm.detach()), None if activity is None else float(activity.detach().mean())


def load_model(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = EventTasks(saved["kind"], saved["bin_us"]).to(device)
    model.load_state_dict(saved["model"])
    return model, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/tactile")
    parser.add_argument("--reviewed", help="JSON mapping recording name to [start_us, end_us)")
    parser.add_argument("--kind", choices=("snn", "cnn"), default="snn")
    parser.add_argument("--bin-us", choices=(100, 250, 500, 1000), type=int, default=250)
    parser.add_argument("--fold", choices=range(3), type=int, default=0)
    parser.add_argument("--windows", type=int, default=16, help="Training windows per recording per epoch")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, help="New checkpoint path")
    parser.add_argument("--freeze-encoder", type=Path, help="Fit a fresh linear probe on a trained frozen representation")
    parser.add_argument("--benchmark-steps", type=int, default=0, help="Short benchmark, discard weights")
    parser.add_argument("--spike-penalty", type=float, default=0.0)
    parser.add_argument("--spike-budget", type=int, default=64, help="Soft budget of output spikes per 32 ms window")
    args = parser.parse_args()
    if args.spike_budget < 1 or args.spike_penalty < 0 or (args.spike_penalty and args.kind != "snn"):
        parser.error("Spike penalty requires SNN and positive spike budget")
    if min(args.windows, args.epochs, args.batch_size, args.threads) < 1 or args.benchmark_steps < 0:
        parser.error("windows, epochs, batch size and threads must be positive")
    if not args.benchmark_steps and (args.out is None or args.out.exists()):
        parser.error("Training requires --out pointing to a new file")
    device = configure_device(args.device, args.threads)
    torch.manual_seed(args.seed)
    trials = read_trials(args.data, args.reviewed)
    train, heldout, _ = split_trials(trials, args.fold)
    metadata = dict(kind=args.kind, bin_us=args.bin_us, fold=args.fold,
                    train_trials=[trial.name for trial in train],
                    heldout_trials=[trial.name for trial in heldout],
                    reviewed=Path(args.reviewed).read_text() if args.reviewed else None)
    if args.freeze_encoder:
        model, saved = load_model(args.freeze_encoder, device)
        if any(saved[key] != value for key, value in metadata.items()):
            parser.error("Probe data, model settings, and fold must match the checkpoint")
        model.encoder.requires_grad_(False)
        model.readout.requires_grad_(False)
        model.head.reset_parameters()
    else:
        model = EventTasks(args.kind, args.bin_us).to(device)
    data = Windows(train, args.kind, args.bin_us, args.windows, args.seed, jitter=True)
    batches = loader(data, args.batch_size, args.seed, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    quiet = sum(trial.direction < 0 for trial in train)
    weight = torch.tensor(quiet / (len(train) - quiet), device=device)
    print(json.dumps(dict(device=str(device), train_trials=len(train),
                          heldout_trials=len(heldout), windows=len(data),
                          updates_per_epoch=len(batches))), flush=True)
    update = lambda batch: update_batch(model, batch, optimizer, weight, device,
                                        args.spike_penalty, args.spike_budget)
    if args.benchmark_steps:
        from itertools import islice
        # Cycle iterators, not batches: each dense batch can occupy tens of MB.
        def infinite_batches():
            while True:
                yield from batches
        stream = infinite_batches()
        warmup = update(next(stream))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for batch in islice(stream, args.benchmark_steps):
            last = update(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        average = (time.perf_counter() - start) / args.benchmark_steps
        print(json.dumps(dict(mean_update_seconds=average,
                              estimated_epoch_seconds=average * len(batches),
                              estimated_run_seconds=average * len(batches) * args.epochs,
                              peak_cuda_mib=torch.cuda.max_memory_allocated(device) / 2**20
                              if device.type == "cuda" else None,
                              warmup=warmup, last=last)), flush=True)
        return
    history = []
    for epoch in range(args.epochs):
        data.epoch = epoch
        start = time.perf_counter()
        scores = [update(batch) for batch in batches]
        report = dict(epoch=epoch + 1, loss=sum(score[0] for score in scores) / len(scores),
                      seconds=time.perf_counter() - start,
                      mean_output_spikes=sum(score[2] for score in scores) / len(scores)
                      if args.kind == "snn" else None)
        history.append(report)
        print(json.dumps(report), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        torch.save(dict(metadata, model=model.state_dict(), history=history,
                        seed=args.seed, windows_per_trial=args.windows, lr=args.lr,
                        spike_penalty=args.spike_penalty, spike_budget=args.spike_budget,
                        probe_of=str(args.freeze_encoder) if args.freeze_encoder else None), stream)


if __name__ == "__main__":
    main()
