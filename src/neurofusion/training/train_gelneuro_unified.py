"""One causal CNN/GRU encoder for quiet, slide, rotation and escape-wriggle."""
import argparse
from collections import defaultdict, Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..models.gelneuro_temporal_cnn import GelNeuroTemporalCNN
from .gelneuro_data import DIRECTIONS, read_trials, GUARD_NS
from .train_gelneuro_temporal import recording_bins, configure_device

CLASSES = ("quiet", "slide", "rotate", "escape_wriggle")
IGNORE = -2
WRIGGLE_IDS = {
    "20261106T075012Z_diagnostic_a83f5296",
    "20261106T075034Z_diagnostic_a6b5bd99",
    "20261106T075059Z_diagnostic_e533931c",
}


class UnifiedTactileEncoder(GelNeuroTemporalCNN):
    def __init__(self):
        super().__init__(recurrent=True)
        self.head = nn.Linear(64, 8)

    def forward(self, bins, state=None):
        features, state = self.encode(bins, state)
        logits = self.head(features)
        return logits[..., :4], logits[..., 4:], state


def task_of(trial):
    task = trial.labels["task"]
    if task in ("background", "static_contact"):
        return 0
    if task == "slide" and trial.labels["contact_level"] != "none":
        return 1
    if task == "rotate":
        return 2
    if trial.name in WRIGGLE_IDS:
        return 3
    return None


def prepare(root, bin_us):
    trials = []
    for trial in read_trials(root):
        task = task_of(trial)
        if task is None:
            continue
        if task == 2:
            folder = Path(root) / trial.name
            marks = [json.loads(line) for line in (folder / "annotations.jsonl").read_text().splitlines()]
            first = next(m["host_marked_ns"] for m in marks if m["name"].startswith("DO ROTATE"))
            last = next(m["host_marked_ns"] for m in marks if m["name"].startswith("STOP MOTION"))
            batches = [json.loads(line) for line in (folder / "events/batches.jsonl").read_text().splitlines()]
            selected = [b for b in batches if b["count"] and
                        first + GUARD_NS <= b["host_received_ns"] < last - GUARD_NS]
            if not selected:
                raise ValueError(f"No guarded rotation interval: {trial.name}")
            trial = replace(trial, start_us=selected[0]["device_first_us"],
                            end_us=selected[-1]["device_last_us"] + 1)
        if task == 3:
            # Operator confirmed these diagnostic records are mostly wriggling throughout.
            trial = replace(trial, start_us=trial.start_us + 550_000,
                            end_us=trial.end_us - 550_000)
        trials.append(trial)
    sequences = {}
    for trial in trials:
        bins, valid, ends = recording_bins(trial, bin_us=bin_us)
        task = task_of(trial)
        target = torch.where(valid != IGNORE, task, IGNORE)
        direction = valid if task == 1 else torch.full_like(valid, IGNORE)
        sequences[trial.name] = (bins, target, direction, ends)
    return trials, sequences


def split(trials, fold):
    groups = defaultdict(list)
    for trial in trials:
        task = trial.labels["task"]
        key = (task,)
        if task == "slide":
            key += (trial.labels["object_id"], trial.labels["contact_level"],
                    trial.labels["direction"], trial.labels["speed_level"])
        groups[key].append(trial)
    train, heldout = [], []
    for group in groups.values():
        group.sort(key=lambda t: (t.labels["repetition"], t.name))
        if len(group) < 3:
            raise ValueError("Each condition needs at least 3 independent recordings")
        for i, trial in enumerate(group):
            (heldout if i % 3 == fold else train).append(trial)
    assert {task_of(t) for t in train} == set(range(4))
    assert {task_of(t) for t in heldout} == set(range(4))
    return train, heldout


def collate(items):
    x, task, direction, _ = zip(*items)
    return (pad_sequence(x, batch_first=True),
            pad_sequence(task, batch_first=True, padding_value=IGNORE),
            pad_sequence(direction, batch_first=True, padding_value=IGNORE))


def classification(confusion, names):
    confusion = np.asarray(confusion)
    tp = confusion.diagonal()
    p = tp / confusion.sum(0).clip(min=1)
    r = tp / confusion.sum(1).clip(min=1)
    f = 2 * tp / (confusion.sum(0) + confusion.sum(1)).clip(min=1)
    return dict(macro_f1=float(f.mean()), accuracy=float(tp.sum()/confusion.sum()),
                by_class={k:dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                                 support=int(confusion[i].sum())) for i,k in enumerate(names)},
                confusion=confusion.tolist())


@torch.inference_mode()
def evaluate(model, trials, sequences, device):
    confusion = np.zeros((4,4), dtype=int)
    directions = np.zeros((4,4), dtype=int)
    record_confusion = np.zeros((4,4), dtype=int)
    joint, slides = 0, 0
    rows, records = [], []
    for trial in trials:
        values, target, direction, ends = sequences[trial.name]
        tasks, dirs, _ = model(values.unsqueeze(0).to(device))
        task_prob = tasks[0].softmax(-1).cpu()
        pred = task_prob.argmax(-1).tolist()
        d = dirs[0].argmax(-1).cpu().tolist()
        votes = []
        for label, y, a, b, end in zip(target.tolist(), direction.tolist(), pred, d, ends):
            if label == IGNORE:
                continue
            confusion[label,a] += 1
            votes.append(a)
            if y != IGNORE:
                directions[y,b] += 1
                slides += 1
                joint += int(a == 1 and y == b)
            rows.append(dict(recording=trial.name, end_us=int(end), task=CLASSES[label],
                             predicted_task=CLASSES[a], direction=DIRECTIONS[y] if y>=0 else None,
                             predicted_direction=DIRECTIONS[b]))
        majority = Counter(votes).most_common(1)[0][0]
        record_confusion[task_of(trial),majority] += 1
        records.append(dict(recording=trial.name, task=CLASSES[task_of(trial)],
                            majority=CLASSES[majority], bins=len(votes)))
    result = dict(task=classification(confusion, CLASSES), direction=classification(directions,DIRECTIONS),
                  slide_joint_recall=joint/slides,
                  recordings=classification(record_confusion,CLASSES),
                  recording_predictions=records, predictions=rows)
    return result


def run(args, fold, trials, sequences, out):
    torch.manual_seed(args.seed)
    device = configure_device(args.device, 4)
    model = UnifiedTactileEncoder().to(device)
    train, heldout = (trials, []) if fold == -1 else split(trials,fold)
    training = [sequences[t.name] for t in train]
    counts = torch.bincount(torch.cat([y[y>=0] for _,y,_,_ in training]),minlength=4).float()
    weights = (counts.sum()/(4*counts)).to(device)
    if args.enhanced:
        from .augment_gelneuro_temporal import AugmentedRecordings
        training = AugmentedRecordings(args.enhanced, train, training, args.seed,
                                        bin_us=args.bin_ms * 1000)
    batches = DataLoader(training,batch_size=4,collate_fn=collate,shuffle=True,
                         generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        if args.enhanced:
            training.epoch = epoch
        loss_sum = 0.
        for bins, task, direction in batches:
            bins,task,direction = bins.to(device),task.to(device),direction.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, dirs, _ = model(bins)
            valid = task != IGNORE
            task_loss = F.cross_entropy(logits[valid],task[valid],weight=weights)
            mask = direction >= 0
            direction_loss = F.cross_entropy(dirs.flatten(0,1), direction.clamp_min(0).flatten(),
                                             reduction="none").reshape_as(direction)
            loss = task_loss + (direction_loss*mask).sum()/mask.sum().clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())
        history.append(loss_sum/len(batches))
        if epoch==0 or (epoch+1)%40==0:
            print(json.dumps(dict(fold=fold,epoch=epoch+1,loss=history[-1])),flush=True)
    seconds = time.perf_counter()-start
    model.eval()
    if fold == -1:
        with (out/"all-data.pt").open("xb") as stream:
            torch.save(dict(model=model.state_dict(),bin_us=args.bin_ms*1000,classes=CLASSES,
                            directions=DIRECTIONS),stream)
        result=dict(scope="fit all eligible recordings; no independent score",
                    train_trials=[t.name for t in train],bin_us=args.bin_ms*1000,
                    epochs=args.epochs,seed=args.seed,enhanced=str(args.enhanced),loss=history,seconds=seconds,
                    task_weights=weights.cpu().tolist())
        (out/"all-data-training.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(scope=result["scope"],seconds=seconds,final_loss=history[-1])),flush=True)
        return result
    result=evaluate(model,heldout,sequences,device)
    result.update(fold=fold,epochs=args.epochs,bin_us=args.bin_ms*1000,seed=args.seed,
                  seconds=seconds,lr=3e-4,enhanced=str(args.enhanced),task_weights=weights.cpu().tolist(),loss=history,
                  train_trials=[t.name for t in train],heldout_trials=[t.name for t in heldout])
    with (out/f"fold{fold}.pt").open("xb") as f:
        torch.save(dict(model=model.state_dict(),bin_us=args.bin_ms*1000,classes=CLASSES,
                        directions=DIRECTIONS),f)
    (out/f"fold{fold}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(fold=fold,seconds=seconds,task=result["task"],direction=result["direction"],
                          joint=result["slide_joint_recall"],recordings=result["recordings"])),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data/raw/tactile")
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--bin-ms",type=int,choices=(50,100),default=50)
    parser.add_argument("--epochs",type=int,default=120)
    parser.add_argument("--seed",type=int,default=31)
    parser.add_argument("--enhanced",type=Path,help="Training-only unified augmented variants")
    parser.add_argument("--fit-all",action="store_true",help="Fit final checkpoint after cross-validation")
    args=parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    trials,sequences=prepare(args.data,args.bin_ms*1000)
    print(json.dumps(dict(recordings=len(trials),
        counts=dict(Counter(CLASSES[task_of(t)] for t in trials)))),flush=True)
    args.out.mkdir(parents=True,exist_ok=False)
    if args.fit_all:
        run(args,-1,trials,sequences,args.out)
        return
    results=[run(args,i,trials,sequences,args.out) for i in range(3)]
    pooled={}
    for key,names in (("task",CLASSES),("direction",DIRECTIONS),("recordings",CLASSES)):
        pooled[key]=classification(np.sum([r[key]["confusion"] for r in results],axis=0),names)
    pooled["folds"]=[dict(fold=r["fold"],task_f1=r["task"]["macro_f1"],
                         direction_f1=r["direction"]["macro_f1"],joint=r["slide_joint_recall"],
                         seconds=r["seconds"]) for r in results]
    pooled["slide_joint_recall"]=sum(
        r["slide_joint_recall"]*sum(r["direction"]["confusion"][i][j] for i in range(4) for j in range(4))
        for r in results)/sum(np.sum(r["direction"]["confusion"]) for r in results)
    (args.out/"summary.json").write_text(json.dumps(pooled,indent=2))


if __name__=="__main__":
    main()
