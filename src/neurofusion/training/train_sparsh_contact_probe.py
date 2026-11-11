"""Fit a frozen Sparsh linear probe on matched light/medium slow slides."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from ..capture.reader import rows, safe_path
from ..models.encoders import ImageBackbone, prepare_sparsh


CLASSES = ("light", "medium")
HISTORY = 5
GUARD_NS = 550_000_000


def eligible_recordings(root):
    for folder in sorted(root.iterdir()):
        if not (folder / "manifest.json").is_file():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        labels = manifest["config"]["labels"]
        task = labels["task"]
        matched_slide = (task == "slide" and labels["object_id"] == "miscrewdriverbox"
                         and labels["speed_level"] == "slow" and labels["contact_level"] in CLASSES)
        if not matched_slide and task not in ("static_contact", "press_release", "background"):
            continue
        if manifest["status"] != "complete" or manifest.get("synthetic") or not (
            json.loads((folder / "quality.json").read_text())["passed"]
        ):
            raise ValueError(f"Recording not eligible: {folder.name}")
        yield folder, labels


def load_rgb(folder, frame):
    with Image.open(safe_path(folder, frame["path"])) as image:
        return np.array(image.convert("RGB"))


def time_pairs(frames, start_ns, stop_ns, limit):
    usable = [i for i in range(HISTORY, len(frames))
              if start_ns <= frames[i - HISTORY]["host_received_ns"]
              and frames[i]["host_received_ns"] < stop_ns]
    if not usable:
        raise ValueError("No causal frame pair within selected interval")
    selected = np.linspace(0, len(usable)-1, min(limit, len(usable)), dtype=int)
    return [usable[i] for i in selected]


def extract(folder, labels, model, background, device):
    frames = list(rows(folder / "gelsight/frames.jsonl"))
    marks = list(rows(folder / "annotations.jsonl"))
    start = next(m["host_marked_ns"] for m in marks if m["name"] == "recording_start")
    end = frames[-1]["host_received_ns"] + 1
    groups = {}
    if labels["task"] == "slide":
        begin = next(m["host_marked_ns"] for m in marks if m["name"].startswith("DO SLIDE"))
        stop = next(m["host_marked_ns"] for m in marks if m["name"].startswith("STOP MOTION"))
        groups["active"] = time_pairs(frames, begin + GUARD_NS, stop - GUARD_NS, 16)
        groups["pre_motion_hold"] = time_pairs(frames, start, begin - 150_000_000, 10)
    else:
        groups["replay"] = time_pairs(frames, start, end, 48)

    indexes = sorted(set().union(*groups.values()))
    features = {}
    baseline = {}
    with torch.inference_mode():
        for first in range(0, len(indexes), 8):
            batch = indexes[first:first+8]
            inputs = {i: load_rgb(folder, frames[i]) for i in set(batch) | {i - HISTORY for i in batch}}
            tensors = torch.stack([prepare_sparsh(inputs[i], inputs[i - HISTORY], background)
                                   for i in batch])
            values = model(tensors.to(device)).cpu()
            image_change = (tensors[:, :3] - 127/255).abs().mean(1)
            summary = torch.stack([image_change.mean((1,2)),
                                   (image_change > .06).float().mean((1,2)),
                                   (tensors[:, :3] - tensors[:, 3:]).abs().mean((1,2,3))], dim=1)
            for i, feature, simple in zip(batch, values, summary):
                features[i], baseline[i] = feature.numpy().copy(), simple.numpy().copy()
    return {key: dict(
        features=np.stack([features[i] for i in ids]),
        baseline=np.stack([baseline[i] for i in ids]),
        time_s=np.asarray([(frames[i]["host_received_ns"] - start) / 1e9 for i in ids]),
        historical_lag_ms=np.asarray([(frames[i]["host_received_ns"] -
                                       frames[i-HISTORY]["host_received_ns"]) / 1e6 for i in ids]),
    ) for key, ids in groups.items()}, [
        dict(name=m["name"], time_s=(m["host_marked_ns"]-start)/1e9)
        for m in marks if m["source"] == "timed_instruction_not_measured_motion"
    ]


def split_recordings(records):
    groups = defaultdict(list)
    for item in records:
        label = item["labels"]
        groups[(label["contact_level"], label["direction"])].append(item)
    return {item["id"]: i % 3 for group in groups.values()
            for i, item in enumerate(sorted(group, key=lambda r: (r["labels"]["repetition"], r["id"])))}


def fit(features, target):
    x = torch.as_tensor(np.asarray(features), dtype=torch.float64)
    y = torch.tensor(target, dtype=torch.float64)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-4)
    x = (x - mean) / scale
    head = nn.Linear(x.shape[-1], 1, dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    weights = torch.where(y == 1, .5 / y.mean(), .5 / (1-y.mean()))
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=200, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(head(x).flatten(), y, weight=weights)
        loss = loss + .05 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        weight = (head.weight[0]/scale).float()
        bias = (head.bias[0] - (head.weight[0]*mean/scale).sum()).float()
    return {"weight": weight, "bias": bias}


def logits(head, x):
    return (torch.as_tensor(x) @ head["weight"] + head["bias"]).numpy()


def metrics(labels, predictions):
    matrix = np.zeros((2,2), dtype=int)
    for a,b in zip(labels,predictions):
        matrix[int(a), int(b)] += 1
    if int(matrix.sum()) != len(labels) or np.any(matrix.sum(1) == 0):
        raise ValueError("Expected one prediction per recording and both classes in evaluation")
    recall = matrix.diagonal() / matrix.sum(1)
    return dict(confusion=matrix.tolist(), correct=int(matrix.trace()), total=int(matrix.sum()),
                balanced_accuracy=float(recall.mean()), recall=dict(zip(CLASSES, recall.tolist())))


def save_head(path, head, reference, training_ids, scope):
    with path.open("xb") as stream:
        torch.save(dict(**head, classes=CLASSES, encoder="sparsh_dino_small", feature_dim=384,
                        history_stride_frames=HISTORY, reference=reference,
                        training_ids=training_ids, scope=scope), stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/tactile"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    recordings = list(eligible_recordings(args.data))
    ref_folder = next(d for d,l in recordings if l["task"] == "background")
    ref_frames = list(rows(ref_folder / "gelsight/frames.jsonl"))
    ref_meta = ref_frames[len(ref_frames)//2]
    reference = str(Path(ref_folder.name) / ref_meta["path"])
    background = load_rgb(ref_folder, ref_meta)
    model = ImageBackbone("sparsh_dino_small", pretrained=True, frozen=True).to(device).eval()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "background.jpg").write_bytes(safe_path(ref_folder, ref_meta["path"]).read_bytes())
    records, replay = [], []
    for folder,labels in recordings:
        features, marks = extract(folder, labels, model, background, device)
        item = dict(id=folder.name, labels=labels, features=features, marks=marks)
        (records if labels["task"] == "slide" else replay).append(item)
        np.savez_compressed(args.out/f"{folder.name}.npz",
                            **{f"{g}_{k}": v for g,x in features.items() for k,v in x.items()})
        print(f"Extracted {folder.name}: { {k: len(v['time_s']) for k,v in features.items()} }", flush=True)
    folds = split_recordings(records)
    target = np.asarray([CLASSES.index(r["labels"]["contact_level"]) for r in records])
    predictors = {
        "sparsh_active": ("active", "features"),
        "image_baseline_active": ("active", "baseline"),
        "sparsh_pre_motion_hold": ("pre_motion_hold", "features"),
    }
    all_scores = {}
    all_rows = [dict(id=r["id"], direction=r["labels"]["direction"],
                     true=CLASSES[t], fold=folds[r["id"]]) for r,t in zip(records,target)]
    for name,(group,field) in predictors.items():
        x = np.stack([r["features"][group][field].mean(0) for r in records])
        scores = np.empty(len(records))
        for fold in range(3):
            test = np.asarray([folds[r["id"]] == fold for r in records])
            head = fit(x[~test], target[~test])
            scores[test] = logits(head, x[test])
            if name == "sparsh_active":
                save_head(args.out/f"fold{fold}.pt", head, reference,
                          [r["id"] for r, heldout in zip(records,test) if not heldout],
                          "Training fold only")
        all_scores[name] = metrics(target, scores >= 0)
        for row,score in zip(all_rows,scores):
            row[f"{name}_p_medium"] = float(torch.sigmoid(torch.tensor(score)))
            row[f"{name}_predicted"] = CLASSES[int(score>=0)]
        if name == "sparsh_active":
            final = fit(x,target)
            save_head(args.out/"all-data.pt", final, reference, [r["id"] for r in records],
                      "Fit all 25 slow slides, no independent score")
    with (args.out/"predictions.csv").open("x", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    replay_rows=[]
    for r in replay:
        f=r["features"]["replay"]
        z=f["features"]
        distances=np.linalg.norm(z - z[:5].mean(0), axis=1)
        scores=torch.sigmoid(torch.from_numpy(logits(final,z))).numpy()
        for t, score, distance, simple in zip(f["time_s"],scores,distances,f["baseline"]):
            replay_rows.append(dict(id=r["id"], task=r["labels"]["task"], seconds=float(t),
                                    p_medium=float(score), distance_from_early_feature=float(distance),
                                    mean_image_change=float(simple[0])))
    with (args.out/"static_replay.csv").open("x", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(replay_rows[0]))
        writer.writeheader()
        writer.writerows(replay_rows)
    summary=dict(reference=reference, history_stride_frames=HISTORY, guard_ms=GUARD_NS/1e6,
                 l2_coefficient=.05, device=str(device), class_counts=dict(Counter(target.tolist())),
                 metrics=all_scores, predictions=all_rows,
                 replay_marks={r["id"]:r["marks"] for r in replay},
                 replay_labels={r["id"]:r["labels"] for r in replay},
                 historical_lag_ms_median=float(np.median(np.concatenate([
                     r["features"]["active"]["historical_lag_ms"] for r in records]))),
                 hold_control_scope="HOLD STILL; contact unknown; not a verified cross-day unloaded control",
                 static_scope="Out-of-task scores; no calibrated load labels")
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(all_scores,indent=2))


if __name__ == "__main__":
    main()
