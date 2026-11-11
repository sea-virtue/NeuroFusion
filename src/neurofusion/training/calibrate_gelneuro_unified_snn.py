"""Calibrate only the host affine decoder using frozen integer event-driven features."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..models.gelneuro_unified_snn import GelNeuroUnifiedSNN
from .train_gelneuro_unified import prepare, evaluate, CLASSES, DIRECTIONS
from .train_gelneuro_unified_snn import aggregate
from .validate_gelneuro_unified_snn import make_specksim, specksim_features


class HeadOnly(nn.Module):
    def __init__(self,head):
        super().__init__()
        self.head=head
    def forward(self,features):
        logits=self.head(features)
        return logits[...,:4],logits[...,4:],None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints",type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--data",type=Path,default=Path("data/raw/tactile"))
    p.add_argument("--fold",type=int,choices=range(3))
    p.add_argument("--fit-all",action="store_true")
    p.add_argument("--epochs",type=int,default=600)
    p.add_argument("--seed",type=int,default=31)
    args=p.parse_args()
    if args.fold is None and not args.fit_all:
        p.error("Specify --fold or --fit-all; separate folds can extract independently")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    fold=args.fold if not args.fit_all else -1
    name=f"fold{fold}" if fold>=0 else "all-data"
    prefix=args.checkpoints/name
    if (args.out/f"{name}.pt").exists() or (args.out/f"{name}.json").exists():
        raise FileExistsError("Calibration output already exists; choose a new path")
    saved=torch.load(prefix.with_suffix(".pt"),map_location="cpu",weights_only=False)
    info=json.loads(prefix.with_suffix(".json").read_text())
    trials,sequences=prepare(args.data,50_000)
    if fold>=0:
        assert set(info["train_trials"]).isdisjoint(info["heldout_trials"])
        assert set(info["train_trials"])|set(info["heldout_trials"])=={t.name for t in trials}
    model=GelNeuroUnifiedSNN(1000).eval()
    model.load_state_dict(saved["model"])
    model.decay.copy_((-1./torch.tensor([50.,150.,500.,1500.])).exp().reshape(1,4,1))
    sim=make_specksim(model)
    args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/f"{name}-native-cache"
    cache.mkdir(exist_ok=True)
    provenance=dict(checkpoint=str(prefix.with_suffix(".pt").resolve()),
        checkpoint_sha256=hashlib.sha256(prefix.with_suffix(".pt").read_bytes()).hexdigest(),
        feature_step_us=1000,readout_us=50000,integer_backbone=True,
        train_trials=info["train_trials"],heldout_trials=info["heldout_trials"],
        scope="Frozen encoder; all fitting and normalization use training IDs only")
    manifest=cache/"manifest.json"
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        if prior["checkpoint_sha256"]!=provenance["checkpoint_sha256"]:
            raise ValueError("Cached features belong to another checkpoint")
    elif list(cache.glob("*.npz")):
        raise ValueError("Partial cache has no provenance; refusing implicit reuse")
    manifest.write_text(json.dumps(provenance,indent=2))
    features={}
    started=time.perf_counter()
    firing=[]
    for index,trial in enumerate(trials):
        _,target,direction,ends=sequences[trial.name]
        path=cache/f"{trial.name}.npz"
        if path.exists():
            with np.load(path) as saved_features:
                assert np.array_equal(saved_features["target"],target.numpy())
                assert np.array_equal(saved_features["direction"],direction.numpy())
                assert np.array_equal(saved_features["ends"],ends)
                z=torch.from_numpy(saved_features["features"].copy())
                count=saved_features["output_spikes"].tolist()
        else:
            z,count=specksim_features(model,sim,trial,ends,1000,torch.device("cpu"))
            np.savez_compressed(path,features=z.numpy(),target=target.numpy(),
                                 direction=direction.numpy(),ends=ends,output_spikes=np.asarray(count))
        features[trial.name]=(z,target,direction,ends)
        firing.extend(count)
        if (index+1)%10==0:
            print(json.dumps(dict(fold=fold,extracted=index+1,total=len(trials))),flush=True)
    extraction_seconds=time.perf_counter()-started
    (cache/"manifest.json").write_text(json.dumps(provenance,indent=2))
    train=[t for t in trials if t.name in info["train_trials"]]
    heldout=[t for t in trials if t.name in info["heldout_trials"]]
    x=torch.cat([features[t.name][0][features[t.name][1]>=0] for t in train])
    y=torch.cat([features[t.name][1][features[t.name][1]>=0] for t in train])
    d=torch.cat([features[t.name][2][features[t.name][1]>=0] for t in train])
    mean=x.mean(0)
    std=x.std(0,unbiased=False).clamp_min(.05)
    normalized=(x-mean)/std
    head=nn.Linear(64,8)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    started=time.perf_counter()
    history=[]
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits=head(normalized)
        task_loss=torch.stack([F.cross_entropy(logits[y==c,:4],y[y==c]) for c in range(4)]).mean()
        slide=d>=0
        loss=task_loss+F.cross_entropy(logits[slide,4:],d[slide])
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    fitting_seconds=time.perf_counter()-started
    with torch.no_grad():
        # Fold training normalization into a single affine host decoder.
        model.head.weight.copy_(head.weight/std)
        model.head.bias.copy_(head.bias-(head.weight*mean/std).sum(1))
    model.eval()
    decoder=HeadOnly(model.head).eval()
    result=dict(fold=fold,epochs=args.epochs,seed=args.seed,lr=.01,weight_decay=.01,
        extraction_seconds=extraction_seconds,fitting_seconds=fitting_seconds,loss=history,
        normalization_mean=mean.tolist(),normalization_std=std.tolist(),
        train_trials=info["train_trials"],heldout_trials=info["heldout_trials"],
        source_checkpoint=str(prefix.with_suffix(".pt")),provenance=provenance,
        scope="Integer event-driven frozen SNN plus train-only calibrated affine host decoder",
        firing=dict(mean_per_second=float(np.mean(firing))*20,
                    p95_per_50ms=float(np.percentile(firing,95)),max_per_50ms=max(firing)))
    if heldout:
        result.update(evaluate(decoder,heldout,features,torch.device("cpu")))
        result["training_diagnostic"]={k:v for k,v in evaluate(decoder,train,features,torch.device("cpu")).items()
                                       if k not in ("predictions","recording_predictions")}
    # Save a 50ms model definition, retaining native calibration metadata.
    target=GelNeuroUnifiedSNN(50_000).eval()
    state=model.state_dict()
    state["decay"]=target.decay
    target.load_state_dict(state)
    saved.update(model=target.state_dict(),bin_us=50000,native_feature_step_us=1000,
        host_decoder_calibrated=True,source_checkpoint=str(prefix.with_suffix(".pt")))
    with (args.out/f"{name}.pt").open("xb") as f: torch.save(saved,f)
    with (args.out/f"{name}.json").open("x") as f:f.write(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("fold","extraction_seconds","fitting_seconds","scope")}),flush=True)
    if heldout:
        print(json.dumps(dict(task_f1=result["task"]["macro_f1"],direction_f1=result["direction"]["macro_f1"],
                              joint=result["slide_joint_recall"])),flush=True)


if __name__=="__main__":
    main()
