"""Train and evaluate the causal Speck-shaped unified SNN on recording-level folds."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..models.gelneuro_unified_snn import GelNeuroUnifiedSNN
from .augment_gelneuro_temporal import AugmentedRecordings
from .train_gelneuro_temporal import configure_device
from .train_gelneuro_unified import (
    prepare, split, collate, evaluate, classification, CLASSES, DIRECTIONS, IGNORE,
    UnifiedTactileEncoder,
)


class BalancedBatches:
    """One recording of each task per batch, with a fixed number of updates."""
    def __init__(self, originals, seed):
        self.groups=[[] for _ in range(4)]
        for index,(_,target,_,_) in enumerate(originals):
            self.groups[int(target[target>=0][0])].append(index)
        self.steps=math.ceil(len(originals)/4)
        self.seed=seed
        self.epoch=0
    def __len__(self):
        return self.steps
    def __iter__(self):
        rng=np.random.default_rng([self.seed,self.epoch])
        for _ in range(self.steps):
            batch=[int(rng.choice(group)) for group in self.groups]
            rng.shuffle(batch)
            yield batch

def raw_counts(values):
    return values.expm1().round().clamp_min(0)


class CountDataset(Dataset):
    def __init__(self, source):
        self.source = source
    def __len__(self):
        return len(self.source)
    def __getitem__(self, i):
        values, task, direction, ends = self.source[i]
        return raw_counts(values),task,direction,ends


def distillation(student, teacher, mask, temperature=2.):
    if not mask.any():
        return student.sum()*0
    return F.kl_div(F.log_softmax(student[mask]/temperature,-1),
                    F.softmax(teacher[mask]/temperature,-1),
                    reduction="batchmean")*temperature**2


def aggregate(results):
    pooled={}
    for key,names in (("task",CLASSES),("direction",DIRECTIONS),("recordings",CLASSES)):
        pooled[key]=classification(np.sum([r[key]["confusion"] for r in results],axis=0),names)
    total=sum(r["direction"]["by_class"][d]["support"] for r in results for d in DIRECTIONS)
    pooled["slide_joint_recall"]=sum(r["slide_joint_recall"]*sum(
        r["direction"]["by_class"][d]["support"] for d in DIRECTIONS) for r in results)/total
    return pooled


def run(args, fold, trials, logs, counts):
    device=configure_device(args.device,4)
    torch.manual_seed(args.seed)
    model=GelNeuroUnifiedSNN(args.bin_ms*1000).to(device)
    train,heldout=(trials,[]) if fold == -1 else split(trials,fold)
    if args.resume_root:
        prefix=args.resume_root/(f"fold{fold}" if fold>=0 else "all-data")
        previous=json.loads(prefix.with_suffix(".json").read_text())
        if set(previous["train_trials"])!={t.name for t in train}:
            raise ValueError("Resume training recordings differ")
        model.load_state_dict(torch.load(prefix.with_suffix(".pt"),map_location=device)["model"])
        model.decay.copy_((-args.bin_ms/torch.tensor([50.,150.,500.,1500.],device=device)).exp().reshape(1,4,1))
    if args.excitatory_only and not args.resume_root:
        with torch.no_grad():
            for conv in model.convolutions:
                weight=conv.weight.abs()
                conv.weight.copy_(weight*4./weight.sum((1,2,3),keepdim=True).clamp_min(1e-8))
    originals=[logs[t.name] for t in train]
    target_counts=torch.bincount(torch.cat([y[y>=0] for _,y,_,_ in originals]),minlength=4).float()
    weights=(target_counts.sum()/(4*target_counts)).to(device)
    source=originals
    if args.enhanced:
        source=AugmentedRecordings(args.enhanced,train,originals,args.seed,bin_us=args.bin_ms*1000)
    sampler=BalancedBatches(originals,args.seed) if args.balanced_batches else None
    if sampler is not None:
        loader=DataLoader(CountDataset(source),batch_sampler=sampler,collate_fn=collate)
    else:
        loader=DataLoader(CountDataset(source),batch_size=args.batch_size,shuffle=True,collate_fn=collate,
                          generator=torch.Generator().manual_seed(args.seed))
    teacher=None
    if args.teacher_root:
        prefix=args.teacher_root/(f"fold{fold}" if fold>=0 else "final/all-data")
        info=json.loads(prefix.with_suffix(".json").read_text()) if fold>=0 else json.loads(
            (args.teacher_root/"final/all-data-training.json").read_text())
        if set(info["train_trials"]) != {t.name for t in train}:
            raise ValueError("Teacher training recordings differ; refusing possible heldout leakage")
        if info["bin_us"] != args.bin_ms*1000:
            raise ValueError("Teacher bin width differs")
        teacher=UnifiedTactileEncoder().to(device).eval()
        teacher.load_state_dict(torch.load(prefix.with_suffix(".pt"),map_location=device)["model"])
        for p in teacher.parameters(): p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    history=[]
    total_epochs=args.warmup+args.transition+args.epochs
    torch.cuda.reset_peak_memory_stats() if device.type=="cuda" else None
    started=time.perf_counter()
    for epoch in range(total_epochs):
        if epoch < args.warmup:
            alpha=0.
        elif epoch < args.warmup+args.transition:
            alpha=(epoch-args.warmup+1)/args.transition
        else:
            alpha=1.
        model.train()
        if sampler is not None: sampler.epoch=epoch
        if args.enhanced: source.epoch=epoch
        losses=[]; rates=[]
        for x,task,direction in loader:
            x,task,direction=x.to(device),task.to(device),direction.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred,dirs,_=model(x,alpha=alpha)
            valid=task!=IGNORE; slide=direction>=0
            if args.balanced_batches:
                loss=torch.stack([F.cross_entropy(pred[task==c],task[task==c])
                                  for c in range(4) if (task==c).any()]).mean()
            else:
                loss=F.cross_entropy(pred[valid],task[valid],weight=weights)
            if slide.any(): loss=loss+F.cross_entropy(dirs[slide],direction[slide])
            if teacher is not None:
                with torch.no_grad(): tp,td,_=teacher(x.log1p())
                task_kd=torch.stack([distillation(pred,tp,task==c) for c in range(4)]).mean() if (
                    args.balanced_batches) else distillation(pred,tp,valid)
                loss=loss+args.distill*(task_kd+distillation(dirs,td,slide))
            # Soft software firing budget, not a hardware event-rate guarantee.
            loss=loss+args.spike_penalty*F.relu(model.last_activity*model.rate_scale-4.).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if args.excitatory_only:
                with torch.no_grad():
                    for conv in model.convolutions: conv.weight.clamp_(min=0.)
            losses.append(float(loss.detach()))
            rates.append(model.last_activity.detach().cpu().tolist())
        history.append(dict(epoch=epoch+1,alpha=alpha,loss=float(np.mean(losses)),
                            mean_spikes_per_neuron_bin=np.mean(rates,axis=0).tolist()))
        if epoch==0 or (epoch+1)%20==0 or epoch==total_epochs-1:
            print(json.dumps(dict(fold=fold,**history[-1])),flush=True)
    seconds=time.perf_counter()-started
    model.eval()
    name=f"fold{fold}" if fold>=0 else "all-data"
    checkpoint=dict(model=model.state_dict(),bin_us=args.bin_ms*1000,kind="unified-iaf-traces",
                    input_gain=model.input_gain,trace_tau_ms=[50,150,500,1500],
                    classes=CLASSES,directions=DIRECTIONS,config=vars_json(args))
    with (args.out/f"{name}.pt").open("xb") as f: torch.save(checkpoint,f)
    result=dict(fold=fold,config=vars_json(args),train_trials=[t.name for t in train],
                heldout_trials=[t.name for t in heldout],seconds=seconds,history=history,
                memory=model.resource_estimate(),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None)
    if heldout:
        result.update(evaluate(model,heldout,counts,device))
        quantized,quant=model.quantized_copy()
        quantized.eval()
        qresult=evaluate(quantized,heldout,counts,device)
        result["int8"]=qresult
        result["quantization"]=[{k:v for k,v in layer.items() if k!="weights"} for layer in quant]
        with (args.out/f"{name}-integer-parameters.pt").open("xb") as f: torch.save(quant,f)
        print(json.dumps(dict(fold=fold,seconds=seconds,task_f1=result["task"]["macro_f1"],
                             direction_f1=result["direction"]["macro_f1"],joint=result["slide_joint_recall"],
                             int8_task_f1=qresult["task"]["macro_f1"],int8_direction_f1=qresult["direction"]["macro_f1"])),flush=True)
    else:
        result["scope"]="All-data candidate; no independent validation score"
        _,quant=model.quantized_copy()
        with (args.out/f"{name}-integer-parameters.pt").open("xb") as f: torch.save(quant,f)
    (args.out/f"{name}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def vars_json(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/raw/tactile"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--enhanced",type=Path)
    p.add_argument("--teacher-root",type=Path)
    p.add_argument("--distill",type=float,default=.5)
    p.add_argument("--excitatory-only",action="store_true",help="Nonnegative core weights to reduce bin/event-order mismatch")
    p.add_argument("--resume-root",type=Path,help="Fold-matched SNN weights; optimizer starts fresh")
    p.add_argument("--balanced-batches",action="store_true",help="One recording per task; class-mean task losses")
    p.add_argument("--device",default="cuda")
    p.add_argument("--bin-ms",type=int,choices=(10,50),default=50)
    p.add_argument("--warmup",type=int,default=40)
    p.add_argument("--transition",type=int,default=20)
    p.add_argument("--epochs",type=int,default=120,help="Fully spiking training epochs after continuation")
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--spike-penalty",type=float,default=.001)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--seed",type=int,default=31)
    p.add_argument("--fold",type=int,choices=range(3))
    p.add_argument("--fit-all",action="store_true")
    args=p.parse_args()
    if min(args.epochs,args.batch_size)<1 or min(args.warmup,args.transition)<0:
        p.error("Invalid epoch count or batch size")
    if args.balanced_batches and args.batch_size!=4:
        p.error("Balanced batches currently require batch-size 4")
    trials,logs=prepare(args.data,args.bin_ms*1000)
    counts={k:(raw_counts(x),t,d,e) for k,(x,t,d,e) in logs.items()}
    print(json.dumps(dict(recordings=len(trials),config=vars_json(args),
                          resource=GelNeuroUnifiedSNN(args.bin_ms*1000).resource_estimate())),flush=True)
    if args.fold is not None:
        for suffix in (".pt",".json"):
            if (args.out/f"fold{args.fold}{suffix}").exists():
                raise FileExistsError("Fold output already exists; use a new path")
    args.out.mkdir(parents=True,exist_ok=args.fold is not None)
    results=[run(args,f,trials,logs,counts) for f in (
        [-1] if args.fit_all else [args.fold] if args.fold is not None else range(3))]
    if not args.fit_all:
        if args.fold is not None:
            present=[args.out/f"fold{f}.json" for f in range(3)]
            if all(path.exists() for path in present):
                results=[json.loads(path.read_text()) for path in present]
        pooled=aggregate(results)
        pooled["int8"]=aggregate([r["int8"] for r in results])
        pooled["folds"]=[dict(fold=r["fold"],task_f1=r["task"]["macro_f1"],
                              direction_f1=r["direction"]["macro_f1"],
                              joint=r["slide_joint_recall"],seconds=r["seconds"]) for r in results]
        (args.out/"summary.json").write_text(json.dumps(pooled,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
