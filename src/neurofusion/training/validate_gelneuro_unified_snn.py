"""Replay original heldout events at finer simulation steps, scoring at CNN endpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch

from ..models.gelneuro_unified_snn import GelNeuroUnifiedSNN
from .train_gelneuro_unified import prepare, classification, CLASSES, DIRECTIONS, IGNORE, task_of
from .train_gelneuro_temporal import configure_device
from .train_gelneuro_unified_snn import aggregate


@torch.inference_mode()
def replay(model, trial, ends, step_us, device):
    readout_us=50_000
    if readout_us%step_us:
        raise ValueError("Replay step must divide the 50ms scoring period")
    ratio=readout_us//step_us
    start=int(ends[0])-readout_us
    stop=int(ends[-1])
    events=trial.events
    state=None; tasks=[]; dirs=[]; output_counts=[]
    # Bound intermediate tensors independently of recording length.
    for begin in range(start,stop,readout_us*5):
        finish=min(begin+readout_us*5,stop)
        n=(finish-begin)//step_us
        a,b=np.searchsorted(events[:,2],[begin,finish])
        e=events[a:b]
        idx=(((e[:,2]-begin)//step_us*2+e[:,3])*32+e[:,1]//4)*32+e[:,0]//4
        counts=np.bincount(idx,minlength=n*2048).reshape(n,2,32,32).astype(np.float32)
        x=torch.from_numpy(counts).unsqueeze(0).to(device)
        p,d,state=model(x,state)
        tasks.append(p[0,ratio-1::ratio].cpu())
        dirs.append(d[0,ratio-1::ratio].cpu())
        spikes=model.last_output_spikes[0].reshape(-1,ratio,16).sum((1,2)).cpu()
        output_counts.extend(spikes.tolist())
    return torch.cat(tasks),torch.cat(dirs),output_counts


def make_specksim(model):
    from torch import nn
    from sinabs.layers import IAFSqueeze
    from sinabs.backend.dynapcnn.specksim import from_sequential
    _,params=model.quantized_copy()
    layers=[]
    for conv,integer in zip(model.convolutions,params):
        clone=nn.Conv2d(conv.in_channels,conv.out_channels,conv.kernel_size,
                        conv.stride,conv.padding,bias=False)
        clone.weight.data.copy_(integer["weights"].float())
        layers.extend([clone,IAFSqueeze(batch_size=1,
            spike_threshold=float(integer["threshold_high"]),min_v_mem=float(integer["threshold_low"]))])
    return from_sequential(nn.Sequential(*layers),input_shape=(2,32,32))


@torch.inference_mode()
def specksim_features(model,sim,trial,ends,step_us,device):
    from ..models.gelneuro_unified_snn import trace_readout
    sim.reset_states()
    begin=int(ends[0])-50_000
    stop=int(ends[-1])
    a,b=np.searchsorted(trial.events[:,2],[begin,stop])
    events=trial.events[a:b]
    data=np.empty(len(events),dtype=sim.output_dtype)
    data["x"]=events[:,0]//4
    data["y"]=events[:,1]//4
    data["t"]=events[:,2]-begin
    data["p"]=events[:,3]
    pieces=[sim(data[i:i+20000]) for i in range(0,len(data),20000)]
    out=np.concatenate(pieces) if pieces else np.empty(0,dtype=sim.output_dtype)
    size=(stop-begin)//step_us
    keep=out["t"]<stop-begin
    index=(out["t"][keep].astype(np.int64)//step_us)*16+out["p"][keep]
    counts=np.bincount(index,minlength=size*16).reshape(size,16).astype(np.float32)
    trace=torch.zeros(1,4,16,device=device)
    x=torch.from_numpy(counts).unsqueeze(0).to(device)
    z,_=trace_readout(x*model.rate_scale,trace,model.decay)
    ratio=50_000//step_us
    output_counts=counts.reshape(-1,ratio,16).sum((1,2)).tolist()
    return z[0,ratio-1::ratio].cpu(),output_counts


@torch.inference_mode()
def specksim_replay(model,sim,trial,ends,step_us,device):
    z,count=specksim_features(model,sim,trial,ends,step_us,device)
    logits=model.head(z.to(device)).cpu()
    return logits[:,:4],logits[:,4:],count


def evaluate_replay(args,fold,trials,sequences,device):
    prefix=args.checkpoints/f"fold{fold}"
    saved=torch.load(prefix.with_suffix(".pt"),map_location=device)
    info=json.loads(prefix.with_suffix(".json").read_text())
    heldout=[t for t in trials if t.name in info["heldout_trials"]]
    model=GelNeuroUnifiedSNN(args.step_us).to(device).eval()
    model.load_state_dict(saved["model"],strict=False)
    # The checkpoint's decay is for its training dt. Replay must recompute decay.
    model.decay.copy_((-args.step_us/1000./torch.tensor([50.,150.,500.,1500.],device=device)).exp().reshape(1,4,1))
    if args.int8 and not args.specksim: model,_=model.quantized_copy()
    sim=make_specksim(model) if args.specksim else None
    confusion=np.zeros((4,4),int); directions=np.zeros((4,4),int); records=np.zeros((4,4),int)
    rows=[]; record_rows=[]; joint=slides=0; firing=[]
    started=time.perf_counter()
    for trial in heldout:
        _,target,direction,ends=sequences[trial.name]
        if args.specksim:
            p,d,count=specksim_replay(model,sim,trial,ends,args.step_us,device)
        else:
            p,d,count=replay(model,trial,ends,args.step_us,device)
        pred=p.argmax(-1).tolist(); dp=d.argmax(-1).tolist(); votes=[]
        assert len(pred)==len(ends)
        for y,dy,a,b,end,n in zip(target.tolist(),direction.tolist(),pred,dp,ends,count):
            firing.append(n)
            if y==IGNORE: continue
            confusion[y,a]+=1; votes.append(a)
            if dy!=IGNORE:
                directions[dy,b]+=1; slides+=1; joint+=int(a==1 and dy==b)
            rows.append(dict(recording=trial.name,end_us=int(end),task=CLASSES[y],
                             predicted_task=CLASSES[a],direction=DIRECTIONS[dy] if dy>=0 else None,
                             predicted_direction=DIRECTIONS[b],output_spikes=n))
        majority=Counter(votes).most_common(1)[0][0]
        records[task_of(trial),majority]+=1
        record_rows.append(dict(recording=trial.name,task=CLASSES[task_of(trial)],majority=CLASSES[majority]))
    result=dict(fold=fold,step_us=args.step_us,int8=args.int8,specksim=args.specksim,seconds=time.perf_counter()-started,
        task=classification(confusion,CLASSES),direction=classification(directions,DIRECTIONS),
        recordings=classification(records,CLASSES),slide_joint_recall=joint/slides,
        predictions=rows,recording_predictions=record_rows,output_counts_per_50ms=firing,
        firing=dict(mean_per_second=float(np.mean(firing))*20,p95_per_50ms=float(np.percentile(firing,95)),
                    max_per_50ms=max(firing)),train_trials=info["train_trials"],heldout_trials=info["heldout_trials"])
    (args.out/f"fold{fold}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("fold","step_us","seconds","firing")}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints",type=Path)
    p.add_argument("--data",type=Path,default=Path("data/raw/tactile"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--step-us",type=int,choices=(1000,10000,50000),default=10000)
    p.add_argument("--device",default="cuda")
    p.add_argument("--int8",action="store_true")
    p.add_argument("--specksim",action="store_true",help="Official integer event-by-event simulator; no device opened")
    p.add_argument("--fold",type=int,choices=range(3))
    args=p.parse_args()
    if args.specksim: args.int8=True
    trials,sequences=prepare(args.data,50000)
    device=configure_device(args.device,4)
    args.out.mkdir(parents=True,exist_ok=False)
    results=[evaluate_replay(args,f,trials,sequences,device) for f in (
        range(3) if args.fold is None else [args.fold])]
    summary=aggregate(results)
    summary.update(step_us=args.step_us,int8=args.int8,specksim=args.specksim,seconds=sum(r["seconds"] for r in results))
    all_counts=sum([r["output_counts_per_50ms"] for r in results],[])
    summary["firing"]=dict(mean_per_second=float(np.mean(all_counts))*20,
        p95_per_50ms=float(np.percentile(all_counts,95)),max_per_50ms=max(all_counts))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
