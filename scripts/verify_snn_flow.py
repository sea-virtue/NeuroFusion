"""Offline SNN flow verification. Does not open devices or send commands."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from neurofusion.capture.reader import iter_frames, iter_event_batches
from neurofusion.models.encoders import prepare_sparsh
from neurofusion.models.policy import NeuroFusionPolicy, Observation
from neurofusion.training.pipeline import load_configuration, run


def recorded_check(folder, fusion, candidate):
    marks=[json.loads(line) for line in (folder/'annotations.jsonl').read_text().splitlines()]
    # Operator timing is used only to choose a diagnostic active interval, not as motion truth.
    active=next((m['host_marked_ns']+550_000_000 for m in marks if m['name'].startswith('DO SLIDE')),0)
    frames=[]
    for meta,image in iter_frames(folder):
        frames.append((meta,image))
        if len(frames)>6: frames.pop(0)
        if len(frames)==6 and meta['host_received_ns']>=active: break
    if len(frames)<6: raise ValueError('Recording needs at least six camera frames')
    frame_time=frames[-1][0]['host_received_ns']
    packets=[]
    start=end=received=None
    for meta,events in iter_event_batches(folder):
        if meta['host_received_ns'] < frame_time or not len(events): continue
        if start is None:
            start=int(events[0,2]); end=start+64*candidate.event_bin_us
        packets.append(np.array(events,copy=True))
        received=meta['host_received_ns']
        if int(events[-1,2])>=end: break
    if start is None or int(packets[-1][-1,2])<end:
        raise ValueError('Not enough event time coverage after the sixth frame')
    raw=np.concatenate(packets)
    raw=raw[(raw[:,2]>=start)&(raw[:,2]<end)]
    reference=Path('outputs/sparsh-contact-probe-20261110/background.jpg')
    if not reference.is_file(): raise FileNotFoundError('Supply the recorded, audited no-contact reference: '+str(reference))
    background=np.array(Image.open(reference).convert('RGB'))
    tactile=prepare_sparsh(frames[-1][1],frames[0][1],background)[None]
    torch.manual_seed(7)
    model=NeuroFusionPolicy(fusion,candidate).eval()
    state=model.initial_state(1,frame_time)
    with torch.inference_mode():
        # No scene RGB or robot feedback is fabricated; their initial placeholders stay zero.
        state=model.step(state,Observation('frame',frame_time,tactile,captured_ns=frame_time,
                                          capture_time_basis='host_receipt_proxy'))
        with_frame=state.memory.clone()
        before_event=state
        state=model.step(state,Observation('event',received,[torch.from_numpy(raw)],captured_ns=received,
                            start_us=start,end_us=end,capture_time_basis='host_receipt_proxy'))
        event_trace_norm=state.event_state[1].norm().item()
        event_changed=not torch.equal(with_frame,state.memory)
        state=model.step(state,Observation('control',received+1))
        proposals=model.propose(state)
        empty=model.step(before_event,Observation('event',received,[torch.empty(0,4,dtype=torch.int64)],captured_ns=received,
                        start_us=start,end_us=end,capture_time_basis='host_receipt_proxy'))
        empty=model.step(empty,Observation('control',received+1))
        event_effect=(state.memory-empty.memory).abs().max().item()
        action_effect=(model.actions.sample(model.condition(state))-model.actions.sample(model.condition(empty))).abs().max().item()
    assert event_changed and torch.isfinite(state.memory).all() and len(proposals)==1
    assert event_trace_norm>0 and event_effect>0 and action_effect>0
    assert len(proposals[0].franka)==7 and len(proposals[0].robotis)==1
    return dict(recording=str(folder),real_gelsight=True,real_events=True,events=len(raw),
        snn_steps=64,snn_trace_norm=event_trace_norm,memory_changed=event_changed,
        event_vs_empty_memory_difference=event_effect,event_vs_empty_action_difference=action_effect,
        proposals=[asdict(p) for p in proposals],robot_commands_sent=0,
        missing_inputs=['scene_RGB','robot_feedback','expert_actions'],
        clock_basis='recorded host receipt proxy; no precision sync or live latency claim',
        scope='Real dual-tactile offline forward; untrained fusion/action head, not task validation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path)
    parser.add_argument('--matrix',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    fusion,candidate=load_configuration('configs/pipeline-snn.toml')
    report={'event_encoder':'trained_snn','robot_commands_sent':0}
    if args.recording: report['recorded_forward']=recorded_check(args.recording,fusion,candidate)
    if args.matrix:
        results=[]
        for method in ('plain','gated','attention'):
            for head in ('mlp','act','diffusion_unet','diffusion_transformer'):
                result=run(replace(fusion,kind=method),replace(candidate,action_head=head),steps=1)
                results.append(dict(fusion=method,head=head,synthetic=True,loss=result['losses'][0],
                                    gradients=result['first_gradient_sums'],proposals=len(result['proposals'])))
                print(method,head,'passed',flush=True)
        report['synthetic_matrix']=results
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
