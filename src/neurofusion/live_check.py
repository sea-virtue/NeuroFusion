"""Live GelSight + Speck -> pretrained encoders -> contact memory -> shadow proposals.

Diagnostic input timestamps are host receipt proxies, NOT synchronized sensor
exposure/device times. No robot, gripper motion, torque or flash commands.
"""
import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from .adapters.live_devices import SpeckSource,GelSightSource
from .models.encoders import prepare_sparsh
from .models.policy import Observation,NeuroFusionPolicy
from .training.pipeline import load_configuration


@torch.inference_mode()
def run(seconds, camera, output, gripper_port=None, gripper_baud=2_000_000):
    if seconds <= 0: raise ValueError('seconds must be positive')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    fusion,candidate=load_configuration('configs/pipeline.toml')
    model=NeuroFusionPolicy(fusion,candidate).eval()
    report={'real_sensor_inputs':True,'trained_contact_policy':False,'robot_commands_sent':0,
        'clock_basis':'host_receipt_proxy; not device/host clock calibration or physical latency measurement',
        'missing_inputs':['scene_RGB','Franka_feedback','gripper_feedback'],
        'frames':0,'event_packets':0,'events':0,'memory_updates':0,'proposals':0,
        'gripper_serial_ports':[str(p) for pattern in ('ttyUSB*','ttyACM*') for p in Path('/dev').glob(pattern)]}
    state=model.initial_state(1)
    history=deque(maxlen=6)
    start=time.monotonic()
    before=state.memory.clone()
    try:
        with ExitStack() as stack, (output/'trace.jsonl').open('x') as trace:
            board=stack.enter_context(SpeckSource())
            gel=stack.enter_context(GelSightSource(camera))
            gripper=None
            if gripper_port is not None:
                from .adapters.robotis import RobotisGripper,GripperConfig
                gripper=stack.enter_context(RobotisGripper(GripperConfig(port=gripper_port,baud=gripper_baud)))
                report['missing_inputs'].remove('gripper_feedback')
                report['gripper_samples']=0
                report['gripper_model']=gripper.table.name
                report['feedback_layout']='q[15]=measured closure fraction; other fields preserved'
            background=None
            while time.monotonic()-start < seconds:
                receipt,image=gel.read()
                report['frames']+=1
                report['camera_shape']=list(image.shape)
                if background is None:
                    # Startup reference only, not a verified unloaded calibration.
                    background=image.copy()
                    from PIL import Image
                    Image.fromarray(image).save(output/'gelsight-first.png')
                    report['background_basis']='first live frame; unloaded condition not verified'
                history.append(image)
                if len(history)==6:
                    tactile=prepare_sparsh(image,history[0],background)[None]
                    state=model.step(state,Observation('frame',time.monotonic_ns(),tactile,
                        captured_ns=receipt,capture_time_basis='host_receipt_proxy'))
                    report['memory_updates']+=1
                receipt,events=board.read()
                if len(events):
                    if (events[1:,2]<events[:-1,2]).any():
                        raise RuntimeError('Device time reversal; stop and resynchronize explicitly')
                    state=model.step(state,Observation('event',time.monotonic_ns(),[torch.from_numpy(events)],
                        captured_ns=receipt,start_us=int(events[:,2].min()),end_us=int(events[:,2].max())+1,
                        capture_time_basis='host_receipt_proxy'))
                    report['events']+=len(events)
                    report['event_packets']+=1
                    report['memory_updates']+=1
                gripper_sample=None
                if gripper is not None:
                    gripper_sample=gripper.read()
                    state=model.step(state,gripper_sample.observation(state))
                    report['gripper_samples']+=1
                if len(history)==6:
                    state=model.step(state,Observation('control',time.monotonic_ns()))
                    proposals=model.propose(state)
                    report['proposals']+=len(proposals)
                    trace.write(json.dumps({'host_ns':state.time_ns,'revision':state.revision,'events':len(events),
                        'memory_norm':state.memory.norm().item(),'gripper':asdict(gripper_sample) if gripper_sample else None,'proposals':[asdict(p) for p in proposals]})+'\n')
                if report['frames']%10==0:
                    print(f"frames={report['frames']} events={report['events']} proposals={report['proposals']}",flush=True)
            from PIL import Image
            Image.fromarray(image).save(output/'gelsight-last.png')
        report['memory_changed']=not torch.equal(before,state.memory)
        report['passed']=report['frames']>=6 and report['events']>0 and report['proposals']>0
    except Exception as error:
        report['passed']=False
        report['error']=repr(error)
        raise
    finally:
        report['elapsed_s']=time.monotonic()-start
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=15)
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gripper-port',help='Optional read-only U2D2 serial input; never sends motion')
    parser.add_argument('--gripper-baud',type=int,default=2_000_000)
    args=parser.parse_args()
    report=run(args.seconds,args.camera,args.output,args.gripper_port,args.gripper_baud)
    raise SystemExit(0 if report['passed'] else 2)


if __name__=='__main__': main()
