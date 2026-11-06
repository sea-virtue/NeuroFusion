"""Real gripper -> policy state -> shadow action; optional bounded open/close test."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def run(output, seconds=3., exercise=False, delta=10, port='/dev/ttyUSB0', baud=2_000_000):
    import torch
    from .adapters.robotis import (RobotisGripper, GripperConfig,
                                  confirmed_execution_observation)
    from .models.policy import NeuroFusionPolicy,Observation
    from .models.split_actions import DeviceProposal
    from .training.pipeline import load_configuration
    if not 0 < seconds <= 60 or type(delta) is not int or not 5 <= delta <= 20:
        raise ValueError('Use 0 < seconds <= 60 and 5 <= delta <= 20')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    fusion,candidate = load_configuration('configs/pipeline.toml')
    model = NeuroFusionPolicy(fusion,candidate).eval()
    state = model.initial_state(1)
    report = {'passed':False,'trained_policy':False,'network_commands_sent':0,
              'franka_commands_sent':0,'measured_samples':0,'confirmed_movements':[],
              'feedback_layout':'q[15]=absolute closure fraction; other q fields preserved',
              'clock_basis':'host_receipt_proxy; no physical latency claim',
              'missing_inputs':['scene_RGB','Franka_feedback','GelSight','GelNeuro']}
    device = RobotisGripper(GripperConfig(port=port,baud=baud,allow_motion=exercise))
    try:
        with device, torch.inference_mode(), (output/'trace.jsonl').open('x') as trace:
            report.update(model_number=device.model_number,model_name=device.table.name,
                          firmware=device.firmware,device_id=device.config.device_id,
                          baud=baud,port=port,limits=device.limits)

            def record(sample):
                nonlocal state
                state = model.step(state,sample.observation(state))
                state = model.step(state,Observation('control',time.monotonic_ns()))
                proposals = model.propose(state)
                report['measured_samples'] += 1
                trace.write(json.dumps({'sample':asdict(sample),
                    'revision':state.revision,'memory_norm':state.memory.norm().item(),
                    'shadow_proposals':[asdict(p) for p in proposals]})+'\n')
                trace.flush()

            initial = device.read()
            report['initial'] = asdict(initial)
            end = time.monotonic()+seconds
            while time.monotonic() < end:
                record(device.read())
                time.sleep(.05)
            if exercise:
                if initial.position_raw-delta < device.limits['min_raw']:
                    raise RuntimeError('Not enough room for the requested opening step')
                print('Enabling bounded motion: open by',delta,'raw counts, then return',flush=True)
                armed = device.arm()
                report['armed'] = asdict(armed)
                origin = armed.position_raw
                for label,target in [('open_small',origin-delta),('close_back',origin)]:
                    # Explicit manual physical target; never send the untrained model output.
                    proposal = DeviceProposal(time.monotonic_ns(),state.revision,(0.,)*7,
                                              (device.closure(target),),False)
                    receipt = device.command(proposal)
                    reached = device.wait(receipt,record)
                    state = model.step(state,confirmed_execution_observation(receipt,reached,state))
                    report['confirmed_movements'].append({'label':label,'receipt':asdict(receipt),'measured':asdict(reached)})
                    trace.write(json.dumps({'confirmed_execution':asdict(receipt),
                        'measured':asdict(reached),'executed_slot_7':state.executed[0,7].item()})+'\n')
                    print(label,'target=',target,'measured=',reached.position_raw,flush=True)
                device.disarm()
            report['final'] = asdict(device.read())
            report['final_q15'] = state.robot[0,15].item()
            report['passed'] = report['measured_samples'] > 0 and (not exercise or len(report['confirmed_movements']) == 2)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['register_writes'] = device.writes
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--exercise',action='store_true',help='Enable small bounded physical motion')
    parser.add_argument('--delta',type=int,default=10)
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--baud',type=int,default=2_000_000)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
