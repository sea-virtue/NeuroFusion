"""Read current feedback, or explicitly run one bounded, monitored Mode 0 trial.

No automatic EEPROM mode changes. Existing incident interlocks remain active.
This entry point never sends learned closure actions or claims contact force.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .adapters.robotis import GripperConfig,RobotisGripper


def run(output,exercise=False,operator_ready=False,current_a=.05,seconds=.25,
        port='/dev/ttyUSB0',baud=2_000_000):
    if not math.isfinite(seconds) or not 0 < seconds <= 1.:
        raise ValueError('Use 0 < seconds <= 1 for this current commissioning entry point')
    if not math.isfinite(current_a) or abs(current_a) > .2:
        raise ValueError('Use signed amperes within +/-0.2 A')
    if exercise and not operator_ready:
        raise ValueError('Current exercise requires an empty, secured, cleared gripper and --operator-ready')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(passed=False,exercise=exercise,trained_policy_commands_sent=0,
                  current_unit='A',force_newtons_measured=False,
                  timing_basis='host receipt proxy; not intrinsic mechanical latency')
    device = RobotisGripper(GripperConfig(port=port,baud=baud,
        allow_motion=exercise,control_mode='current'))
    try:
        with device,(output/'trace.jsonl').open('x') as trace:
            report.update(model_number=device.model_number,firmware=device.firmware,
                          limits=device.limits,initial=asdict(device.read()))
            def record(sample):
                trace.write(json.dumps(asdict(sample))+'\n')
                trace.flush()
            if exercise:
                report['armed'] = asdict(device.arm())
                receipt,last = device.hold_current(current_a,seconds,record)
                report.update(receipt=asdict(receipt),last_active_sample=asdict(last) if last else None)
            report['final'] = asdict(device.read())
            report['passed'] = True
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['writes'] = device.writes
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--exercise',action='store_true')
    parser.add_argument('--operator-ready',action='store_true')
    parser.add_argument('--current-a',type=float,default=.05)
    parser.add_argument('--seconds',type=float,default=.25)
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--baud',type=int,default=2_000_000)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))


if __name__ == '__main__':
    main()
