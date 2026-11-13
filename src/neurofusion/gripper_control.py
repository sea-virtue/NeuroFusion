"""Manual, bounded RH-P12 position commands without model dependencies.

close/open move relative to measured position, in encoder counts, not mm.
Manual motion uses the hardware position range and operator position ceiling.
Torque is disabled at the end, including after the optional hold interval.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from .adapters.robotis import GripperConfig, RobotisGripper


def _target(action, origin, steps, position, limits):
    target = position if action == 'move' else origin + (steps if action == 'close' else -steps)
    if not limits['min_raw'] <= target <= limits['max_raw']:
        raise ValueError(f'Target {target} outside device position limits')
    return target


def run(action='status', *, steps=12, position=None, hold_seconds=0.,
        port='/dev/ttyUSB0', baud=2_000_000, device_id=1, output=None, acceleration='max',speed='max'):
    if action not in ('status','close','open','move'):
        raise ValueError('Action must be status, close, open, or move')
    if acceleration not in ('max','low'):
        raise ValueError('acceleration must be max or low')
    if speed not in ('max','low'):
        raise ValueError('speed must be max or low')
    if type(steps) is not int or steps < 1:
        raise ValueError('Use a positive integer --steps; steps are encoder counts, not mm')
    if action == 'move' and type(position) is not int:
        raise ValueError('move requires an integer --position')
    if action != 'move' and position is not None:
        raise ValueError('--position is only valid with move')
    if not math.isfinite(hold_seconds) or not 0 <= hold_seconds <= 30:
        raise ValueError('Use 0 <= --hold-seconds <= 30')
    if action == 'status' and hold_seconds:
        raise ValueError('status does not enable torque or hold position')
    device = RobotisGripper(GripperConfig(
        port=port,baud=baud,device_id=device_id,allow_motion=action != 'status',
        use_max_acceleration=acceleration == 'max',use_max_velocity=speed == 'max'))
    if output is None:
        root = Path('outputs/gripper-control')
        root.mkdir(parents=True,exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=action+'-',dir=root))
    else:
        output = Path(output)
        output.mkdir(parents=True,exist_ok=False)
    report = dict(passed=False,action=action,steps=steps if action in ('close','open') else None,
                  hold_seconds=hold_seconds,port=port,baud=baud,device_id=device_id,
                  acceleration=acceleration,speed=speed,
                  trained_policy_commands_sent=0,franka_commands_sent=0)
    try:
        with device,(output/'trace.jsonl').open('x') as trace:
            if action != 'status':
                # Full-range manual moves are explicitly authorized. Derive
                # both travel bounds from the connected hardware; other tools
                # retain the driver's smaller commissioning defaults.
                span = device.limits['max_raw']-device.limits['min_raw']
                device.config = replace(device.config,max_step_raw=span,session_radius_raw=span)
                report['manual_travel_limits_raw'] = dict(max_step_raw=span,session_radius_raw=span)
            def record(phase, sample):
                trace.write(json.dumps(dict(phase=phase,sample=asdict(sample)))+'\n')
                trace.flush()
            initial = device.read()
            record('initial',initial)
            report.update(initial=asdict(initial),model_number=device.model_number,
                          model_name=device.table.name,firmware=device.firmware,
                          limits=device.limits,current_cap_a=device.config.current_cap_a,
                          position_ceiling_raw=device.position_ceiling_raw,
                          command_ceiling_raw=device.command_ceiling_raw,
                          usb_low_latency_enabled=device.usb_low_latency_enabled)
            print(f'当前位置 {initial.position_raw}，使能 {int(initial.torque_enabled)}，'
                  f'实测电流 {initial.current_a:.4f} A',flush=True)
            if action != 'status':
                # Reject hardware/operator position violations before arming.
                device._check_position_target(_target(action,initial.position_raw,steps,position,device.limits))
                armed = device.arm()
                record('armed',armed)
                report['armed'] = asdict(armed)
                report['profile_acceleration_raw'] = device._read(device.table.profile_acceleration,4)
                report['profile_velocity_raw'] = device._read(device.table.profile_velocity,4)
                print(f'运动参数：速度 {report["profile_velocity_raw"]} raw，'
                      f'加速度 {report["profile_acceleration_raw"]} raw',flush=True)
                target = _target(action,armed.position_raw,steps,position,device.limits)
                report['target_raw'] = target
                print(f'准备发送目标位置 {target}（相对使能后实测位置 {target-armed.position_raw:+d} 个计数）',flush=True)
                proposal = SimpleNamespace(time_ns=time.monotonic_ns(),
                    robotis=(device.closure(target),),normalized=False)
                receipt = device.command(proposal)
                print(f'已下发目标位置 {receipt.target_raw}',flush=True)
                reached = device.wait(receipt,lambda sample:record('moving',sample))
                report.update(receipt=asdict(receipt),reached=asdict(reached))
                print(f'已到位：目标 {target}，实测 {reached.position_raw}',flush=True)
                deadline = time.monotonic()+hold_seconds
                if hold_seconds:
                    print(f'保持使能 {hold_seconds:g} 秒',flush=True)
                while time.monotonic() < deadline:
                    sample = device.read()
                    record('holding',sample)
                    device._healthy(sample)
                    device._check_motion_sample(sample)
                    if abs(sample.position_raw-target) > device.config.tolerance_raw:
                        raise RuntimeError('Position moved outside target tolerance during hold')
                    time.sleep(min(.05,max(0.,deadline-time.monotonic())))
                device.disarm()
            final = device.read()
            record('final',final)
            report['final'] = asdict(final)
            if action != 'status' and final.torque_enabled:
                raise RuntimeError('Could not confirm torque off')
            report['passed'] = True
            print(f'结束：实测位置 {final.position_raw}，使能 {int(final.torque_enabled)}，'
                  f'实测电流 {final.current_a:.4f} A',flush=True)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['writes'] = device.writes
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'记录：{output / "report.json"}',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('status','close','open','move'),nargs='?',default='status')
    parser.add_argument('--steps',type=int,default=12,help='close/open positive relative counts; target must stay within position limits; default 12')
    parser.add_argument('--position',type=int,help='move to an absolute encoder position within hardware/operator limits')
    parser.add_argument('--hold-seconds',type=float,default=0.,help='Keep torque on after arrival for 0..30 seconds, then disable')
    parser.add_argument('--acceleration',choices=('max','low'),default='max',
                        help='max: use device Acceleration Limit for the RAM profile; low: use raw 1')
    parser.add_argument('--speed',choices=('max','low'),default='max',
                        help='max: use device Velocity Limit for the RAM profile; low: use 0.57 rpm')
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--baud',type=int,default=2_000_000)
    parser.add_argument('--device-id',type=int,default=1)
    parser.add_argument('--output',help='New directory for report and feedback; generated automatically by default')
    args = parser.parse_args()
    try:
        run(**vars(args))
    except (RuntimeError,ValueError,TimeoutError,OSError) as error:
        parser.exit(1,f'夹爪命令失败：{error}\n')


if __name__ == '__main__':
    main()
