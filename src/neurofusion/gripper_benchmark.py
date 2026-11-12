"""Bounded, operator-observed RN commissioning; raw telemetry, no learned actions.

The production arm() interlock stays in force. --operator-ready authorizes only
this diagnostic after the operator clears and secures the empty gripper.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
from .adapters.robotis import (RobotisGripper,GripperConfig,motion_inhibit_reason,
                              decode_rn_state)


def decode_block(data):
    return decode_rn_state(data)


def validate_sample(sample, *, target, low, high, current_raw):
    if sample['goal_raw'] != target:
        raise RuntimeError(f"Goal changed unexpectedly: expected {target}, read {sample['goal_raw']}")
    if not low <= sample['position_raw'] <= high:
        raise RuntimeError(f"Position {sample['position_raw']} outside diagnostic range [{low},{high}]")
    if abs(sample['current_raw']) > current_raw+7:
        raise RuntimeError('Measured current exceeded diagnostic cap plus 28mA margin')
    if not 15 <= sample['voltage_v'] <= 30 or sample['temperature_c'] >= 60:
        raise RuntimeError('Invalid voltage or elevated temperature')


def trial_metrics(samples, start_raw, target_raw, sent_ns, ack_ns):
    if not samples or start_raw == target_raw:
        raise ValueError('A nonzero movement and samples are required')
    direction = 1 if target_raw > start_raw else -1
    distance = abs(target_raw-start_raw)
    threshold = min(3,distance)
    first = next((s for s in samples if direction*(s['position_raw']-start_raw) >= threshold),None)
    reached = next((s for s in samples if abs(s['position_raw']-target_raw) <= 2),None)
    tail_start = samples[-1]['query_end_ns']-150_000_000
    tail = [s for s in samples if s['query_end_ns'] >= tail_start]
    errors = [s['position_raw']-target_raw for s in tail]
    intervals = [(b['query_end_ns']-a['query_end_ns'])/1e6 for a,b in zip(samples,samples[1:])]
    mean_tail = statistics.mean(s['position_raw'] for s in tail)
    elapsed = (reached['query_end_ns']-sent_ns)/1e9 if reached else None
    return dict(start_raw=start_raw,target_raw=target_raw,
        command_ack_ms=(ack_ns-sent_ns)/1e6,
        first_observed_3_count_change_ms=(first['query_end_ns']-sent_ns)/1e6 if first else None,
        first_within_2_counts_ms=elapsed*1000 if elapsed else None,
        settled_after_ms=(samples[-1]['query_end_ns']-sent_ns)/1e6,
        steady_mean_raw=mean_tail,steady_mean_error_raw=statistics.mean(errors),
        steady_max_abs_error_raw=max(abs(e) for e in errors),
        overshoot_raw=max(0,max(direction*(s['position_raw']-target_raw) for s in samples)),
        mean_endpoint_speed_counts_s=abs(mean_tail-start_raw)/elapsed if elapsed else None,
        peak_abs_current_a=max(abs(s['current_a']) for s in samples),
        sample_interval_ms_median=statistics.median(intervals) if intervals else None,
        sample_interval_ms_max=max(intervals) if intervals else None,
        telemetry_peak_speed_rpm=max(abs(s['velocity_raw'])*.114 for s in samples),
        samples=len(samples),
        timing_basis='host command start to returned encoder feedback; 3-count threshold, not intrinsic motor or physical force latency')


class Bench:
    def __init__(self,device,trace,origin):
        self.device,self.trace,self.origin = device,trace,origin
        self.current_raw,self.velocity_raw = device.limits['current_raw'],4
        self.low = max(device.limits['min_raw'],origin-4)
        self.high = min(device.limits['max_raw'],origin+25)

    def emit(self,kind,**fields):
        self.trace.write(json.dumps(dict(kind=kind,host_ns=time.monotonic_ns(),**fields))+'\n')
        self.trace.flush()

    def sample(self,phase):
        d = self.device
        begin = time.monotonic_ns()
        data,result,error = d._packet.readTxRx(d._port,d.config.device_id,596,30)
        end = time.monotonic_ns()
        d._check(result,error)
        sample = dict(query_start_ns=begin,query_end_ns=end,**decode_block(data))
        self.emit('sample',phase=phase,**sample)
        return sample

    def profile(self,current_raw,velocity_raw,position=None):
        d,t = self.device,self.device.table
        if not 1 <= current_raw <= d.limits['current_raw'] or not 1 <= velocity_raw <= 8:
            raise ValueError('Diagnostic requires a positive ceiling within hardware current limit and at most .912rpm')
        if position is None:
            position = d._read(t.goal_position,4,True)
        if not self.low <= position <= self.high:
            raise ValueError('Profile target outside diagnostic range')
        d._write(t.profile_acceleration,4,1)
        d._write_position_profile(position,current_raw,velocity_raw)
        for address,size,value in ((t.goal_current,2,current_raw),
                (t.profile_velocity,4,velocity_raw),(t.profile_acceleration,4,1),
                (t.goal_position,4,position)):
            if d._read(address,size) != value:
                raise RuntimeError('Profile write/readback mismatch')
        self.current_raw,self.velocity_raw = current_raw,velocity_raw
        self.emit('profile',current_raw=current_raw,current_cap_a=current_raw*.00402,
                  velocity_raw=velocity_raw,velocity_cap_rpm=velocity_raw*.114,acceleration_raw=1)

    def monitor(self,target,seconds,phase,require_settle=False):
        samples = []
        deadline = time.monotonic()+seconds
        stable_since = None
        while time.monotonic() < deadline:
            sample = self.sample(phase)
            samples.append(sample)
            validate_sample(sample,target=target,low=self.low,high=self.high,current_raw=self.current_raw)
            if sample['profile_velocity_raw'] != self.velocity_raw or sample['goal_current_raw'] != self.current_raw:
                raise RuntimeError('Profile changed unexpectedly after enabling torque')
            if len(samples)%10 == 1:
                if self.device._read(self.device.table.torque,1) != 1 or self.device._read(self.device.table.hardware_error,1):
                    raise RuntimeError('Torque lost or hardware fault')
            if abs(sample['position_raw']-target) <= 2 and abs(sample['velocity_raw']) <= 1:
                stable_since = stable_since if stable_since is not None else time.monotonic()
                if require_settle and time.monotonic()-stable_since >= .20:
                    return samples
            else:
                stable_since = None
            time.sleep(.003)
        if require_settle:
            raise TimeoutError(f'No stable arrival at {target} within {seconds}s')
        return samples

    def move(self,target,label):
        d = self.device
        sample = self.sample(label+'-before')
        if not self.low <= target <= self.high or abs(target-sample['position_raw']) > 20:
            raise ValueError('Target outside diagnostic step/range bounds')
        start = time.monotonic_ns()
        d._write(d.table.goal_position,4,target)
        ack = time.monotonic_ns()
        self.emit('command',label=label,target_raw=target,sent_ns=start,ack_ns=ack)
        samples = self.monitor(target,4.,label,require_settle=True)
        metrics = trial_metrics(samples,sample['position_raw'],target,start,ack)
        metrics.update(label=label,current_cap_a=self.current_raw*.00402,
                       velocity_cap_rpm=self.velocity_raw*.114)
        print(json.dumps(metrics),flush=True)
        return metrics


def run(output,stage,operator_ready=False):
    if not operator_ready:
        raise ValueError('This diagnostic requires a cleared, secured, empty gripper and --operator-ready')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    report = dict(passed=False,stage=stage,operator_confirmed_empty_clear_secured=True,
                  prior_motion_hold=motion_inhibit_reason(),network_commands_sent=0,
                  force_newtons_measured=False,trials=[],cleanup={})
    d = RobotisGripper(GripperConfig(allow_motion=True))
    enabled_attempted = False
    saved = {}
    try:
        with d,(output/'trace.jsonl').open('x') as trace:
            if d.model_number != 35073:
                raise ValueError('This commissioning block is verified only for RN model 35073')
            initial = d.read()
            d._healthy(initial)
            if initial.torque_enabled or d.limits['mode'] != 5 or d.limits['return_level'] != 2:
                raise RuntimeError('Require initially disabled current-based position mode and write replies')
            if not 5 <= initial.position_raw <= 1120:
                raise RuntimeError('Too close to a hard stop for the initial commissioning range')
            report.update(initial=asdict(initial),limits=d.limits,firmware=d.firmware)
            bench = Bench(d,trace,initial.position_raw)
            t = d.table
            saved = {address:(size,d._read(address,size)) for address,size in
                     ((t.goal_current,2),(t.profile_velocity,4),(t.profile_acceleration,4))}
            try:
                # Mode 5 needs a nonzero current ceiling to actively hold position.
                # Preload measured position/velocity/ceiling before torque-on, as
                # in the supplied RN reference. Do not force a changed target or
                # diagnose zero-current drift as a failed position controller.
                report['initialization']='measured position with positive current ceiling and velocity; preload before torque-on'
                bench.profile(bench.current_raw,4,position=initial.position_raw)
                bench.sample('pre-enable')
                enabled_attempted = True
                bench.emit('torque_enable_request')
                d._write(t.torque,1,1)
                bench.monitor(initial.position_raw,.6,'enable-hold')
                report['hold_passed'] = True
                print('Enable/hold passed; goal and measured position stayed inside bounds',flush=True)
                if stage != 'hold':
                    profiles = ([(bench.current_raw,4,1)] if stage == 'small' else
                                [(bench.current_raw,4,3),(bench.current_raw,8,3)])
                    for current,velocity,repeats in profiles:
                        bench.profile(current,velocity)
                        for repeat in range(repeats):
                            for direction,target in (('close',initial.position_raw+12),('open',initial.position_raw)):
                                report['trials'].append(bench.move(target,f'i{current}-v{velocity}-r{repeat}-{direction}'))
                report['passed'] = True
            finally:
                # Always disable before changing targets or restoring RAM limits.
                if enabled_attempted:
                    try:
                        d._write(t.torque,1,0)
                        if d._read(t.torque,1) != 0:
                            raise RuntimeError('Torque-off readback failed')
                        report['cleanup']['torque_off_confirmed'] = True
                        time.sleep(.05)
                        current_position = d._read(t.present_position,4,True)
                        d._write(t.goal_position,4,current_position)
                        for address,(size,value) in saved.items():
                            d._write(address,size,value)
                        report['cleanup']['ram_profile_restored'] = True
                        report['final'] = asdict(d.read())
                    except BaseException as cleanup_error:
                        report['passed'] = False
                        report['cleanup']['error'] = repr(cleanup_error)
                        raise
    except BaseException as error:
        report['passed'] = False
        report['error'] = repr(error)
        raise
    finally:
        report['writes'] = d.writes
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('writes','trials')},indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=('hold','small','compare'),default='hold')
    parser.add_argument('--operator-ready',action='store_true')
    run(**vars(parser.parse_args()))


if __name__ == '__main__': main()
