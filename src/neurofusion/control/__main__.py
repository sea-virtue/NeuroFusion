"""Replay a slip teacher or consume host-timestamped SNN JSONL on stdin."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import queue
import sys
import threading
import time
from .slip import SlipConfig, SlipController, SlipPrediction, GripperFeedback
from .session import Episode


def replay(config, rows, output, *, synthetic=False):
    controller = SlipController(config)
    episode = Episode(output,config,synthetic=synthetic,
                      metadata={'execution':'replay_shadow','hardware_commands_sent':0})
    try:
        for row in rows:
            episode.tick(controller,row['now_ns'],GripperFeedback(**row['feedback']),
                         SlipPrediction(**row['snn']) if row.get('snn') else None)
        status = 'shadow_finished' if controller.phase == 'FINISHED' else 'shadow_incomplete'
        episode.close(status)
    except BaseException as exc:
        episode.close('failed',repr(exc))
        raise
    return controller.phase


def demo_rows(config):
    """Deterministic test fixture, NOT a friction/actuator simulation or BC data."""
    closure = config.grip_closure
    for i in range(401):
        now = 1_000_000_000+i*10_000_000
        onset = 50 <= i < 110
        # Fixture eventually returns to the grip position after timed braking.
        if i < 50: closure = max(config.grip_closure-.002,closure-.00005)
        elif i < 105: closure = config.grip_closure-.002
        else: closure = min(config.grip_closure,closure+.0003)
        prediction = SlipPrediction(i,now,now,'slide' if onset else 'quiet',config.expected_direction,.99,.99)
        yield {'now_ns':now,'feedback':asdict(GripperFeedback(now,closure)),'snn':asdict(prediction)}


def stream(config, output, *, execute=False, port='/dev/ttyUSB0', tactile_recording=None):
    """Periodic watchdog independent of stdin arrival. Stdin never supplies feedback.

    Input predictions must originate on this host monotonic clock. Any stream
    failure requests bounded re-closing; gripper failure still propagates to the
    existing driver's context cleanup. No mode changes or incident-lock bypass.
    """
    from ..adapters.robotis import RobotisGripper, GripperConfig
    messages = queue.Queue(maxsize=64)
    overflow = threading.Event()

    def read_input():
        try:
            for line in sys.stdin:
                item = SlipPrediction(**json.loads(line))
                try: messages.put_nowait(item)
                except queue.Full:
                    overflow.set()
                    return
            messages.put(('error','SNN input closed'))
        except BaseException as exc:
            overflow.set()

    controller,latest = SlipController(config),None
    episode = Episode(output,config,synthetic=False,tactile_recording=tactile_recording,
                      metadata={'execution':'robotis' if execute else 'live_shadow',
                                'snn_input':'host_clock_stdin_jsonl','physical_stop_verified':False})
    try:
        with RobotisGripper(GripperConfig(port=port,allow_motion=execute)) as device:
            # Arm only after an initial fresh classification. Never enable before
            # there is a usable SNN heartbeat or automatically clear motion hold.
            threading.Thread(target=read_input,daemon=True).start()
            try: latest = messages.get(timeout=5.)
            except queue.Empty: raise RuntimeError('No initial SNN prediction')
            now = time.monotonic_ns()
            if (not isinstance(latest,SlipPrediction) or latest.available_ns > now or
                    not 0 <= now-latest.captured_ns <= config.max_prediction_age*1e9):
                raise RuntimeError('No fresh initial SNN prediction')
            if execute:
                sample = device.read()
                scale = device.limits['max_raw']-device.limits['min_raw']
                if config.max_tracking_delta*scale > device.config.max_step_raw:
                    raise ValueError('Tracking bound exceeds driver per-command bound')
                if abs(config.grip_closure*scale+device.limits['min_raw']-sample.position_raw)+config.max_open_delta*scale > device.config.session_radius_raw:
                    raise ValueError('Episode would exceed current driver commissioning range')
                device.arm()
            next_tick = time.monotonic()
            deadline = next_tick+config.release_timeout*2+config.slide_seconds+config.brake_timeout+5
            while time.monotonic() < deadline:
                while True:
                    try:
                        item = messages.get_nowait()
                        if isinstance(item,SlipPrediction):
                            if latest and item.sequence <= latest.sequence:
                                overflow.set()
                            latest = item
                        else: overflow.set()
                    except queue.Empty: break
                if overflow.is_set(): latest = None
                sample = device.read()
                device._healthy(sample)
                if execute and (not sample.torque_enabled or abs(sample.current_a) > device.config.current_cap_a+.03):
                    raise RuntimeError('Gripper torque/current fault')
                now = time.monotonic_ns()
                decision = episode.tick(controller,now,GripperFeedback(sample.host_ns,sample.closure),latest,
                                        device if execute else None,telemetry=asdict(sample))
                if decision['phase'] in controller.terminal: break
                next_tick += .02  # Target 50 Hz, measured tick gaps are checked.
                time.sleep(max(0.,next_tick-time.monotonic()))
            else: raise TimeoutError('Episode exceeded host watchdog')
            if execute and controller.phase == 'FINISHED':
                print('Sequence finished; retaining grip. Support ruler, then Ctrl+C to disable torque.',flush=True)
                try:
                    while True:
                        sample = device.read()
                        device._healthy(sample)
                        if not sample.torque_enabled or abs(sample.current_a) > device.config.current_cap_a+.03:
                            raise RuntimeError('Gripper fault during final hold')
                        episode.tick(controller,time.monotonic_ns(),GripperFeedback(sample.host_ns,sample.closure),
                                     None,telemetry=asdict(sample))
                        time.sleep(.02)
                except KeyboardInterrupt:
                    pass
            # Explicit release of torque is existing commissioning behavior;
            # support the ruler physically before ending a hardware session.
            episode.close('finished' if controller.phase == 'FINISHED' and execute else
                          'shadow_finished' if controller.phase == 'FINISHED' else 'failed',controller.reason)
    except BaseException as exc:
        if not episode.stream.closed: episode.close('failed',repr(exc))
        raise
    return controller.phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('demo','replay','stream'))
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--port',default='/dev/ttyUSB0')
    parser.add_argument('--tactile-recording',type=Path)
    args = parser.parse_args()
    config = SlipConfig(**json.loads(args.config.read_text()))
    if args.mode != 'stream' and args.execute: parser.error('--execute is only valid in stream mode')
    if args.mode == 'demo': phase = replay(config,demo_rows(config),args.output,synthetic=True)
    elif args.mode == 'replay':
        if args.input is None: parser.error('replay requires --input')
        with args.input.open() as handle:
            phase = replay(config,(json.loads(line) for line in handle),args.output)
    else:
        phase = stream(config,args.output,execute=args.execute,port=args.port,tactile_recording=args.tactile_recording)
    print(json.dumps({'phase':phase,'output':str(args.output),'hardware_execution_requested':args.execute}))
    return 0 if phase == 'FINISHED' else 1


if __name__ == '__main__': raise SystemExit(main())
