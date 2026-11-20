"""Maintain a light grip on a balloon as a hanging weight oscillates."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import time
from uuid import uuid4

import numpy as np

from ...adapters.live_devices import SpeckSource
from ...adapters.robotis import RobotisGripper
from ...gripper_control import command_position, position_config
from .controller import Config, ContactGuard, GripOffsetRegulator, Regulator
from .learning import CAPTURE_TASKS, collect, encode_trial
from .network import ContactHead, fit_trial, load_reference
from .replay import replay
from .tactile import ContactImage, GelSightReader, RecentEvents, camera_index, capture_reference

ROOT = Path(__file__).resolve().parents[4]
DEMO = ROOT / 'demos' / 'balloon-grip'
PRESSURE_TRAVEL_LIMIT_COUNTS = 50


def recent(now, stamp, age_ns):
    return stamp is not None and 0 <= now - stamp <= age_ns


def write_row(trace, row):
    trace.write(json.dumps(row, allow_nan=False) + '\n')
    trace.flush()


def verify_startup_hold(device, origin, trace, *, max_drift=3, current_cap=None, startup_target=None):
    """Do not start the regulator until position holds after torque enable."""
    sample = device.arm()
    goal = origin
    if startup_target is not None:
        write_row(trace, {'phase': 'armed', 'gripper': asdict(sample)})
        receipt = command_position(device, startup_target)
        goal = receipt.target_raw
        write_row(trace, {'phase': 'startup_position_command', 'receipt': asdict(receipt)})

        def check_arrival(feedback):
            write_row(trace, {'phase': 'startup_arrival', 'gripper': asdict(feedback)})
            if abs(feedback.position_raw - origin) > min(4, device.config.session_radius_raw):
                raise RuntimeError('Startup moved outside the session position range')
            if feedback.goal_position_raw != goal or not 0 < feedback.goal_current_a <= current_cap:
                raise RuntimeError('Startup position or Goal Current readback changed')

        sample = device.wait(receipt, check_arrival, poll_s=.003)
    deadline = time.monotonic() + .50
    while True:
        write_row(trace, {'phase': 'armed_hold', 'gripper': asdict(sample)})
        device._healthy(sample)
        device._check_motion_sample(sample)
        if current_cap is not None and not 0 < sample.goal_current_a <= current_cap:
            raise RuntimeError('Unexpected Goal Current during startup hold')
        if abs(sample.position_raw - goal) > max_drift or sample.goal_position_raw != goal:
            raise RuntimeError('Gripper drifted during startup hold; closing the motion session')
        if time.monotonic() >= deadline:
            return sample.goal_position_raw
        sample = device.read()


def gripper_config_for_run(args, config, radius):
    pressure_mode = args.pressure_checkpoint is not None
    base = position_config(port=args.port, allow_motion=args.execute)
    return replace(
        base,
        use_hardware_current_limit=pressure_mode or args.manual_current_limit,
        position_profile='direct' if pressure_mode else base.position_profile,
        current_cap_a=config.current_cap_a,
        max_step_raw=config.tracking_limit_counts,
        session_radius_raw=radius,
        tolerance_raw=1 if args.contact_head and not (args.manual_current_limit or pressure_mode) else 3,
        timeout_s=.35 if args.manual_current_limit and not pressure_mode else base.timeout_s,
    )


def live(args, config, output, report):
    index = camera_index(args.camera)
    report['camera_index'] = index
    image = predictor = None
    if not args.probe:
        reference = load_reference(args.reference_image)
        image = ContactImage(reference)
        report['reference_image'] = str(args.reference_image.resolve())
        if args.contact_head:
            predictor = ContactHead(args.contact_head, args.reference_image, reference)
            report['contact_head'] = str(args.contact_head.resolve())
            report['controller'] = 'sparsh_contact_offset'
        else:
            report['controller'] = 'image_deformation'

    pressure = None
    if args.pressure_checkpoint:
        from .pressure import PressureModel, pressure_target

        pressure = PressureModel(args.pressure_checkpoint, args.pressure_confidence,
                                  args.pressure_release_confidence)
        pressure.predict(np.empty((0, 4), dtype=np.int64), 0, pressure.bin_us, 0)
        pressure.reset()
        report.update(controller='gelneuro_pressure_with_sparsh_agreement',
                      pressure_checkpoint=str(args.pressure_checkpoint.resolve()),
                      pressure_bin_us=pressure.bin_us, pressure_confidence=args.pressure_confidence,
                      pressure_release_confidence=args.pressure_release_confidence,
                      pressure_travel_limit_counts=PRESSURE_TRAVEL_LIMIT_COUNTS,
                      pressure_predictions=0)

    radius = max(config.max_open_counts, config.max_close_counts)
    if predictor is not None:
        radius = max(min(config.max_open_counts, config.network_max_offset_counts),
                     min(config.max_close_counts, config.network_close_limit))
    if pressure:
        radius = args.pressure_step
    gripper_config = gripper_config_for_run(args, config, radius)
    if predictor is not None and args.execute and not gripper_config.use_hardware_current_limit and config.current_cap_a > .20:
        raise ValueError('Sparsh pilot requires an explicit current cap of at most 0.20 A')
    age_ns = round(config.max_sensor_age_s * 1e9)
    history = RecentEvents(round(config.motion_window_s * 1e9))
    with ExitStack() as stack:
        trace = stack.enter_context((output / 'trace.jsonl').open('x'))
        board = stack.enter_context(SpeckSource(clock_hz=100) if pressure else SpeckSource(clock_hz=30, clock_lag_us=0))
        if pressure:
            pressure_trace = stack.enter_context((output / 'pressure.jsonl').open('x', buffering=64 * 1024))
        gel = stack.enter_context(GelSightReader(index, image, predictor))
        device = stack.enter_context(RobotisGripper(gripper_config))
        current_cap = device.config.current_cap_a
        if not gripper_config.use_hardware_current_limit and current_cap != config.current_cap_a:
            raise RuntimeError('Gripper did not retain the explicitly configured current cap')
        report['gripper_startup'] = asdict(device.read())
        report['configured_gripper_current_cap_a'] = current_cap
        report['gripper_current_limit_source'] = 'device' if gripper_config.use_hardware_current_limit else 'explicit'
        report['gripper_session_radius_raw'] = radius
        report['gripper_position_profile'] = device.config.position_profile
        report['gripper_profile_velocity_limit_raw'] = device.limits['velocity_raw']
        report['gripper_profile_acceleration_limit_raw'] = device.limits['acceleration_raw']

        startup_deadline = time.monotonic() + 10
        run_deadline = None
        start_ns = last_frame_ns = None
        baseline = []
        frame_intervals_ms = []
        inference_ages_ms = []
        regulator = contact_guard = None
        goal = None
        pressure_goal = None
        next_status = 0.0
        latest_action = None
        while run_deadline is None or time.monotonic() < run_deadline:
            if gel.error is not None:
                raise RuntimeError('GelSight acquisition failed') from gel.error
            received_ns, batch = board.read()
            report['events'] += len(batch)
            event_count, positive_events = history.update(received_ns, batch)
            new_pressure = False
            if pressure:
                for reading in pressure.update(received_ns, batch, board.watermark_us):
                    new_pressure = True
                    report['pressure_predictions'] += 1
                    pressure_trace.write(json.dumps(asdict(reading), allow_nan=False) + '\n')
            frame = gel.latest
            now_ns = time.monotonic_ns()
            guard_reading = gel.latest_guard
            guard_ratio = None
            guard_burst = guard_lost = guard_first = False
            if contact_guard is not None and guard_reading is not None:
                optical_ns, optical_score, optical_ready_ns = guard_reading
                if recent(now_ns, optical_ns, age_ns):
                    guard_ratio, guard_burst, guard_lost = contact_guard.observe(optical_ns, optical_score)
                    if not pressure and frame is not None and optical_ns > frame[0]:
                        # Use the fresh optical sign while the network is working on this frame.
                        if guard_lost or (guard_burst and recent(now_ns, frame[0], age_ns)):
                            guard_first = True
                            frame = optical_ns, frame[1], optical_score, optical_ready_ns
                        else:
                            guard_ratio = frame[2] / guard_baseline
                            guard_burst = guard_lost = False
            fresh = frame is not None and recent(now_ns, frame[0], age_ns)
            clock_fresh = recent(now_ns, board.watermark_host_ns, age_ns)
            if not (fresh and clock_fresh):
                if start_ns is not None or time.monotonic() > startup_deadline:
                    missing = 'GelSight frames' if not fresh else 'GelNeuro hardware clock'
                    raise RuntimeError(f'Missing fresh {missing}')
                time.sleep(.005)
                continue
            report['last_gelneuro_clock_us'] = board.last_clock_us
            if start_ns is None:
                start_ns = now_ns
                if args.probe:
                    run_deadline = time.monotonic() + args.seconds
            stamp, score, guard_score, ready_ns = frame
            new_frame = last_frame_ns is None or stamp > last_frame_ns
            pressure_tick = pressure is not None and regulator is not None and new_pressure
            if not (new_frame or pressure_tick or (goal is not None and not pressure)):
                time.sleep(.002 if pressure else .005)
                continue
            sample = device.read()
            device._healthy(sample)
            now_ns = time.monotonic_ns()
            if not all(recent(now_ns, timestamp, age_ns) for timestamp in
                       (stamp, sample.host_ns, board.watermark_host_ns)):
                raise RuntimeError('Tactile data or gripper feedback expired during control tick')
            motion = len(history.sites) >= config.gelneuro_min_sites
            row = {'frame_ns': stamp, 'score': score, 'gelneuro_events': event_count,
                   'image_guard_score': guard_score, 'prediction_delay_ms': (ready_ns - stamp) / 1e6,
                   'gelneuro_sites': len(history.sites), 'gelneuro_motion_assist': motion and predictor is not None,
                   'gelneuro_positive_events': positive_events,
                   'gelneuro_clock_us': board.last_clock_us, 'gripper': asdict(sample)}
            if goal is not None:
                try:
                    if pressure and abs(sample.position_raw - origin) > PRESSURE_TRAVEL_LIMIT_COUNTS:
                        raise RuntimeError(f'夹爪位置 {sample.position_raw} 距起点 {origin} 已超过 '
                                           f'{PRESSURE_TRAVEL_LIMIT_COUNTS} 个编码器计数')
                    device._check_motion_sample(sample)
                    if sample.goal_position_raw != goal:
                        raise RuntimeError('Unexpected goal register change during balloon control')
                    if not 0 < sample.goal_current_a <= current_cap:
                        raise RuntimeError('Goal Current outside the configured cap during balloon control')
                    if not lower - 3 <= sample.position_raw <= upper + 3:
                        raise RuntimeError('Gripper moved outside the active session position range')
                    if abs(sample.position_raw - goal) > config.tracking_limit_counts + 3:
                        raise RuntimeError('Gripper drifted beyond the active goal tracking range')
                except Exception as error:
                    row['error'] = repr(error)
                    write_row(trace, row)
                    raise
            if not (new_frame or pressure_tick):
                time.sleep(.005)
                continue
            if new_frame:
                if last_frame_ns is not None:
                    frame_intervals_ms.append((stamp - last_frame_ns) / 1e6)
                last_frame_ns = stamp
                report['frames'] += 1
                inference_ages_ms.append(row['prediction_delay_ms'])

            if args.probe:
                write_row(trace, row)
                continue
            if regulator is None:
                baseline.append((score, guard_score))
                write_row(trace, row)
                if now_ns - start_ns < config.startup_s * 1e9:
                    continue
                scores, guards = np.asarray(baseline).T
                setpoint = float(np.median(scores))
                guard_baseline = float(np.median(guards))
                spread = float(np.quantile(scores, .9) - np.quantile(scores, .1))
                guard_spread = float(np.quantile(guards, .9) - np.quantile(guards, .1))
                report.update(setpoint=setpoint, startup_spread=spread, image_guard_baseline=guard_baseline)
                if len(baseline) < 5 or min(setpoint, guard_baseline) < config.min_startup_score:
                    raise RuntimeError('Insufficient initial GelSight contact; start with the balloon lightly gripped')
                if max(spread / setpoint, guard_spread / guard_baseline) > 2 * config.deadband_fraction:
                    raise RuntimeError('Initial grip changed during startup; keep the balloon still until READY')
                origin = sample.position_raw
                lower = max(device.limits['min_raw'], origin - (radius if pressure else config.max_open_counts))
                ceiling = device.command_ceiling_raw
                if ceiling is None:
                    ceiling = device.limits['max_raw']
                upper = min(ceiling, origin + (radius if pressure else config.max_close_counts))
                if not lower < origin < upper:
                    raise RuntimeError('Initial position has no room for both opening and closing')
                regulator = (GripOffsetRegulator(config, setpoint, origin, lower, upper) if predictor
                             else Regulator(config, setpoint, lower, upper))
                if not pressure:
                    lower, upper = regulator.lower, regulator.upper
                if predictor:
                    contact_guard = ContactGuard(config, guard_baseline)
                report.update(startup_position_raw=origin, lower_raw=lower, upper_raw=upper)
                if args.execute:
                    startup_target = min(origin + 2, upper) if args.manual_current_limit and not pressure else None
                    try:
                        goal = verify_startup_hold(
                            device, origin, trace, max_drift=gripper_config.tolerance_raw, current_cap=current_cap,
                            startup_target=startup_target,
                        )
                    finally:
                        report['startup_target_raw'] = startup_target
                    if startup_target is not None:
                        regulator = GripOffsetRegulator(config, setpoint, goal, lower, upper)
                        lower, upper = regulator.lower, regulator.upper
                        report.update(lower_raw=lower, upper_raw=upper, control_anchor_raw=goal)
                    report['startup_hold_passed'] = True
                if pressure:
                    pressure.reset()
                    pressure_goal = goal if args.execute else origin
                run_deadline = time.monotonic() + args.seconds
                mode = '执行' if args.execute else '影子模式'
                print(f'READY {mode}，起点 {origin}，形变目标 {setpoint:.3f}，范围 {lower}–{upper}', flush=True)
                continue

            ratio = score / regulator.setpoint
            frame_s = stamp / 1e9
            if pressure:
                reading = pressure.latest
                target = None
                reason = 'hold'
                if reading is not None:
                    row['pressure'] = asdict(reading)
                try:
                    if guard_lost:
                        reason = 'contact_lost'
                        if args.execute:
                            raise RuntimeError('GelSight contact lost; grip session stopped')
                    else:
                        reason, proposed = pressure_target(
                            reading, now_ns=now_ns, clock_us=board.last_clock_us, sight_ns=stamp,
                            ratio=ratio, deadband=config.network_deadband_fraction, lower=lower, upper=upper,
                        )
                        action_s = now_ns / 1e9
                        if proposed is not None:
                            if proposed != pressure_goal and action_s - regulator.last_action_s >= config.min_command_interval_s:
                                target = proposed
                            else:
                                reason = 'hold'
                        if target is not None:
                            if args.execute:
                                row['receipt'] = asdict(command_position(device, target, time_ns=now_ns))
                                report['commands_acked'] += 1
                                goal = target
                            pressure_goal = target
                            regulator.last_action_s = action_s
                            latest_action = f'{reason} → {target}'
                except BaseException as error:
                    row['error'] = repr(error)
                    raise
                finally:
                    row.update(ratio=ratio, image_guard_ratio=guard_ratio, reason=reason, target_raw=target)
                    report['decisions'][reason] = report['decisions'].get(reason, 0) + 1
                    write_row(trace, row)
                if now_ns / 1e9 >= next_status:
                    probabilities = reading.probabilities if reading else {}
                    print(f'形变 {ratio:.2f}×，位置 {sample.position_raw}，'
                          f'加压 {probabilities.get("press", 0):.2f}，卸压 {probabilities.get("release", 0):.2f}，'
                          f'{latest_action or reason}', flush=True)
                    latest_action = None
                    next_status = now_ns / 1e9 + 1
                continue
            if predictor:
                if guard_ratio is None:
                    guard_ratio, guard_burst, guard_lost = contact_guard.observe(stamp, guard_score)
                decision = regulator.decide(
                    frame_s, ratio, sample.position_raw, goal, motion=motion,
                    guard_ratio=guard_ratio, burst=guard_burst,
                )
                row.update(image_guard_ratio=guard_ratio, optical_burst=guard_burst,
                           optical_before_network=guard_first)
            else:
                decision = regulator.decide(frame_s, ratio, sample.position_raw, goal)
                guard_lost = False
            row.update(ratio=ratio, **asdict(decision))
            if guard_lost:
                row.update(reason='contact_lost', target_raw=None)
            reason = row['reason']
            report['decisions'][reason] = report['decisions'].get(reason, 0) + 1
            try:
                if args.execute and reason == 'contact_lost':
                    raise RuntimeError('GelSight contact lost; grip session stopped')
                if row['target_raw'] is not None:
                    regulator.last_action_s = frame_s
                    latest_action = f'{decision.reason} → {decision.target_raw}'
                    if args.execute:
                        row['receipt'] = asdict(command_position(device, decision.target_raw, time_ns=now_ns))
                        report['commands_acked'] += 1
                        goal = decision.target_raw
            except BaseException as error:
                row['error'] = repr(error)
                raise
            finally:
                write_row(trace, row)
            if frame_s >= next_status:
                print(f'形变 {ratio:.2f}×，位置 {sample.position_raw}，'
                      f'GelNeuro {event_count} 事件，{latest_action or reason}', flush=True)
                latest_action = None
                next_status = frame_s + 1

        if frame_intervals_ms:
            report['gelsight_frame_interval_p50_ms'] = float(np.median(frame_intervals_ms))
            report['gelsight_frame_interval_p95_ms'] = float(np.quantile(frame_intervals_ms, .95))
            report['prediction_delay_p95_ms'] = float(np.quantile(inference_ages_ms, .95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true', help='Validate configuration without opening devices')
    modes.add_argument('--probe', action='store_true', help='Read both tactile heads and gripper with no motion')
    modes.add_argument('--reference', action='store_true', help='Capture unloaded GelSight reference with no motion')
    modes.add_argument('--shadow', action='store_true', help='Run feedback without enabling motion (default)')
    modes.add_argument('--execute', action='store_true', help='Enable bounded live gripper feedback')
    modes.add_argument('--collect', choices=CAPTURE_TASKS, help='Record original GelSight frames and GelNeuro events')
    modes.add_argument('--encode', type=Path, metavar='TRIAL', help='Encode a saved trial with the pretrained Sparsh model')
    modes.add_argument('--fit-contact', type=Path, metavar='TRIAL', help='Fit the pilot Sparsh contact head')
    modes.add_argument('--replay', type=Path, metavar='TRIAL', help='Replay tactile data with no hardware')
    parser.add_argument('--contact-head', type=Path, help='Use this Sparsh head in shadow/execute/replay')
    parser.add_argument('--pressure-checkpoint', type=Path, help='Run the GelNeuro pressure head in live control')
    parser.add_argument('--pressure-confidence', type=float, default=.55, help='Press score threshold for opening')
    parser.add_argument('--pressure-release-confidence', type=float, default=.35,
                        help='Release score threshold for closing (default: 0.35)')
    parser.add_argument('--pressure-step', type=int, default=3, help='Maximum pressure-mode offset from the starting position')
    parser.add_argument('--manual-current-limit', action='store_true',
                        help='With --execute, use the device current limit as in gripper.sh')
    parser.add_argument('--background', type=Path, help='Unloaded raw trial used by --encode')
    parser.add_argument('--features-out', type=Path)
    parser.add_argument('--history-frames', type=int, default=1, help='Causal Sparsh frame stride')
    parser.add_argument('--device', default='cpu', help='Torch device for offline encoding')
    parser.add_argument('--extension-mm', type=float, help='Manually measured displacement for --collect hold')
    parser.add_argument('--repetition', type=int, default=1)
    parser.add_argument('--object-id', default='balloon-01')
    parser.add_argument('--notes', default='')
    parser.add_argument('--top-assist', action='store_true', help='Mark top support applied throughout capture')
    parser.add_argument('--camera', type=int, help='Video index; by default use GelSight by-ID')
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--seconds', type=float, help='Run duration after startup')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reference-image', type=Path, default=DEMO / 'outputs' / 'gelsight-unloaded.local.png')
    args = parser.parse_args()
    if args.manual_current_limit and not (args.execute and args.contact_head):
        parser.error('--manual-current-limit requires --execute and --contact-head')
    if args.pressure_checkpoint:
        if not args.contact_head or any((args.probe, args.reference, args.collect, args.encode, args.fit_contact, args.replay)):
            parser.error('--pressure-checkpoint requires --contact-head and live, shadow or dry-run mode')
        if not all(1 / 3 < confidence < 1 for confidence in
                   (args.pressure_confidence, args.pressure_release_confidence)):
            parser.error('Press and release confidence must be between 1/3 and 1')
        if not 1 <= args.pressure_step <= PRESSURE_TRAVEL_LIMIT_COUNTS:
            parser.error(f'Pressure travel from the starting position must be between 1 and {PRESSURE_TRAVEL_LIMIT_COUNTS} counts')
    if args.seconds is None:
        if args.collect:
            args.seconds = {'load-cycle': 16, 'pull-rebound': 12, 'oscillation': 12}.get(args.collect, 8)
        else:
            args.seconds = 45
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    if args.camera is not None and args.camera < 0:
        parser.error('--camera must be nonnegative')
    if args.collect:
        if args.extension_mm is not None and (
            args.collect != 'hold' or not math.isfinite(args.extension_mm) or args.extension_mm < 0
        ):
            parser.error('--extension-mm requires a nonnegative measured displacement with --collect hold')
        if args.repetition < 1 or args.seconds <= 4:
            parser.error('Collection requires a positive repetition and --seconds greater than 4')
        return collect(args, DEMO, camera_index(args.camera))
    if args.encode:
        if args.background is None or args.history_frames < 1:
            parser.error('--encode requires --background and a positive --history-frames')
        output = args.features_out or DEMO / 'outputs' / f'{args.encode.resolve().name}.sparsh.npz'
        try:
            encode_trial(args.encode, args.background, output, args.history_frames, args.device)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f'{error}\n')
        return 0
    local = DEMO / 'configs' / 'config.local.json'
    path = args.config or (local if local.is_file() else DEMO / 'configs' / 'config.example.json')
    try:
        config = Config(**json.loads(path.read_text()))
    except (ValueError, OSError, TypeError) as error:
        parser.error(str(error))
    if args.pressure_checkpoint and 2 * args.pressure_step > config.tracking_limit_counts:
        parser.error('Full pressure reversal must fit within the configured position tracking limit')
    pilot = args.contact_head or DEMO / 'models' / 'contact-head.npz'
    if args.fit_contact or args.replay:
        try:
            if args.fit_contact:
                fit_trial(args.fit_contact, args.reference_image, pilot)
            else:
                output = DEMO / 'outputs' / f'replay-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}'
                print(f'日志目录 {output}', flush=True)
                replay(args.replay, pilot, args.reference_image, config, output)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f'{error}\n')
        return 0
    if args.dry_run:
        settings = asdict(config)
        if args.pressure_checkpoint:
            settings['pressure_travel_limit_counts'] = PRESSURE_TRAVEL_LIMIT_COUNTS
            settings['pressure_step_counts'] = args.pressure_step
        print(json.dumps(settings, indent=2))
        return 0
    if args.reference:
        capture_reference(camera_index(args.camera), args.reference_image)
        print(f'已保存空载参考 {args.reference_image}', flush=True)
        return 0

    mode = 'probe' if args.probe else 'execute' if args.execute else 'shadow'
    output = DEMO / 'outputs' / f'{mode}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'mode': mode, 'config': asdict(config), 'requested_seconds': args.seconds,
              'frames': 0, 'events': 0, 'commands_acked': 0, 'decisions': {}}
    print(f'日志目录 {output}', flush=True)

    def stop_on_term(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_term)
    try:
        live(args, config, output, report)
        report['status'] = 'completed'
        return 0
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        return 130
    except Exception as error:
        report.update(status='failed', error=repr(error))
        parser.exit(1, f'{error}\n')
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
