"""Replay recorded tactile input through the live network regulator."""

from collections import Counter
from dataclasses import asdict
import json
import time

import numpy as np

from ...capture.reader import iter_event_batches, iter_frames
from .controller import ContactGuard, GripOffsetRegulator
from .learning import verified_recording
from .network import ContactHead, load_reference, recording_start
from .tactile import ContactImage, RecentEvents, downsample


def replay(trial, checkpoint, reference_path, config, output):
    verified_recording(trial)
    reference = load_reference(reference_path)
    predictor = ContactHead(checkpoint, reference_path, reference)
    guard = ContactImage(reference)
    start = recording_start(trial)
    events = iter(iter_event_batches(trial))
    pending = next(events, None)
    history = RecentEvents(round(config.motion_window_s * 1e9))
    startup, latencies = [], []
    origin = position = 100  # Offset units only; the recorder has no gripper telemetry.
    regulator = contact_guard = None
    counts = Counter()
    motion_frames = 0
    stop = None

    output.mkdir(parents=True, exist_ok=False)
    with (output / 'trace.jsonl').open('x', encoding='utf-8') as trace:
        for metadata, image in iter_frames(trial):
            stamp = metadata['host_received_ns']
            relative = (stamp - start) / 1e9
            while pending is not None and pending[0]['host_received_ns'] <= stamp:
                batch, values = pending
                history.update(batch['host_received_ns'], values)
                pending = next(events, None)
            event_count, positive = history.update(stamp, np.empty((0, 4), dtype=np.int64))
            motion = len(history.sites) >= config.gelneuro_min_sites
            rgb = downsample(image)
            before = time.perf_counter_ns()
            score = predictor.update(rgb)
            if score is None or relative < .9:
                continue
            latencies.append((time.perf_counter_ns() - before) / 1e6)
            measured_contact = guard.score(rgb)
            if regulator is None:
                if relative < config.startup_s:
                    startup.append((score, measured_contact))
                    continue
                if len(startup) < 5:
                    raise RuntimeError('Insufficient recorded initial contact samples')
                setpoint, guard_baseline = np.median(startup, axis=0)
                if min(setpoint, guard_baseline) < config.min_startup_score:
                    raise RuntimeError('Initial replay interval has insufficient contact')
                regulator = GripOffsetRegulator(
                    config, float(setpoint), origin,
                    origin - config.max_open_counts, origin + config.max_close_counts,
                )
                contact_guard = ContactGuard(config, guard_baseline)
            ratio = score / regulator.setpoint
            guard_ratio, guard_burst, contact_missing = contact_guard.observe(stamp, measured_contact)
            decision = regulator.decide(
                relative, ratio, position, position, motion=motion,
                guard_ratio=guard_ratio, burst=guard_burst,
            )
            if contact_missing or decision.reason == 'contact_lost':
                stop = 'contact_lost'
            if decision.target_raw is not None and stop is None:
                regulator.last_action_s = relative
                position = decision.target_raw
            reason = stop or decision.reason
            row = {
                'relative_s': relative,
                'network_score': score, 'image_guard_score': measured_contact, 'image_guard_ratio': guard_ratio,
                'ratio': ratio, 'projected_ratio': decision.projected_ratio,
                'synthetic_offset_raw': position - origin,
                'reason': reason,
                'gelneuro_events': event_count, 'gelneuro_positive_events': positive,
                'gelneuro_sites': len(history.sites), 'motion_assist': motion, 'optical_burst': guard_burst,
            }
            trace.write(json.dumps(row, allow_nan=False) + '\n')
            counts[reason] += 1
            motion_frames += int(motion)
            if stop:
                break
    if regulator is None or not counts:
        raise RuntimeError('Replay did not reach the decision phase')
    report = {
        'trial': str(trial.resolve()), 'model': str(checkpoint.resolve()),
        'feedback_basis': 'instant arrival assumed; recorded tactile data unchanged',
        'setpoint': regulator.setpoint, 'image_guard_baseline': float(guard_baseline),
        'decisions': dict(counts), 'gelneuro_motion_frames': motion_frames,
        'network_latency_p50_ms': float(np.median(latencies)),
        'network_latency_p95_ms': float(np.quantile(latencies, .95)),
        'commands_acked': 0, 'status': stop or 'completed',
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)
