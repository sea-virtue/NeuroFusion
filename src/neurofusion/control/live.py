"""Publish real Speck classifications on stdout; never enables a motor.

An independent board measurement timestamp closes even empty DVS windows.
It uses the hardware clock, not a host-time extrapolation or a repeated label.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from .slip import SlideConfirmation


class EventWindows:
    """Contiguous native-bin windows, including empty windows between events.

    Fail on device-clock resets, wraparound or backlog rather than dropping
    data, rebasing clocks or labelling historical input as a fresh observation.
    """
    def __init__(self, bin_us, window_us=20_000, max_pending_us=10_000_000):
        if bin_us <= 0 or window_us <= 0 or window_us % bin_us or max_pending_us < window_us:
            raise ValueError('Window must contain whole native bins within the backlog bound')
        self.bin_us, self.window_us, self.max_pending_us = bin_us, window_us, max_pending_us
        self.pending = np.empty((0, 4), dtype=np.int64)
        self.receipts = np.empty(0, dtype=np.int64)
        self.start_us = self.last_us = self.last_host_ns = None
        self.watermark_us = self.watermark_host_ns = None

    def push(self, events, received_ns):
        events = np.asarray(events)
        if events.size == 0:
            events = events.reshape(0, 4)
        if (events.ndim != 2 or events.shape[1] != 4 or
                not np.isfinite(events).all() or not np.equal(events, np.floor(events)).all()):
            raise ValueError('Expected integer Nx4 events')
        if type(received_ns) is not int or received_ns < 0 or (
                self.last_host_ns is not None and received_ns < self.last_host_ns):
            raise ValueError('Host receipt clock reversed')
        self.last_host_ns = received_ns
        if not len(events):
            return
        x, y, t, p = events.T
        if ((x < 0) | (x >= 128) | (y < 0) | (y >= 128) | (t < 0) |
                ((p != 0) & (p != 1))).any() or (np.diff(t) < 0).any():
            raise ValueError('Events outside sensor bounds or device clock reversed')
        if self.last_us is not None and int(t[0]) < self.last_us:
            raise ValueError('Device clock reset/wrapped or out-of-order batch')
        if self.start_us is not None and int(t[0]) < self.start_us:
            raise ValueError('Late DVS event arrived after its window was closed')
        if self.start_us is None:
            self.start_us = int(t[0]) // self.bin_us * self.bin_us
        self.last_us = int(t[-1])
        # A sparse sensor can legitimately have seconds between events. This
        # span bounds catch-up computation, not USB queue age (a separate clock).
        if len(self.pending) + len(events) > 1_000_000:
            raise RuntimeError('Event count exceeded buffer capacity; no events were silently dropped')
        self.pending = np.concatenate((self.pending, events.astype(np.int64)))
        self.receipts = np.concatenate((self.receipts, np.full(len(events), received_ns, dtype=np.int64)))
        self.advance_watermark(self.last_us,received_ns)

    def advance_watermark(self, watermark_us, received_ns):
        if type(watermark_us) is not int or watermark_us < 0 or type(received_ns) is not int or received_ns < 0:
            raise ValueError('Integer device watermark and host receipt required')
        if self.last_host_ns is not None and received_ns < self.last_host_ns:
            raise ValueError('Host receipt clock reversed')
        if self.start_us is None:
            self.start_us = watermark_us//self.bin_us*self.bin_us
        if watermark_us-self.start_us > self.max_pending_us:
            raise RuntimeError('Device-time gap exceeds the bounded SNN catch-up budget; not evidence of USB failure')
        if self.watermark_us is None or watermark_us > self.watermark_us:
            self.watermark_us,self.watermark_host_ns = watermark_us,received_ns

    def ready(self):
        while self.start_us is not None and self.watermark_us >= self.start_us + self.window_us:
            end_us = self.start_us + self.window_us
            count = int(np.searchsorted(self.pending[:, 2], end_us))
            # This is when hardware evidence closed the window, not exposure.
            captured_ns = self.watermark_host_ns
            result = (self.pending[:count].copy(), self.start_us, end_us, captured_ns)
            self.pending, self.receipts = self.pending[count:], self.receipts[count:]
            self.start_us = end_us
            yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--expected-direction', required=True, choices=('up', 'down', 'left', 'right'))
    parser.add_argument('--seconds', type=float, default=10., help='0 runs until Ctrl+C')
    parser.add_argument('--window-ms', type=int, default=20)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--clock-hz', type=float, default=100.)
    parser.add_argument('--poll-ms', type=float, default=2.)
    parser.add_argument('--slide-confirm-frames', type=int, default=1)
    parser.add_argument('--slide-confidence', type=float, default=.8,
                        help='Threshold for P(slide), independent of the winning class')
    parser.add_argument('--max-prediction-age', type=float, default=.15)
    parser.add_argument('--output', required=True, type=Path, help='New diagnostic directory, exclusive creation')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 0 or args.threads < 1:
        parser.error('seconds must be finite and nonnegative; threads must be positive')
    if not np.isfinite(args.clock_hz) or not 0 < args.clock_hz <= 100 or not np.isfinite(args.poll_ms) or args.poll_ms <= 0:
        parser.error('clock-hz must be in (0,100] Hz; poll-ms must be finite and positive')
    import torch
    from ..adapters.live_devices import SpeckSource
    from .snn import SNNRecognizer
    torch.set_num_threads(args.threads)
    recognizer = SNNRecognizer(args.checkpoint, expected_direction=args.expected_direction)
    windows = EventWindows(recognizer.bin_us, args.window_ms * 1000)
    confirmation = SlideConfirmation(args.slide_confirm_frames,args.slide_confidence,args.max_prediction_age)
    # Warm kernels before opening hardware; actual recurrent state starts fresh.
    recognizer.predict([], 0, windows.window_us, time.monotonic_ns())
    recognizer.reset()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='starting', synthetic=False, hardware_commands_sent=0,
                  checkpoint=str(args.checkpoint), model_id=recognizer.model_id,
                  native_bin_us=recognizer.bin_us, window_us=windows.window_us,
                  predictions=0, windows_processed=0, events=0, startup_events_excluded=0,
                  intermediate_windows_not_published=0,
                  time_semantics='Shared FPGA power-measurement/DVS counter closes windows with 10 ms device lag; captured_ns is watermark host receipt, not exposure. Late events fail closed.',
                  clock_source='Speck board measurement timestamp',requested_clock_hz=args.clock_hz,clock_lag_us=10_000,
                  requested_inference_hz=1000/args.window_ms,poll_ms=args.poll_ms,
                  slide_confirm_frames=args.slide_confirm_frames,slide_confidence=args.slide_confidence,
                  slide_decision='P(slide) >= threshold; consecutive native windows; latched before coalescing',
                  recording_kind='Event/classification diagnostic only; not a complete dual-camera BC episode')
    started = time.monotonic()
    latencies = []
    prediction_times = []
    try:
        with (args.output / 'event-windows.jsonl').open('x') as raw, (
                args.output / 'predictions.jsonl').open('x') as predictions, SpeckSource(clock_hz=args.clock_hz) as source:
            report['actual_clock_hz'] = source.actual_clock_hz
            _, startup = source.read()
            report['startup_events_excluded'] = len(startup)
            started = last_prediction = time.monotonic()
            while args.seconds == 0 or time.monotonic() - started < args.seconds:
                received, events = source.read()
                report['events'] += len(events)
                windows.push(events, received)
                if source.watermark_us is not None:
                    # The snapshot was collected in this read; it is not a
                    # prediction of device time based on elapsed host time.
                    windows.advance_watermark(source.watermark_us,received)
                newest = None
                ready_count = 0
                for values, begin, end, captured in windows.ready():
                    before = time.monotonic_ns()
                    prediction = recognizer.predict(values, begin, end, captured)
                    prediction = confirmation.observe(prediction,begin,end,windows.watermark_us)
                    latencies.append((time.monotonic_ns() - before) / 1e6)
                    raw.write(json.dumps(dict(events=values.tolist(), start_us=begin, end_us=end,
                                              captured_host_ns=captured)) + '\n')
                    newest = prediction
                    ready_count += 1
                    report['windows_processed'] += 1
                # Sparse events can close many historical windows at once.
                # Advance all recurrent state, but publish only the newest one:
                # never present a catch-up burst as independent host arrivals.
                # Confirm native windows before coalescing; retain the original
                # evidence time so later quiet cannot erase or refresh it.
                if newest is not None:
                    report['intermediate_windows_not_published'] += ready_count - 1
                    if newest.available_ns - newest.captured_ns > 150_000_000:
                        raise RuntimeError('Newest prediction older than the controller freshness bound')
                    text = json.dumps(asdict(newest))
                    predictions.write(text + '\n')
                    print(text, flush=True)
                    report['predictions'] += 1
                    prediction_times.append(newest.available_ns)
                    last_prediction = time.monotonic()
                if time.monotonic() - last_prediction > 5.:
                    raise RuntimeError('No advancing Speck event watermark; no fresh predictions')
                time.sleep(args.poll_ms/1000.)
            report['pending_events_at_end'] = len(windows.pending)
            if not report['predictions']:
                raise RuntimeError('No complete real event window received')
        report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if latencies:
            report['inference_ms'] = dict(median=float(np.median(latencies)),
                                          p95=float(np.percentile(latencies, 95)), maximum=max(latencies))
        if len(prediction_times)>1:
            gaps=np.diff(np.asarray(prediction_times,dtype=np.int64))/1e6
            report['effective_inference_hz']=(len(prediction_times)-1)*1e9/(prediction_times[-1]-prediction_times[0])
            report['prediction_gap_ms']=dict(median=float(np.median(gaps)),p95=float(np.percentile(gaps,95)),maximum=float(gaps.max()))
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), file=sys.stderr, flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
