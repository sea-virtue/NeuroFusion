"""Launch the existing Speck publisher and gripper controller as one session."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile

from .slip import SlipConfig


def build_commands(args, output, config):
    publisher = [sys.executable, '-m', 'neurofusion.control.live',
                 '--checkpoint', str(args.checkpoint), '--expected-direction', config.expected_direction,
                 '--seconds', '0', '--window-ms', str(args.window_ms),
                 '--clock-hz', str(args.clock_hz), '--poll-ms', str(args.poll_ms),
                 '--output', str(output/'events')]
    controller = [sys.executable, '-m', 'neurofusion.control', 'stream',
                  '--config', str(output/'config.json'), '--control-hz', str(args.control_hz),
                  '--output', str(output/'control')]
    if not args.shadow:
        controller.append('--execute')
    return publisher,controller


def stop_child(process):
    """Give the Python contexts time to disable torque/release the camera."""
    if process is None or process.poll() is not None:
        return False
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5.)
        return False
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return True


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root/'configs/slip-catch-search.local.json')
    parser.add_argument('--checkpoint', type=Path,
                        default=root/'artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window-ms', type=int, default=1, help='Inference window; actual publication rate also depends on device timestamp delivery')
    parser.add_argument('--clock-hz', type=float, default=100.)
    parser.add_argument('--poll-ms', type=float, default=.25)
    parser.add_argument('--control-hz', type=float, default=100.)
    parser.add_argument('--shadow', action='store_true', help='Read feedback without enabling torque or sending targets')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without opening devices or creating output')
    args = parser.parse_args()
    for name in ('control_hz','clock_hz','poll_ms'):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.window_ms < 1:
        parser.error('window-ms must contain at least one native 1 ms bin')
    if args.clock_hz > 100:
        parser.error('This Speck SDK supports clock-hz at most 100')
    if not args.config.is_file() or not args.checkpoint.is_file():
        parser.error('Local control configuration and checkpoint must exist; see --config and --checkpoint')
    settings = json.loads(args.config.read_text())
    config = SlipConfig(**settings)
    planned = args.output or root/'outputs/slip-catch-<new-session>'
    if args.dry_run:
        publisher,controller = build_commands(args,planned,config)
        print(f'推理窗口 {args.window_ms} ms；分类发布频率按实际设备时间更新；夹爪控制 {args.control_hz:g} Hz')
        print(shlex.join(publisher)+' | '+shlex.join(controller))
        return 0
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True,exist_ok=False)
    else:
        parent = root/'outputs'
        parent.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix='slip-catch-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-',dir=parent))
    (output/'config.json').write_text(json.dumps(settings,indent=2)+'\n')
    boundary = root/'configs/gripper-position-limits.local.json'
    if boundary.exists():
        (output/'gripper-position-limits.json').write_bytes(boundary.read_bytes())
    publisher,controller = build_commands(args,output,config)
    report = dict(status='starting',config_source=str(args.config),shadow=args.shadow,
                  requested_inference_hz=1000/args.window_ms,requested_control_hz=args.control_hz,
                  publisher_command=publisher,controller_command=controller)
    def save():
        (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    previous = {s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
    source = consumer = None
    print(f'日志目录：{output}',flush=True)
    print(f'推理窗口 {args.window_ms} ms；夹爪控制 {args.control_hz:g} Hz；'
          f'{"shadow，只读" if args.shadow else "实机运行"}。Ctrl+C结束。',flush=True)
    save()
    try:
        with (output/'inference.stderr.log').open('x') as source_log:
            # Separate sessions keep a terminal Ctrl+C in this supervisor. It
            # forwards SIGINT once to each child, allowing normal SDK cleanup.
            source = subprocess.Popen(publisher,cwd=root,stdout=subprocess.PIPE,stderr=source_log,start_new_session=True)
            consumer = subprocess.Popen(controller,cwd=root,stdin=source.stdout,start_new_session=True)
            source.stdout.close()
            report.update(status='running',publisher_pid=source.pid,controller_pid=consumer.pid)
            save()
            source_notified = False
            while consumer.poll() is None and not stopped:
                if source.poll() is not None and not source_notified:
                    print(f'推理进程已结束；详细记录：{output/"inference.stderr.log"}',file=sys.stderr,flush=True)
                    source_notified = True
                    report['publisher_exited_before_controller'] = True
                try:
                    consumer.wait(timeout=.1)
                except subprocess.TimeoutExpired:
                    pass
            report['forced_controller_stop'] = stop_child(consumer)
            report['forced_publisher_stop'] = stop_child(source)
        report.update(status='stopped_by_user' if stopped else 'finished' if consumer.returncode == 0 else 'failed',
                      controller_exit_code=consumer.returncode,publisher_exit_code=source.returncode)
        summary = output/'events/report.json'
        if summary.exists():
            inference = json.loads(summary.read_text())
            report['effective_inference_hz'] = inference.get('effective_inference_hz')
            if report['effective_inference_hz'] is not None:
                print(f'本轮实际推理输出：{report["effective_inference_hz"]:.1f} Hz',flush=True)
        if report['forced_controller_stop'] or report['forced_publisher_stop']:
            print('进程未正常退出，已强制终止；请检查设备状态和日志。',file=sys.stderr)
            return 1
        return 130 if stopped else consumer.returncode
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        stop_child(consumer)
        stop_child(source)
        raise
    finally:
        save()
        for signum,handler in previous.items():
            signal.signal(signum,handler)


if __name__ == '__main__':
    raise SystemExit(main())
