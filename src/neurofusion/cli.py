import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import Config
from .demo import run_demo


def main(argv=None):
    parser = argparse.ArgumentParser(description='NeuroFusion research scaffold')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='SYNTHETIC dry run only: no hardware or learned model')
    demo.add_argument('--config', type=Path)
    demo.add_argument('--control-hz', type=int, help='Logical tick frequency, not measured wall-clock throughput')
    demo.add_argument('--without-events', action='store_true')
    demo.add_argument('--output', type=Path, help='Save synthetic trace JSON; existing files are not overwritten')
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config) if args.config else Config()
        if args.control_hz is not None:
            config = replace(config, control_hz=args.control_hz)
        report = run_demo(config, include_events=not args.without_events)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        print('SYNTHETIC DEMO | no hardware | no trained network | no real-time performance claim')
        for action in report['action_proposals']:
            print(f"logical {action['time_ns']/1e6:5.1f} ms | revision {action['memory_revision']:2} | action {action['values']}")
        print(f"Shadow proposals: {len(report['action_proposals'])}; robot commands sent: 0")
        return 0
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')
