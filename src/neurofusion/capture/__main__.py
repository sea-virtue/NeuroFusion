"""python -m neurofusion.capture {gui,record,inspect,browse}."""
import argparse
import json
import time

from .schema import CaptureConfig, DIRECTIONS, TASKS, TrialLabels


def main():
    parser = argparse.ArgumentParser(description="Raw tactile recording without model inference or robot commands")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui", help="WSLg buttons, trial conditions and live previews")
    gui.add_argument("--root", default="data/raw/tactile")
    gui.add_argument("--camera", type=int, default=0)
    gui.add_argument("--events-only", action="store_true")
    record = commands.add_parser("record", help="Record one operator-labelled trial")
    record.add_argument("--root", default="data/raw/tactile")
    record.add_argument("--output", help="Exclusive new recording directory; never overwrites")
    record.add_argument("--seconds", type=float, default=8)
    record.add_argument("--lead-in", type=float, default=2)
    record.add_argument("--tail-out", type=float, default=2)
    record.add_argument("--events-only", action="store_true")
    record.add_argument("--camera", type=int, default=0)
    record.add_argument("--poll-ms", type=float, default=5)
    record.add_argument("--frame-queue", type=int, default=8)
    record.add_argument("--task", choices=TASKS, required=True)
    record.add_argument("--object-id", default="unrecorded")
    record.add_argument("--direction", choices=DIRECTIONS, default="none")
    record.add_argument("--speed-level", default="unmeasured")
    record.add_argument("--contact-level", default="unmeasured")
    record.add_argument("--repetition", type=int, default=1)
    record.add_argument("--material", default="unrecorded")
    record.add_argument("--direction-convention", default="unrecorded; define physical directions before training")
    record.add_argument("--mounting-notes", default="unrecorded; two different contact patches")
    record.add_argument("--notes", default="")
    inspect = commands.add_parser("inspect", help="Decode saved frames and verify event/index integrity")
    inspect.add_argument("path")
    inspect.add_argument("--save", action="store_true", help="Write quality.json")
    browse = commands.add_parser("browse", help="Browse and visually inspect recorded tactile trials")
    browse.add_argument("--root", default="data/raw/tactile")
    args = parser.parse_args()
    if args.command == "browse":
        from .browser import run_browser
        run_browser(args.root)
        return 0
    if args.command == "gui":
        from .gui import run_gui
        run_gui(args.root, args.camera, events_only=args.events_only)
        return 0
    if args.command == "inspect":
        from .reader import inspect_recording
        result = inspect_recording(args.path, save=args.save)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 2
    labels = TrialLabels(**{key: getattr(args, key) for key in TrialLabels.__dataclass_fields__})
    config = CaptureConfig(root=args.root, output=args.output, seconds=args.seconds,
                           lead_in=args.lead_in, tail_out=args.tail_out, gelsight=not args.events_only,
                           camera=args.camera, poll_ms=args.poll_ms, frame_queue=args.frame_queue, labels=labels)
    config.validate()
    from .recorder import CaptureSession
    from .reader import inspect_recording
    session = CaptureSession(config).start()
    print(f"OUTPUT: {session.root}\nOpening sensors; wait for HOLD STILL before starting. Close other sensor viewers first.", flush=True)
    while not session.finished:
        try:
            for update in session.poll():
                if update["kind"] == "phase":
                    print(f"\a{update['text']}", flush=True)
                elif update["kind"] in ("progress", "error", "ready"):
                    print(json.dumps(update, ensure_ascii=False), flush=True)
            session.latest_previews()  # Drain/discard display copies only.
            time.sleep(0.05)
        except KeyboardInterrupt:
            print("Stopping; preserving partial trial and releasing sensors...", flush=True)
            session.stop()
    result = inspect_recording(session.root, save=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
