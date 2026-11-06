"""Render one recorded tactile trial side by side, without opening hardware."""
import argparse
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from neurofusion.capture.reader import iter_event_batches, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    root = args.trial.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Select a complete recording")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    start = manifest["host_recording_start_ns"]
    seconds = manifest["config"]["seconds"]
    labels = manifest["config"]["labels"]
    frames = list(rows(root / "gelsight/frames.jsonl"))
    frame_times = [(row["host_received_ns"] - start) / 1e9 for row in frames]
    batches = [(meta, values) for meta, values in iter_event_batches(root) if len(values)]
    receipt_times = np.array([(meta["host_received_ns"] - start) / 1e9 for meta, _ in batches])
    events = np.concatenate([values for _, values in batches]) if batches else np.empty((0, 4), dtype=np.int64)
    event_times = np.repeat(receipt_times, [len(values) for _, values in batches])
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fonts = {name: ImageFont.truetype(path, size) for name, path, size in (
        ("title", bold_path, 28), ("panel", bold_path, 22),
        ("normal", font_path, 18), ("small", font_path, 15))}
    width, height, fps = 1280, 760, 25
    colors = {"bg": "#0c1220", "panel": "#141e30", "text": "#e5edf8", "muted": "#9bacbe", "accent": "#5de0ce"}

    @lru_cache(maxsize=8)
    def picture(index):
        with Image.open(root / frames[index]["path"]) as source:
            source.draft("RGB", (590, 480))
            image = source.convert("RGB")
            image.thumbnail((590, 480), Image.Resampling.LANCZOS)
            return image

    def render(t):
        canvas = Image.new("RGB", (width, height), colors["bg"])
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 18), "NeuroFusion  |  Recorded tactile data", font=fonts["title"], fill=colors["text"])
        draw.text((30, 59), f"Operator label: slide {labels['direction']} / {labels['speed_level']}  |  Object: {labels['object_id']}  |  Repeat {labels['repetition']}", font=fonts["normal"], fill=colors["muted"])
        for x in (24, 652):
            draw.rounded_rectangle((x, 102, x + 604, 634), radius=12, fill=colors["panel"])
        draw.text((44, 117), "GelSight  |  tactile images", font=fonts["panel"], fill=colors["accent"])
        draw.text((672, 117), "GelNeuro  |  polarity events", font=fonts["panel"], fill=colors["accent"])
        index = bisect_right(frame_times, t) - 1
        if index >= 0:
            image = picture(index)
            canvas.paste(image, (24 + (604 - image.width) // 2, 158 + (474 - image.height) // 2))
        else:
            draw.text((80, 360), "Waiting for first recorded frame", font=fonts["normal"], fill=colors["muted"])
        lower, upper = np.searchsorted(event_times, [max(0, t - 0.1), t], side="right")
        selected = events[lower:upper]
        raster = np.full((128, 128, 3), [12, 18, 32], dtype=np.uint8)
        if len(selected):
            raster[selected[:, 1], selected[:, 0]] = np.where(selected[:, 3, None] > 0,
                np.array([255, 107, 87], dtype=np.uint8), np.array([53, 139, 255], dtype=np.uint8))
        canvas.paste(Image.fromarray(raster).resize((448, 448), Image.Resampling.NEAREST), (730, 164))
        draw.text((44, 648), f"Time {t:05.2f} / {seconds:.1f} s  |  Image {max(index + 1, 0)}/{len(frames)}", font=fonts["normal"], fill=colors["text"])
        draw.text((672, 648), f"Red: ON   Blue: OFF   |   {upper - lower:,} events / 100 ms", font=fonts["normal"], fill=colors["text"])
        draw.rounded_rectangle((30, 686, 1250, 694), radius=4, fill="#283850")
        if t > 0:
            draw.rounded_rectangle((30, 686, 30 + round(1220 * min(t / seconds, 1)), 694), radius=4, fill=colors["accent"])
        draw.text((30, 714), "Real recorded inputs | Display aligned by host receipt time; hardware synchronization not established.", font=fonts["small"], fill=colors["muted"])
        return canvas

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for number in range(round(seconds * fps)):
            encoder.stdin.write(render(number / fps).tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("Video rendering failed")
    candidates = np.arange(max(1.0, frame_times[0]), min(6.5, seconds), 1 / fps)
    counts = np.searchsorted(event_times, candidates, side="right") - np.searchsorted(event_times, candidates - 0.1, side="right")
    best = float(candidates[int(np.argmax(counts))])
    poster = args.output.with_suffix(".png")
    render(best).save(poster)
    print(json.dumps({"video": str(args.output.resolve()), "screenshot": str(poster.resolve()), "screenshot_time_s": best,
                      "source": str(root), "events": len(events), "frames": len(frames)}, indent=2), flush=True)
    if args.open:
        log = args.output.with_suffix(".player.log").open("w")
        player = subprocess.Popen(["ffplay", "-hide_banner", "-loglevel", "error", "-loop", "0", "-window_title",
            "NeuroFusion tactile playback | SPACE: pause | Q: close", "-x", "1280", "-y", "760", str(args.output.resolve())],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"PLAYER_PID={player.pid}", flush=True)


if __name__ == "__main__":
    main()
