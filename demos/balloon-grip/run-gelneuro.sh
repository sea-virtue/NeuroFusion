#!/usr/bin/env bash
set -euo pipefail

balloon_demo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
balloon_root="$(cd -- "$balloon_demo/../.." && pwd)"
output="$balloon_demo/outputs/gelneuro-cpu-live-$(date +%Y%m%d-%H%M%S)-$$"
export PYTHONPATH="$balloon_root/src${PYTHONPATH:+:$PYTHONPATH}"

printf 'GelNeuro CPU 日志目录 %s\n' "$output" >&2
exec "$balloon_root/.venv/live/bin/python" -m neurofusion.demos.slip_recatch.live \
  --backend cpu \
  --checkpoint "$balloon_root/artifacts/checkpoints/gelneuro-snn-20261111/final/all-data.pt" \
  --expected-direction up --seconds 10 --window-ms 10 --clock-hz 100 --poll-ms 2 \
  --output "$output" "$@"
