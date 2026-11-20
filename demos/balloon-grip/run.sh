#!/usr/bin/env bash
set -euo pipefail
balloon_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$balloon_root/src:$balloon_root/.venv/balloon-encoder${PYTHONPATH:+:$PYTHONPATH}"
exec "$balloon_root/.venv/live/bin/python" -m neurofusion.demos.balloon_grip "$@"
