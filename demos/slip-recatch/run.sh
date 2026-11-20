#!/usr/bin/env bash
set -euo pipefail
recatch_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$recatch_root"
export PYTHONPATH="$recatch_root/src:$recatch_root/.venv/chip-tools${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$recatch_root/.venv/chip-tools/matplotlib-cache"
exec "$recatch_root/.venv/live/bin/python" -m neurofusion.demos.slip_recatch "$@"
