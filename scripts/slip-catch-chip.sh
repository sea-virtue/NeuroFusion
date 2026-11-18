#!/usr/bin/env bash
set -euo pipefail
chip_demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$chip_demo_root/.venv/chip-tools${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$chip_demo_root/.venv/chip-tools/matplotlib-cache"
exec bash "$chip_demo_root/scripts/slip-catch.sh" --backend chip "$@"
