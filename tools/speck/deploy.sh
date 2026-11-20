#!/usr/bin/env bash
set -euo pipefail
speck_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$speck_root"
export PYTHONPATH="$speck_root/src:$speck_root/.venv/chip-tools${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$speck_root/.venv/chip-tools/matplotlib-cache"
if [[ "${1:-}" == export ]]; then
    shift
    exec "$speck_root/.venv/live/bin/python" -m neurofusion.training.export_gelneuro_unified_snn "$@"
fi
exec "$speck_root/.venv/live/bin/python" -m neurofusion.speck_deploy "$@"
