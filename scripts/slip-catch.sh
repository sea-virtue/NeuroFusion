#!/usr/bin/env bash
set -euo pipefail

slip_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$slip_project_root"
export PYTHONPATH="$slip_project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$slip_project_root/.venv/live/bin/python" -m neurofusion.control.launch "$@"
