#!/usr/bin/env bash
set -euo pipefail

gripper_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$gripper_project_root"
export PYTHONPATH="$gripper_project_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$gripper_project_root/.venv/bin/python" -m neurofusion.gripper_control "$@"
