#!/usr/bin/env bash
set -euo pipefail

balloon_demo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
balloon_root="$(cd -- "$balloon_demo/../.." && pwd)"
exec bash "$balloon_demo/run.sh" \
    --contact-head "$balloon_demo/models/contact-head.npz" \
    --pressure-checkpoint "$balloon_root/outputs/gelneuro-pressure-20261120/final/all-data.pt" \
    --config "$balloon_demo/configs/fast-close-20.json" \
    --seconds 35 \
    "$@"
