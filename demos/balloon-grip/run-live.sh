#!/usr/bin/env bash
set -euo pipefail

balloon_demo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$balloon_demo/run.sh" \
    --execute \
    --contact-head "$balloon_demo/models/contact-head.npz" \
    --manual-current-limit \
    --config "$balloon_demo/configs/fast-close-20.json" \
    --seconds 70 \
    "$@"
