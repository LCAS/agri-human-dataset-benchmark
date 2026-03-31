#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$BENCH_ROOT/configs/tracking/default.yaml}"

python "$BENCH_ROOT/src/run_tracker.py" --config "$CONFIG_FILE"
