#!/usr/bin/env bash
set -euo pipefail

python src/run_benchmark.py \
  --config configs/benchmark_ultralytics_zedrgb.yaml \
  --out ../../reports/benchmarks/summary/ultralytics/summary_ultralytics_zedrgb.csv
