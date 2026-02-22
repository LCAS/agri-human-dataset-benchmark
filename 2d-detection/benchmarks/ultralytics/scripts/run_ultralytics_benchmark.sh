#!/usr/bin/env bash
set -euo pipefail

python src/run_benchmark.py \
  --config configs/benchmark_ultralytics_coco128.yaml \
  --out ../../reports/benchmarks/ultralytics/summary_coco128.csv
