# YOLOv8 vs YOLOv11 2D Detection Benchmark

This project provides an end-to-end, reproducible benchmark to compare YOLOv8 and YOLOv11 on a public dataset using the Ultralytics runtime.

## What it does
- Trains and evaluates YOLOv8 and YOLOv11 with identical settings
- Aggregates mAP50, mAP50-95, precision, recall, and training time
- Writes results to `runs/benchmarks/summary.csv` and `runs/benchmarks/summary.json`

## Dataset
By default this uses `coco128`, a small public subset of COCO that Ultralytics downloads automatically.

If you want a different dataset, update `configs/benchmark.yaml`:
- `dataset`: path to your dataset YAML
- `imgsz`, `epochs`, `batch` as desired

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the benchmark
```bash
bash scripts/run_benchmark.sh
```

Or run directly:
```bash
python benchmarks/run_benchmark.py --config configs/benchmark.yaml
```

## Notes
- Model names in `configs/benchmark.yaml` map to Ultralytics model weights.
- If you have a GPU, set `device: 0` in the config; leave as `null` for CPU.
- `runs/benchmarks/` contains training logs and per-model artifacts.
