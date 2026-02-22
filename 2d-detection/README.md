# 2D Detection Workspace

## Structure
- `benchmarks/ultralytics`: Ultralytics YOLO benchmark code, configs, and scripts
- `benchmarks/mmdetection`: custom MMDetection benchmark code, configs, and scripts
- `third_party/mmdetection`: vendored upstream MMDetection source
- `reports/benchmarks`: committed benchmark summaries and selected run artifacts
- `notebooks`: analysis notebooks

## Cluster Entrypoints
- `2d-detection/benchmarks/ultralytics/scripts/run_ultralytics_benchmark.sbatch`
- `2d-detection/benchmarks/mmdetection/scripts/run_mmdetection_benchmark.sbatch`
