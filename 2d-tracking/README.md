# 2D Tracking Workspace

## Structure
- `common/mot`: shared MOT conversion and evaluation utilities used across tracking benchmarks
- `benchmarks/boxmot`: BoxMOT-based tracking code, configs, scripts, and reports
- `benchmarks/norfair`: Norfair-based tracking code, configs, scripts, and reports
- `third_party/norfair`: vendored upstream Norfair source

## Benchmark Layout
- `2d-tracking/common/mot`: shared tracker-agnostic MOT conversion and evaluation logic
- `2d-tracking/benchmarks/boxmot/src`: Python entrypoints for BoxMOT tracking, MOT conversion, and evaluation
- `2d-tracking/benchmarks/boxmot/configs`: YAML configs for tracking and evaluation runs
- `2d-tracking/benchmarks/boxmot/scripts`: local and cluster launchers
- `2d-tracking/benchmarks/norfair/src`: Python entrypoints for tracking, MOT conversion, and evaluation
- `2d-tracking/benchmarks/norfair/configs`: YAML configs for tracking and evaluation runs
- `2d-tracking/benchmarks/norfair/scripts`: local and cluster launchers
- `2d-tracking/reports`: generated tracking runs and evaluation summaries

## Reports
- `2d-tracking/reports/runs`: per-run outputs for individual tracking models
- `2d-tracking/reports/summary`: shared evaluation summaries across tracking benchmarks
