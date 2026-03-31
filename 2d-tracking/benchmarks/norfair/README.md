# Norfair Tracking Benchmark

This benchmark wraps Norfair as a lightweight baseline for tracking frame-wise person detections.

## Structure

- `src/run_tracker.py`: main Norfair tracking entrypoint
- `src/convert_gt_to_mot.py`: wrapper around the shared GT-to-MOT utility
- `src/evaluate_mot.py`: wrapper around the shared MOT evaluation utility
- `configs/tracking`: tracking configs
- `configs/evaluation`: evaluation configs
- `scripts`: local and cluster launchers

## Environment

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Run with the default YAML:

```bash
python src/run_tracker.py --config configs/tracking/default.yaml
```

Or use explicit arguments:

```bash
python src/run_tracker.py \
  --detections-json /path/to/detections.json \
  --frames-dir /path/to/frames \
  --out-mot reports/runs/example/tracks.txt \
  --out-video reports/runs/example/tracking.mp4 \
  --distance-function iou \
  --distance-threshold 0.75
```

## Config Notes

Key fields include:

- `distance_function`
- `distance_threshold`
- `hit_counter_max`
- `initialization_delay`
- `detection_threshold`

If you only need a MOT output file, `frames_dir` can be omitted. If you want rendered videos or annotated frames, `frames_dir` must be provided.

## MOT Conversion And Evaluation

Convert ground truth:

```bash
python src/convert_gt_to_mot.py \
  --ground-truth-json /path/to/annotations.json \
  --out reports/runs/example/ground_truth.txt
```

Evaluate predictions:

```bash
python src/evaluate_mot.py --config configs/evaluation/default.yaml
```

## Outputs

- MOT predictions under `2d-tracking/reports/runs/`
- optional rendered videos or annotated frames
- summary CSV and JSON metrics under `2d-tracking/reports/summary/`

## Notes

- the tracker expects the common detection JSON format with `File` and `Labels`
- missing render frames fall back to blank images so long runs can continue
- class labels are preserved through matching, which helps avoid cross-class association if more classes are introduced later

## Path Conventions

All relative paths in configs resolve from `2d-tracking`.
