# DeepSORT Tracking Benchmark

## Structure
- `src/run_tracker.py`: main DeepSORT tracking entrypoint
- `src/convert_gt_to_mot.py`: wrapper around the shared MOT ground-truth conversion utility
- `src/evaluate_mot.py`: wrapper around the shared MOT evaluation utility
- `configs/tracking`: tracking run configs
- `configs/evaluation`: evaluation configs
- `scripts`: local and SLURM launchers
- `2d-tracking/common/mot`: shared tracker-agnostic MOT utilities
- `2d-tracking/reports/runs`: generated tracking outputs shared across tracking models
- `2d-tracking/reports/summary`: evaluation summaries shared across tracking models

## Environment

Install the pinned Python dependencies from `requirements.txt`:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Run a tracking job from the default config:

```bash
python src/run_tracker.py --config configs/tracking/default.yaml
```

Or use explicit arguments:

```bash
python src/run_tracker.py \
  --detections-json /path/to/detections.json \
  --frames-dir /path/to/frames \
  --model-path /path/to/mars-small128.pb \
  --out-mot reports/runs/example/deepsort_tracks.txt \
  --out-video reports/runs/example/deepsort_tracking.mp4
```

Convert ground truth into MOT format:

```bash
python src/convert_gt_to_mot.py \
  --ground-truth-json /path/to/annotations.json \
  --out reports/runs/example/ground_truth.txt
```

Evaluate predictions:

```bash
python src/evaluate_mot.py --config configs/evaluation/default.yaml
```

## Path Conventions

- Paths in tracking and evaluation YAML configs resolve from `2d-tracking`.
- Generated run artifacts belong under `2d-tracking/reports/runs/`.
- Compact evaluation summaries belong under `2d-tracking/reports/summary/`.

## Notes

- The tracking pipeline expects the detector export format with `File` and `Labels` fields.
- `frames_dir` is required because DeepSORT extracts appearance features from real image crops.
- `model_path` should point to a frozen TensorFlow ReID graph such as `mars-small128.pb`.
- `reid_input_name` and `reid_output_name` default to `images` and `features`, which matches common MARS exports.
