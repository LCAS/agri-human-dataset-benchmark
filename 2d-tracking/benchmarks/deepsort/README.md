# DeepSORT Tracking Benchmark

This benchmark provides a local DeepSORT implementation tuned for the repository's common detection export format.

## Structure

- `src/run_tracker.py`: main DeepSORT tracking entrypoint
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

## Important Requirements

DeepSORT in this repository is appearance-based and requires:

- a real `frames_dir`
- a frozen TensorFlow ReID graph, usually a `.pb` file such as `mars-small128.pb`

Without those two inputs, the tracker cannot run correctly.

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
  --model-path /path/to/mars-small128.pb \
  --out-mot reports/runs/example/deepsort_tracks.txt \
  --out-video reports/runs/example/deepsort_tracking.mp4
```

## Config Notes

Key runtime fields include:

- `min_confidence`
- `min_detection_height`
- `nms_max_overlap`
- `max_cosine_distance`
- `nn_budget`
- `max_iou_distance`
- `max_age`
- `n_init`
- `encoder_batch_size`
- `reid_input_name`
- `reid_output_name`

The default tensor names are `images` and `features`, which match common MARS-style frozen graphs.

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

## Path Conventions

All relative paths in configs resolve from `2d-tracking`.
