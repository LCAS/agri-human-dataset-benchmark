# BoxMOT Tracking Benchmark

This benchmark wraps several BoxMOT trackers behind one consistent interface so they can be compared on the same detection export and evaluated with the same MOT metrics.

## Structure

- `src/run_tracker.py`: main BoxMOT tracking entrypoint
- `src/convert_gt_to_mot.py`: wrapper around the shared GT-to-MOT utility
- `src/evaluate_mot.py`: wrapper around the shared MOT evaluation utility
- `configs/tracking`: tracking configs
- `configs/evaluation`: evaluation configs
- `scripts`: local and cluster launchers

## Supported Trackers

- `boosttrack`
- `botsort`
- `bytetrack`
- `deepocsort`
- `ocsort`
- `strongsort`

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

Switch tracker from the CLI:

```bash
python src/run_tracker.py \
  --config configs/tracking/default.yaml \
  --tracker boosttrack \
  --reid-weights /path/to/osnet_x0_25_msmt17.pt
```

Override tracker-specific parameters:

```bash
python src/run_tracker.py \
  --config configs/tracking/default.yaml \
  --tracker botsort \
  --reid-weights /path/to/osnet_x0_25_msmt17.pt \
  --tracker-kwarg track_buffer=60 \
  --tracker-kwarg with_reid=true
```

## Config Notes

The base config controls:

- `detections_json`
- `frames_dir`
- `mot_output`
- `output_video`
- `save_frames_dir`
- `frame_rate`
- `tracker`
- `reid_weights`
- `device`
- `half`
- `per_class`
- `tracker_kwargs`

`tracker_kwargs` are forwarded to the selected BoxMOT tracker after validation against the installed version's constructor.

## When Frames Matter

Some BoxMOT trackers can technically run with blank fallback frames, but appearance-heavy trackers usually should not. Real frames are strongly recommended for:

- `strongsort`
- `botsort` with ReID enabled
- `boosttrack` when ReID is enabled
- `deepocsort` unless embedding extraction is disabled

If `frames_dir` is missing and a frame cannot be loaded, the wrapper synthesizes a blank canvas so the run can continue, but that is a fallback, not the preferred operating mode.

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
- optional rendered MP4 or frame dumps under the configured run directory
- summary CSV and JSON metrics under `2d-tracking/reports/summary/`

## Path Conventions

All relative paths in configs resolve from `2d-tracking`.
