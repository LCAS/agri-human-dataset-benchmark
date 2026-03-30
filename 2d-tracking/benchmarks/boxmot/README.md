# BoxMOT Tracking Benchmark

## Structure
- `src/run_tracker.py`: main BoxMOT tracking entrypoint
- `src/convert_gt_to_mot.py`: wrapper around the shared MOT ground-truth conversion utility
- `src/evaluate_mot.py`: wrapper around the shared MOT evaluation utility
- `configs/tracking`: tracking run configs
- `configs/evaluation`: evaluation configs
- `scripts`: local and SLURM launchers
- `2d-tracking/common/mot`: shared tracker-agnostic MOT utilities
- `2d-tracking/reports/runs`: generated tracking outputs shared across tracking models
- `2d-tracking/reports/summary`: evaluation summaries shared across tracking models

## Supported Trackers

- `boosttrack`
- `botsort`
- `bytetrack`
- `deepocsort`
- `ocsort`
- `strongsort`

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

Switch trackers from the CLI:

```bash
python src/run_tracker.py \
  --config configs/tracking/default.yaml \
  --tracker boosttrack \
  --reid-weights /path/to/osnet_x0_25_msmt17.pt
```

Override tracker-specific parameters without editing YAML:

```bash
python src/run_tracker.py \
  --config configs/tracking/default.yaml \
  --tracker botsort \
  --reid-weights /path/to/osnet_x0_25_msmt17.pt \
  --tracker-kwarg track_buffer=60 \
  --tracker-kwarg with_reid=true
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
- Appearance-based trackers need `reid_weights`; real frames are strongly recommended for accurate ReID and camera-motion compensation.
- When a frame is missing on disk, `run_tracker.py` falls back to a blank canvas sized from the detections so long runs do not abort mid-sequence.
