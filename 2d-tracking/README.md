# 2D Tracking Workspace

This workspace benchmarks multi-object tracking on top of the detector outputs produced by `2d-detection`.

Its role in the repository is to answer a different safety question from detection alone: not only can a robot see a person, but can it keep a stable identity over time well enough to support motion reasoning, warning logic, and later fusion with other sensors.

## What This Workspace Is For

Use `2d-tracking` to:

- run multiple tracker families on a shared detection export format
- compare tracker behaviour under different detector quality levels
- generate MOTChallenge-format tracking outputs
- evaluate trackers with the same MOT metrics across frameworks

## Structure

- `common/mot`: shared MOT conversion and evaluation utilities
- `benchmarks/boxmot`: BoxMOT-based tracking wrappers
- `benchmarks/deepsort`: local DeepSORT implementation with frozen-graph ReID
- `benchmarks/norfair`: Norfair-based tracking wrapper
- `reports`: generated tracking runs and summary metrics

## Tracking Workflow

1. Start from a frame-ordered detections JSON file, typically exported by `2d-detection`.
2. Run one tracker framework to produce a MOT-format prediction file.
3. Convert project ground truth to MOT format with `common/mot/convert_gt_to_mot.py`.
4. Evaluate predictions with `common/mot/evaluate_mot.py`.
5. Compare summary metrics across trackers and datasets.

## Common Input Format

All implemented trackers read the same repository JSON shape:

```json
[
  {
    "File": "frame_000001.png",
    "Labels": [
      {
        "Class": "person",
        "BoundingBoxes": [100.0, 50.0, 80.0, 180.0]
      }
    ]
  }
]
```

`File` is the frame name, and `BoundingBoxes` are expected in `x, y, w, h` form.

## Frameworks

### BoxMOT

Supports several modern trackers behind one wrapper.

- local docs: [`benchmarks/boxmot/README.md`](benchmarks/boxmot/README.md)

### DeepSORT

Implements a dedicated DeepSORT pipeline with TensorFlow frozen-graph appearance embeddings.

- local docs: [`benchmarks/deepsort/README.md`](benchmarks/deepsort/README.md)

### Norfair

Provides a lightweight tracker benchmark with simple configuration and optional rendering.

- local docs: [`benchmarks/norfair/README.md`](benchmarks/norfair/README.md)

### Shared MOT Utilities

- local docs: [`common/mot/README.md`](common/mot/README.md)

## Reports

- `reports/runs`: per-run tracker outputs, videos, and intermediate artifacts
- `reports/summary`: shared MOT evaluation summaries across tracking frameworks

## Path Conventions

Tracking configs resolve relative paths from `2d-tracking` unless the path is already absolute. That applies to:

- detections JSON inputs
- frame directories
- MOT outputs
- rendered videos and frame dumps
- evaluation CSV and JSON summaries

## Relationship To The Wider Benchmark

The current code here is 2D tracking, but it is meant to be the temporal evaluation layer of a larger agricultural safety benchmark. In future extensions, the same ideas can be applied to 3D tracking and multimodal fusion outputs built from RGB, fisheye, LiDAR, and rosbag-derived streams.
