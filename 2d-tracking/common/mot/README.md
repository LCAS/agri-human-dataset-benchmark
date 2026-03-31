# Shared MOT Utilities

This directory contains tracker-agnostic utilities shared by the tracking benchmarks.

## What It Does

- convert project annotation JSON into MOTChallenge ground-truth text
- evaluate predicted MOT files against ground truth using `motmetrics`

These utilities keep benchmark logic consistent across BoxMOT, DeepSORT, and Norfair.

## Files

- `convert_gt_to_mot.py`: convert project annotations to MOT text
- `evaluate_mot.py`: evaluate one predicted MOT file against one GT file
- `__init__.py`: package marker

## Annotation Input Format

The converter expects the same frame-wise JSON style used elsewhere in the repository:

```json
[
  {
    "File": "frame_000001.png",
    "Labels": [
      {
        "Class": "person7",
        "BoundingBoxes": [100.0, 50.0, 80.0, 180.0]
      }
    ]
  }
]
```

`convert_gt_to_mot.py` will:

- read frame records in order
- extract `Class` and `BoundingBoxes`
- convert each box to MOT `frame,id,x,y,w,h,...`
- derive a stable track id from numeric suffixes when present
- fall back to a stable hash for non-numeric labels

## Convert Ground Truth

```bash
python common/mot/convert_gt_to_mot.py \
  --ground-truth-json /path/to/annotations.json \
  --out reports/runs/example/ground_truth.txt
```

## Evaluate A Prediction

Use one of the benchmark-specific wrappers so the correct default config is supplied:

```bash
python benchmarks/boxmot/src/evaluate_mot.py \
  --config benchmarks/boxmot/configs/evaluation/default.yaml
```

Or with direct overrides:

```bash
python benchmarks/boxmot/src/evaluate_mot.py \
  --config benchmarks/boxmot/configs/evaluation/default.yaml \
  --gt reports/runs/example/ground_truth.txt \
  --pred reports/runs/example/tracks.txt \
  --out-csv reports/summary/example_metrics.csv
```

## Evaluation Metrics

The default metric list includes:

- `idf1`
- `idp`
- `idr`
- `precision`
- `recall`
- `mota`
- `motp`
- `num_switches`
- `num_false_positives`
- `num_misses`
- `mostly_tracked`
- `mostly_lost`

## Path Convention

Relative paths resolve from `2d-tracking`, which keeps configs portable across benchmark wrappers.
