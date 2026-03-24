# Norfair Tracking Scripts

This repository contains two custom scripts built around [Norfair](https://github.com/tryolabs/norfair) to reproduce tracking results on datasets organized with separate `Annotation/` and `Data/` folders.

## Included scripts

- `my_norfair.py`: runs tracking for a single detections JSON file and its corresponding frames folder.
- `batch_my_norfair.py`: scans a dataset root, finds valid scenario/camera combinations, runs `my_norfair.py` for each one, and stores all outputs automatically.

## Repository purpose

The scripts in this repository are intended to:

- export tracking results in MOT format,
- optionally generate an annotated MP4 video,
- optionally save annotated PNG frames,
- support both single-case execution and batch processing across multiple scenarios/cameras.

## Requirements

This project uses the Conda environment defined in `environment.yml`.

Pinned versions:

- Python 3.10
- numpy 1.26
- opencv-python 4.8.1.78
- norfair 2.3.0

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate Norfair
```

For the complete Ubuntu setup, system dependencies, reproducibility notes, and troubleshooting, see (UBUNTU_SETUP.md).

## Quick start

Check the available CLI arguments:

```bash
python my_norfair.py --help
python batch_my_norfair.py --help
```

### Single-case example

```bash
python my_norfair.py \
  --json /path/to/json/cam_fish_left_ann.json \
  --frames_dir /path/to/frames/fisheye_images_12 \
  --out_mot /path/to/results.txt \
  --out_video /path/to/tracking.mp4 \
  --save_frames_dir /path/to/result_frames \
  --distance_function iou \
  --distance_threshold 0.75 \
  --hit_counter_max 15 \
  --initialization_delay None \
  --box_thickness 2 \
  --id_offset 15
```

### Batch example

```bash
python batch_my_norfair.py \
  --root /path/to/root \
  --distance_function iou \
  --distance_threshold 0.75 \
  --hit_counter_max 15 \
  --initialization_delay None \
  --box_thickness 2 \
  --id_offset 15
```

## Expected dataset structure

```text
<root>/
  Annotation/
    <scenario>_json_files/
      fisheye_images_12/cam_fish_left_ann.json
      fisheye_images_13/cam_fish_front_ann.json
      fisheye_images_14/cam_fish_right_ann.json
      output_images/cam_zed_rgb_ann.json
  Data/
    <scenario>_label/
      fisheye_images_12/ ...frames...
      fisheye_images_13/ ...frames...
      fisheye_images_14/ ...frames...
      output_images/     ...frames...
```

## Main parameters

### `my_norfair.py`

- `--json`: path to the detections JSON file.
- `--frames_dir`: folder containing the frames.
- `--out_mot`: output MOT file.
- `--out_video`: output MP4 with drawn bounding boxes.
- `--save_frames_dir`: folder to save annotated PNG frames.
- `--distance_function`: Norfair distance function, for example `iou`.
- `--distance_threshold`: matching threshold used by Norfair.
- `--hit_counter_max`: maximum missed frames before deletion.
- `--initialization_delay`: frames required to confirm a track.
- `--box_thickness`: bounding box line thickness.
- `--id_offset`: vertical offset for the ID text.

### `batch_my_norfair.py`

- `--root`: dataset root containing `Annotation/` and `Data/`.
- `--distance_function`
- `--distance_threshold`
- `--hit_counter_max`
- `--initialization_delay`
- `--box_thickness`
- `--id_offset`

## Outputs

### Single-case execution

Depending on the CLI arguments provided, `my_norfair.py` can generate:

- `results.txt`
- `tracking.mp4`
- `result_frames/`  (annotated PNG frames)

### Batch execution

`batch_my_norfair.py` generates one output folder per scenario/camera under:

```text
<root>/Results/Norfair/<scenario_label>/<cam>/
```

Typical outputs are:

- `results.txt`
- `tracking.mp4`
- `result_frames/` (annotated PNG frames)

## Notes

- `batch_my_norfair.py` does not change the tracking logic of `my_norfair.py`; it automates repeated executions.
- If your dataset uses different folder names or suffixes, adjust the matching logic in `batch_my_norfair.py`.
- Make sure you have enough disk space if frame export is enabled for long sequences.