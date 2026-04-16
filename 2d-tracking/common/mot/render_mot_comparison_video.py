"""
Render a comparison video with ground-truth and predicted MOT boxes over frames.
Relative CLI paths resolve from the 2d-tracking root.

Example:
    python common/mot/render_mot_comparison_video.py \
      --frame-index-json reports/runs/detections/example_detections.json \
      --frames-dir /path/to/frames \
      --gt-mot reports/runs/GTs_MOT/example.txt \
      --pred-mot reports/runs/example/tracks.txt \
      --out-video reports/runs/example/comparison.mp4

    python common/mot/render_mot_comparison_video.py \
        --frame-index-json reports/runs/detections/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb_yolo11s2_detections.json \
        --frames-dir D:/AOC/datasets/aghri-val/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/sensor_data/cam_zed_rgb \
        --gt-mot reports/runs/GTs_MOT/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb.txt \
        --pred-mot reports/runs/tracker_suite_zedrgb_all_trackers/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb_yolo11s2_detections/boxmot_botsort_tracks.txt \
        --out-video reports/runs/tracker_suite_zedrgb_all_trackers/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb_yolo11s2_detections/boxmot_botsort_vs_gt.mp4 \
        --title "footpath1_oj zed_rgb | BoTSORT"

    python common/mot/render_mot_comparison_video.py --frame-index-json reports/runs/detections/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb_yolo11s2_detections.json --frames-dir D:\AOC\datasets\aghri-val\footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label\sensor_data\cam_zed_rgb --gt-mot reports/runs/GTs_MOT/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb.txt --pred-mot reports/runs/tracker_suite_zedrgb_all_trackers/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb_yolo11s2_detections/boxmot_botsort_tracks.txt --out-video reports/runs/tracker_suite_zedrgb_all_trackers/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label_zed_rgb_yolo11s2_detections/boxmot_botsort_vs_gt.mp4 --title "footpath1_oj zed_rgb | BoTSORT vs GT"

    python common/mot/render_mot_comparison_video.py --frame-index-json reports/runs/detections/in_straw_3pick_diff_st_10_24_2024_5_a_label_zed_rgb_yolo11s2_detections.json --frames-dir D:\AOC\datasets\aghri-val\in_straw_3pick_diff_st_10_24_2024_5_a_label\sensor_data\cam_zed_rgb --gt-mot reports/runs/GTs_MOT/in_straw_3pick_diff_st_10_24_2024_5_a_label_zed_rgb.txt --pred-mot reports/runs/tracker_suite_zedrgb_all_trackers/in_straw_3pick_diff_st_10_24_2024_5_a_label_zed_rgb_yolo11s2_detections/boxmot_botsort_tracks.txt --out-video reports/runs/tracker_suite_zedrgb_all_trackers/in_straw_3pick_diff_st_10_24_2024_5_a_label_zed_rgb_yolo11s2_detections/boxmot_botsort_vs_gt.mp4 --title "in_straw zed_rgb | BoTSORT vs GT"

    python common/mot/render_mot_comparison_video.py --frame-index-json reports/runs/detections/out_vine_4swap+walk_st_ly_11_06_2024_2_label_zed_rgb_yolo11s2_detections.json --frames-dir D:\AOC\datasets\aghri-val\out_vine_4swap+walk_st_ly_11_06_2024_2_label\sensor_data\cam_zed_rgb --gt-mot reports/runs/GTs_MOT/out_vine_4swap+walk_st_ly_11_06_2024_2_label_zed_rgb.txt --pred-mot reports/runs/tracker_suite_zedrgb_all_trackers/out_vine_4swap+walk_st_ly_11_06_2024_2_label_zed_rgb_yolo11s2_detections/boxmot_botsort_tracks.txt --out-video reports/runs/tracker_suite_zedrgb_all_trackers/out_vine_4swap+walk_st_ly_11_06_2024_2_label_zed_rgb_yolo11s2_detections/boxmot_botsort_vs_gt.mp4 --title "out_vine zed_rgb | BoTSORT vs GT"
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Deque, Dict, Iterable, List, Tuple

import cv2
import numpy as np


TRACKING_ROOT = Path(__file__).resolve().parents[2]
GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 80, 255)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (24, 24, 24)
LABEL_FONT_SCALE = 0.42
LABEL_THICKNESS = 1
LABEL_PADDING_X = 4
LABEL_PADDING_Y = 3
LABEL_BG_COLOR = (20, 20, 20)


@dataclass(frozen=True)
class MotBox:
    track_id: int
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Tuple[int, int]:
        return int(round(self.x + self.w / 2.0)), int(round(self.y + self.h / 2.0))


def resolve_path(path: Path) -> Path:
    """Resolve repo-relative CLI paths from the 2d-tracking root."""

    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def load_mot(path: Path) -> Dict[int, List[MotBox]]:
    """Read one MOT text file into a frame-indexed dictionary of boxes."""

    by_frame: DefaultDict[int, List[MotBox]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 6:
                continue
            frame_idx = int(float(row[0]))
            by_frame[frame_idx].append(
                MotBox(
                    track_id=int(float(row[1])),
                    x=float(row[2]),
                    y=float(row[3]),
                    w=float(row[4]),
                    h=float(row[5]),
                )
            )
    return dict(by_frame)


def frame_files_from_json(path: Path) -> List[str]:
    """Load frame filenames from the frame-ordered detector export JSON."""

    records = json.loads(path.read_text(encoding="utf-8"))
    return [str(record.get("File")) for record in records]


def draw_legend(frame, title: str, frame_idx: int, total_frames: int) -> None:
    """Draw a compact status panel in the top-left corner."""

    cv2.rectangle(frame, (12, 12), (420, 92), PANEL_COLOR, thickness=-1)
    cv2.putText(frame, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Frame {frame_idx}/{total_frames}",
        (24, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "GT", (500, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, "Prediction", (550, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PRED_COLOR, 2, cv2.LINE_AA)


def draw_trails(
    frame,
    boxes: Iterable[MotBox],
    history: Dict[int, Deque[Tuple[int, int]]],
    color: Tuple[int, int, int],
) -> None:
    """Update and draw short center-point trails for one set of tracks."""

    for box in boxes:
        history.setdefault(box.track_id, deque(maxlen=20)).append(box.center)

    for points in history.values():
        if len(points) < 2:
            continue
        cv2.polylines(
            frame,
            [np.asarray(points, dtype=np.int32)],
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )


def draw_boxes(
    frame,
    boxes: Iterable[MotBox],
    color: Tuple[int, int, int],
    prefix: str,
    label_y_offset: int,
) -> None:
    """Draw bounding boxes and track IDs for one track source."""

    for box in boxes:
        x1 = int(round(box.x))
        y1 = int(round(box.y))
        x2 = int(round(box.x + box.w))
        y2 = int(round(box.y + box.h))
        label = f"{prefix}:{box.track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            LABEL_THICKNESS,
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        tag_width = text_width + (2 * LABEL_PADDING_X)
        tag_height = text_height + baseline + (2 * LABEL_PADDING_Y)
        tag_x1 = max(6, min(frame.shape[1] - tag_width - 6, x1))
        tag_y1 = max(6, y1 - tag_height - 2 + label_y_offset)
        tag_x2 = tag_x1 + tag_width
        tag_y2 = tag_y1 + tag_height
        text_x = tag_x1 + LABEL_PADDING_X
        text_y = tag_y1 + LABEL_PADDING_Y + text_height

        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), LABEL_BG_COLOR, thickness=-1)
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, thickness=1)

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            TEXT_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    """Define the CLI for GT-vs-prediction rendering."""

    parser = argparse.ArgumentParser(description="Render GT and predicted MOT tracks over video frames.")
    parser.add_argument("--frame-index-json", required=True, type=Path, help="Frame-ordered JSON used to recover frame filenames.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing image frames.")
    parser.add_argument("--gt-mot", required=True, type=Path, help="Ground-truth MOT txt file.")
    parser.add_argument("--pred-mot", required=True, type=Path, help="Predicted MOT txt file.")
    parser.add_argument("--out-video", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--title", type=str, default="GT vs Prediction", help="Overlay title.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for GT/prediction comparison video rendering."""

    args = parse_args()
    frame_index_json = resolve_path(args.frame_index_json)
    frames_dir = resolve_path(args.frames_dir)
    gt_mot = resolve_path(args.gt_mot)
    pred_mot = resolve_path(args.pred_mot)
    out_video = resolve_path(args.out_video)

    frame_files = frame_files_from_json(frame_index_json)
    gt_by_frame = load_mot(gt_mot)
    pred_by_frame = load_mot(pred_mot)

    if not frame_files:
        raise ValueError(f"No frame records found in {frame_index_json}")

    first_frame = cv2.imread(str(frames_dir / frame_files[0]))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read frame: {frames_dir / frame_files[0]}")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )

    gt_history: Dict[int, Deque[Tuple[int, int]]] = {}
    pred_history: Dict[int, Deque[Tuple[int, int]]] = {}

    for frame_idx, file_name in enumerate(frame_files, start=1):
        frame_path = frames_dir / file_name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        gt_boxes = gt_by_frame.get(frame_idx, [])
        pred_boxes = pred_by_frame.get(frame_idx, [])

        draw_legend(frame, args.title, frame_idx, len(frame_files))
        draw_trails(frame, gt_boxes, gt_history, GT_COLOR)
        draw_trails(frame, pred_boxes, pred_history, PRED_COLOR)
        draw_boxes(frame, gt_boxes, GT_COLOR, "GT", -6)
        draw_boxes(frame, pred_boxes, PRED_COLOR, "PR", 16)
        writer.write(frame)

    writer.release()
    print(f"Comparison video written: {out_video}")


if __name__ == "__main__":
    main()
