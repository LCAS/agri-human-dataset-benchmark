"""
Render a comparison video with ground-truth and detector boxes over frames.
Relative CLI paths resolve from the 2d-detection workspace first, then the
repository root so GT annotations can be referenced from 2d-tracking.

Example:
    python common/render_detection_comparison_video.py ^
      --gt-json "D:/AOC/datasets/aghri-val/out_vine_4swap+walk_st_ly_11_06_2024_2_label/annotations/cam_zed_rgb_ann.json" ^
      --pred-json "D:/AOC/agri-human-dataset-benchmark/2d-tracking/reports/runs/detections/out_vine_4swap+walk_st_ly_11_06_2024_2_label_zed_rgb_yolo11s2_detections.json" ^
      --frames-dir "D:/AOC/datasets/aghri-val/out_vine_4swap+walk_st_ly_11_06_2024_2_label/sensor_data/cam_zed_rgb" ^
      --out-video "reports/detections/out_vine_4swap+walk_st_ly_11_06_2024_2_label_yolo11s_vs_gt.mp4" ^
      --title "out_vine_4swap+walk_st_ly_11_06_2024_2_label | detector vs GT"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_ROOT.parent
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
class JsonBox:
    label: str
    x: float
    y: float
    w: float
    h: float
    confidence: Optional[float] = None


def resolve_path(path: Path) -> Path:
    """Resolve CLI paths from the detection workspace, then the repo root."""

    if path.is_absolute():
        return path

    workspace_path = (WORKSPACE_ROOT / path).resolve()
    if workspace_path.exists():
        return workspace_path

    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path

    return workspace_path


def parse_bbox(raw_box: object) -> Optional[Tuple[float, float, float, float]]:
    """Normalize one bbox value into x, y, w, h form."""

    if not isinstance(raw_box, Sequence) or isinstance(raw_box, (str, bytes)) or len(raw_box) != 4:
        return None
    return tuple(float(value) for value in raw_box)


def parse_confidence(label_record: Dict[str, object]) -> Optional[float]:
    """Extract an optional confidence field from one label record."""

    for key in ("Confidence", "confidence", "score"):
        value = label_record.get(key)
        if value is not None:
            return float(value)
    return None


def load_boxes_by_frame(path: Path) -> Tuple[List[str], Dict[str, List[JsonBox]]]:
    """Read one frame-wise JSON file into ordered filenames and per-frame boxes."""

    records = json.loads(path.read_text(encoding="utf-8"))
    frame_files: List[str] = []
    boxes_by_frame: Dict[str, List[JsonBox]] = {}
    seen = set()

    for record in records:
        if not isinstance(record, dict):
            continue

        file_name = record.get("File")
        if not file_name:
            continue

        file_name = str(file_name)
        if file_name not in seen:
            frame_files.append(file_name)
            seen.add(file_name)

        boxes: List[JsonBox] = []
        for label_record in record.get("Labels", []):
            if not isinstance(label_record, dict):
                continue

            bbox = parse_bbox(
                label_record.get("BoundingBoxes")
                or label_record.get("bbox")
                or label_record.get("box")
            )
            if bbox is None:
                continue

            x, y, w, h = bbox
            boxes.append(
                JsonBox(
                    label=str(label_record.get("Class") or label_record.get("label") or "person"),
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=parse_confidence(label_record),
                )
            )

        boxes_by_frame[file_name] = boxes

    return frame_files, boxes_by_frame


def merged_frame_files(primary: Iterable[str], secondary: Iterable[str]) -> List[str]:
    """Preserve primary order and append any secondary-only frame filenames."""

    merged: List[str] = []
    seen = set()
    for file_name in list(primary) + list(secondary):
        if file_name in seen:
            continue
        merged.append(file_name)
        seen.add(file_name)
    return merged


def draw_legend(frame, title: str, frame_idx: int, total_frames: int) -> None:
    """Draw a compact status panel in the top-left corner."""

    cv2.rectangle(frame, (12, 12), (460, 92), PANEL_COLOR, thickness=-1)
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
    cv2.putText(frame, "Detection", (550, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PRED_COLOR, 2, cv2.LINE_AA)


def draw_boxes(
    frame,
    boxes: Iterable[JsonBox],
    color: Tuple[int, int, int],
    prefix: str,
    label_y_offset: int,
) -> None:
    """Draw bounding boxes and short labels for one source."""

    for box in boxes:
        x1 = int(round(box.x))
        y1 = int(round(box.y))
        x2 = int(round(box.x + box.w))
        y2 = int(round(box.y + box.h))

        label = f"{prefix}:{box.label}"
        if box.confidence is not None:
            label = f"{label} {box.confidence:.2f}"

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
    """Define the CLI for GT-vs-detection rendering."""

    parser = argparse.ArgumentParser(description="Render GT and detector boxes over video frames.")
    parser.add_argument("--gt-json", required=True, type=Path, help="Ground-truth annotation JSON.")
    parser.add_argument("--pred-json", required=True, type=Path, help="Detector output JSON.")
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing image frames.")
    parser.add_argument("--out-video", required=True, type=Path, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--title", type=str, default="GT vs Detection", help="Overlay title.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for GT/detection comparison video rendering."""

    args = parse_args()
    gt_json = resolve_path(args.gt_json)
    pred_json = resolve_path(args.pred_json)
    frames_dir = resolve_path(args.frames_dir)
    out_video = resolve_path(args.out_video)

    gt_frame_files, gt_by_frame = load_boxes_by_frame(gt_json)
    pred_frame_files, pred_by_frame = load_boxes_by_frame(pred_json)
    frame_files = merged_frame_files(gt_frame_files, pred_frame_files)

    if not frame_files:
        raise ValueError(f"No frame records found in {gt_json} or {pred_json}")

    first_frame_path = frames_dir / frame_files[0]
    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read frame: {first_frame_path}")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (width, height),
    )

    for frame_idx, file_name in enumerate(frame_files, start=1):
        frame_path = frames_dir / file_name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        gt_boxes = gt_by_frame.get(file_name, [])
        pred_boxes = pred_by_frame.get(file_name, [])

        draw_legend(frame, args.title, frame_idx, len(frame_files))
        draw_boxes(frame, gt_boxes, GT_COLOR, "GT", -6)
        draw_boxes(frame, pred_boxes, PRED_COLOR, "DT", 16)
        writer.write(frame)

    writer.release()
    print(f"Comparison video written: {out_video}")


if __name__ == "__main__":
    main()
