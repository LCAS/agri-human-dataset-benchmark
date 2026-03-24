"""
Objective: Keep the Norfair logic from the official repository (v2.3.x) but
using *native bbox mode* (boxes as two points) and *native tracker arguments*
exposed in the CLI.
"""
from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2

from norfair import Detection, Tracker
from norfair import drawing
from norfair.drawing.color import Palette

# ==============================
# CONFIG
# ==============================
@dataclass
class Config:
    # Path to the detections JSON and to the images (frames folder)
    detections_json: Path
    frames_dir: Optional[Path] = None  # If left as None, no video is drawn
    output_video: Optional[Path] = None  # Optional .mp4 (size inferred from the first frame)

    # Tracker hyperparameters
    distance_function: str = "iou"  # use Norfair's native IoU
    distance_threshold: float = 0.75
    hit_counter_max: int = 15
    initialization_delay: Optional[int] = None  # default = hit_counter_max / 2
    detection_threshold: float = 0.0  # do not filter detections by score (if there is no score)

    # MOT outputs for motmetrics (MOTChallenge format: frame,id,x,y,w,h,1,-1,-1,-1)
    mot_output: Optional[Path] = None

    # Drawing
    box_thickness: int = 2
    id_offset: int = 10  # pixels to separate the text from the top edge of the box
    save_frames_dir: Optional[Path] = None  # save one PNG per frame with drawn bboxes


# ==============================
# FORMAT UTILITIES
# ==============================
def _to_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert [x, y, w, h] → [x1, y1, x2, y2].
    If there is *no* positive width/height, assume it already comes as [x1, y1, x2, y2].
    """
    x, y, w, h = map(float, b)
    # If w,h are positive, treat them as xywh format
    if w > 0 and h > 0:
        return x, y, x + w, y + h
    # Otherwise, assume it is already xyxy
    return x, y, w, h


def _labels_iter(labels_field) -> Iterable[Tuple[str, Tuple[float, float, float, float]]]:
    """Return (label, (x1,y1,x2,y2)) for any form in which `Labels` may come.
    Supports:
      - {"Class": str, "BoundingBoxes": [x,y,w,h]}
      - {"label": str, "bbox": [x,y,w,h]}
      - [{...}, {...}] list of objects
      - [] no detections
    """
    if labels_field is None:
        return []

    if isinstance(labels_field, dict):
        lab = labels_field.get("Class") or labels_field.get("class") or labels_field.get("label") or labels_field.get("Label")
        bb = labels_field.get("BoundingBoxes") or labels_field.get("bbox") or labels_field.get("box")
        if lab is not None and bb is not None:
            yield str(lab), _to_xyxy(bb)
        return

    if isinstance(labels_field, list):
        for item in labels_field:
            if not isinstance(item, dict):
                continue
            lab = item.get("Class") or item.get("class") or item.get("label") or item.get("Label")
            bb = item.get("BoundingBoxes") or item.get("bbox") or item.get("box")
            if lab is None or bb is None:
                continue
            yield str(lab), _to_xyxy(bb)
        return

    # Other type: ignore it
    return []


def detections_from_json_record(rec: Dict) -> List[Detection]:
    dets: List[Detection] = []
    dropped = 0
    for lab, (x1, y1, x2, y2) in _labels_iter(rec.get("Labels")):
        # Normalize and validate bbox
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        # Norfair bbox: two points [[x1,y1],[x2,y2]]
        pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        dets.append(Detection(points=pts, scores=None, label=lab))
    if dropped:
        import warnings
        warnings.warn(f"Dropped {dropped} invalid bbox(es) in record {rec.get('File', '<no-file>')}")
    return dets


# ==============================
# RENDERING and MOT EXPORT
# ==============================
class MotWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("")  # truncate

    def write_frame(self, frame_idx: int, tracked_objects: Sequence):
        # MOT format: frame,id,x,y,w,h,conf,cx,cy,cz. conf=1 (no confidence score) and the last three fields are -1 (ignored)
        lines = []
        for tobj in tracked_objects:
            # estimate as bbox: [[x1,y1],[x2,y2]]
            x1, y1 = tobj.estimate[0]
            x2, y2 = tobj.estimate[1]
            w = x2 - x1
            h = y2 - y1
            lines.append(f"{frame_idx},{tobj.id},{x1:f},{y1:f},{w:f},{h:f},1,-1,-1,-1\n")
        with open(self.path, "a", encoding="utf-8") as f:
            f.writelines(lines)

# ==============================
# CLI UTILITIES
# ==============================
def _optional_int(value: str) -> Optional[int]:
    value = str(value).strip()
    if value.lower() == "none":
        return None
    return int(value)

# ==============================
# MAIN PIPELINE
# ==============================
def run(cfg: Config):
    # Load JSON
    recs = json.loads(Path(cfg.detections_json).read_text(encoding="utf-8"))

    # Norfair tracker with native arguments
    tracker = Tracker(
        distance_function=cfg.distance_function,
        distance_threshold=float(cfg.distance_threshold),
        hit_counter_max=cfg.hit_counter_max,
        initialization_delay=cfg.initialization_delay,
        detection_threshold=cfg.detection_threshold,
    )

    # Optional MOT writer
    mot_writer = MotWriter(cfg.mot_output) if cfg.mot_output else None

    # Optional video
    writer = None
    first_frame_size = None

    # Iterate frames in order
    for frame_idx, rec in enumerate(recs, start=1):
        # Create Norfair detections (bbox)
        dets = detections_from_json_record(rec)

        # Update tracker
        tracked_objects = tracker.update(dets)

        # MOT export
        if mot_writer:
            mot_writer.write_frame(frame_idx, tracked_objects)

        # Optional drawing
        if cfg.frames_dir is not None:
            file_name = Path(rec.get("File", f"frame_{frame_idx:05d}.png")).name
            frame_path = cfg.frames_dir / file_name
            frame = cv2.imread(str(frame_path))
            if frame is None:
                # If the frame is missing, generate a black canvas with heuristic size
                if first_frame_size is None:
                    first_frame_size = (720, 1280, 3)
                frame = np.zeros(first_frame_size, dtype=np.uint8)
            else:
                first_frame_size = frame.shape

            # Draw tracked objects as boxes
            drawing.draw_boxes(
                frame,
                drawables=tracked_objects,
                draw_ids=False,
                color="by_id",
                thickness=cfg.box_thickness,
            )
            for tobj in tracked_objects:
                x1, y1 = tobj.estimate[0]
                text = f"ID{tobj.id}"
                pos = (int(x1), max(0, int(y1) - cfg.id_offset))
                color = Palette.choose_color(tobj.id)
                cv2.putText(
                    frame,
                    text,
                    pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Write to video if requested
            if cfg.output_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(cfg.output_video), fourcc, 30.0, (w, h))
                writer.write(frame)

            # Save one PNG per frame if requested
            if cfg.save_frames_dir is not None:
                cfg.save_frames_dir.mkdir(parents=True, exist_ok=True)
                out_path = cfg.save_frames_dir / file_name
                cv2.imwrite(str(out_path), frame)

    if writer is not None:
        writer.release()


# ==============================
# MAIN
# ==============================
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path, help="Path to the detections JSON file")
    p.add_argument("--frames_dir", type=Path, default=None, help="Folder containing the frames (optional)")
    p.add_argument("--out_mot", type=Path, default=None, help="Output MOT file (optional)")
    p.add_argument("--out_video", type=Path, default=None, help="Output MP4 with drawn bounding boxes (optional)")
    p.add_argument("--distance_function", type=str, default="iou", help="Norfair distance function")
    p.add_argument("--distance_threshold", type=float, default=0.75, help="Norfair distance threshold for matching")
    p.add_argument("--hit_counter_max", type=int, default=15, help="Maximum missed frames before track deletion")
    p.add_argument("--initialization_delay", type=_optional_int, default=None, help="Frames required to confirm a track (default: hit_counter_max/2)")
    p.add_argument("--box_thickness", type=int, default=2, help="Bounding box line thickness (px)")
    p.add_argument("--id_offset", type=int, default=10, help="Vertical offset for the ID text (px)")
    p.add_argument("--save_frames_dir", type=Path, default=None, help="Folder to save PNG images with drawn bounding boxes (optional)")

    args = p.parse_args()

    cfg = Config(
        detections_json=args.json,
        frames_dir=args.frames_dir,
        output_video=args.out_video,
        distance_function=args.distance_function,
        distance_threshold=args.distance_threshold,
        hit_counter_max=args.hit_counter_max,
        initialization_delay=args.initialization_delay,
        mot_output=args.out_mot,
        box_thickness=args.box_thickness,
        id_offset=args.id_offset,
        save_frames_dir=args.save_frames_dir,
    )

    run(cfg)

if __name__ == "__main__":
        main()