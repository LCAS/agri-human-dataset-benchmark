"""
Convert a ground-truth annotations JSON file into MOTChallenge format.
Relative CLI paths resolve from the 2d-tracking root.

Examples:
    python src/convert_gt_to_mot.py --ground-truth-json /path/to/annotations.json --out reports/runs/example/ground_truth.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

TRACKING_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def _labels_iter(record: Dict[str, Any]) -> Iterable[Tuple[str, Tuple[float, float, float, float]]]:
    # Ground-truth records follow the same File/Labels pattern as detector exports.
    for label in record.get("Labels", []):
        if not isinstance(label, dict):
            continue
        class_name = label.get("Class") or label.get("class") or label.get("label")
        bbox = label.get("BoundingBoxes") or label.get("bbox") or label.get("box")
        if class_name is None or bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        yield str(class_name), (x, y, w, h)


def class_to_track_id(class_name: str) -> int:
    # Prefer explicit numeric suffixes such as "human7". Fall back to a stable
    # hash so repeated runs keep the same IDs for non-numeric labels.
    digits = "".join(char for char in class_name if char.isdigit())
    if digits:
        return int(digits)

    digest = hashlib.sha1(class_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def convert_gt_to_mot(annotations_json: Path, output_path: Path) -> None:
    records = json.loads(annotations_json.read_text(encoding="utf-8"))
    lines = []

    for frame_idx, record in enumerate(records, start=1):
        for class_name, (x, y, w, h) in _labels_iter(record):
            track_id = class_to_track_id(class_name)
            lines.append(f"{frame_idx},{track_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")

    print(f"GT MOT file written: {output_path}")
    print(f"Total frames: {len(records)}, Total lines: {len(lines)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert annotations JSON into MOT format.",
        epilog=(
            "Example:\n"
            "  python src/convert_gt_to_mot.py --ground-truth-json /path/to/annotations.json "
            "--out reports/runs/example/ground_truth.txt"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ground-truth-json", required=True, type=Path, help="Ground-truth annotations JSON file.")
    parser.add_argument("--out", required=True, type=Path, help="Output MOT txt file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_gt_to_mot(_resolve_path(args.ground_truth_json), _resolve_path(args.out))


if __name__ == "__main__":
    main()
