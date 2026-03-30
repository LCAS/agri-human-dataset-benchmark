"""
Benchmark-local wrapper for the shared MOT ground-truth conversion utility.
"""
from __future__ import annotations

import sys
from pathlib import Path

TRACKING_ROOT = Path(__file__).resolve().parents[3]
COMMON_ROOT = TRACKING_ROOT / "common"
if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from mot.convert_gt_to_mot import class_to_track_id, convert_gt_to_mot, labels_iter, main, parse_args, resolve_path

__all__ = (
    "class_to_track_id",
    "convert_gt_to_mot",
    "labels_iter",
    "main",
    "parse_args",
    "resolve_path",
)


if __name__ == "__main__":
    main()
