"""
Bulk-convert all scenario GT annotations under reports/GTs into MOT format.
Relative CLI paths resolve from the 2d-tracking root.

Examples:
    python common/mot/convert_all_gt_to_mot.py
    python common/mot/convert_all_gt_to_mot.py --skip-existing
    python common/mot/convert_all_gt_to_mot.py --include in_straw_3pick_diff_st_10_24_2024_5_a_label
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from convert_gt_to_mot import TRACKING_ROOT, convert_gt_to_mot, resolve_path


CAMERA_SUFFIXES = {
    "cam_fish_front_ann": "front_fisheye",
    "cam_zed_rgb_ann": "zed_rgb",
}


def iter_annotation_jsons(gt_root: Path, includes: set[str] | None) -> Iterable[Path]:
    """Yield annotation JSON files for each scenario directory under gt_root."""

    for scenario_dir in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        if includes and scenario_dir.name not in includes:
            continue
        for annotation_json in sorted(scenario_dir.glob("*_ann.json")):
            yield annotation_json


def output_path_for(annotation_json: Path, output_dir: Path) -> Path:
    """Map one GT annotation file to the repo's MOT output naming convention."""

    scenario_name = annotation_json.parent.name
    camera_name = CAMERA_SUFFIXES.get(annotation_json.stem, annotation_json.stem.removesuffix("_ann"))
    return output_dir / f"{scenario_name}_{camera_name}.txt"


def parse_args() -> argparse.Namespace:
    """Define the CLI for bulk GT conversion."""

    parser = argparse.ArgumentParser(
        description="Convert all GT annotation JSON files under reports/GTs into MOT format.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("reports/GTs"),
        help="Directory containing one subdirectory per GT scenario.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/runs/GTs_MOT"),
        help="Directory where MOT txt files will be written.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Optional list of scenario directory names to convert.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing MOT files that already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for batch GT-to-MOT conversion."""

    args = parse_args()
    gt_root = resolve_path(args.gt_root)
    output_dir = resolve_path(args.out_dir)
    includes = set(args.include) if args.include else None

    converted = 0
    skipped = 0

    for annotation_json in iter_annotation_jsons(gt_root, includes):
        output_path = output_path_for(annotation_json, output_dir)
        if args.skip_existing and output_path.exists():
            print(f"Skipping existing file: {output_path}")
            skipped += 1
            continue

        convert_gt_to_mot(annotation_json, output_path)
        converted += 1

    print(f"Finished bulk GT conversion from {gt_root}")
    print(f"Converted files: {converted}, Skipped existing: {skipped}")


if __name__ == "__main__":
    main()
