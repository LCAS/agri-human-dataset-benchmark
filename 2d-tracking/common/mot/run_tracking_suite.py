"""
Run tracker predictions and MOT evaluation across available tracking inputs.
Relative CLI paths resolve from the 2d-tracking root.

Supported input modes:
- detector exports under reports/runs/detections
- GT annotations under reports/GTs, normalized into detector-style JSON so
  trackers can be run with "perfect detections" but without leaking GT IDs as
  semantic classes

Example:
    python common/mot/run_tracking_suite.py --skip-existing
    python common/mot/run_tracking_suite.py --sources gt --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TRACKING_ROOT = Path(__file__).resolve().parents[2]
CAMERA_SUFFIXES = {
    "cam_fish_front_ann": "front_fisheye",
    "cam_zed_rgb_ann": "zed_rgb",
}


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    framework: str
    run_script: Path
    eval_script: Path
    config_path: Path
    import_name: str
    requires_frames: bool = False
    reid_weights: Optional[Path] = None
    model_path: Optional[Path] = None


@dataclass(frozen=True)
class InputRecord:
    input_json: Path
    gt_mot: Path
    input_kind: str
    input_name: str
    run_name: str
    source_path: Path


TRACKER_SPECS = {
    "norfair": TrackerSpec(
        key="norfair",
        framework="norfair",
        run_script=Path("benchmarks/norfair/src/run_tracker.py"),
        eval_script=Path("benchmarks/norfair/src/evaluate_mot.py"),
        config_path=Path("benchmarks/norfair/configs/tracking/default.yaml"),
        import_name="norfair",
    ),
    "boxmot_bytetrack": TrackerSpec(
        key="boxmot_bytetrack",
        framework="boxmot",
        run_script=Path("benchmarks/boxmot/src/run_tracker.py"),
        eval_script=Path("benchmarks/boxmot/src/evaluate_mot.py"),
        config_path=Path("benchmarks/boxmot/configs/tracking/bytetrack.yaml"),
        import_name="boxmot",
    ),
    "boxmot_ocsort": TrackerSpec(
        key="boxmot_ocsort",
        framework="boxmot",
        run_script=Path("benchmarks/boxmot/src/run_tracker.py"),
        eval_script=Path("benchmarks/boxmot/src/evaluate_mot.py"),
        config_path=Path("benchmarks/boxmot/configs/tracking/ocsort.yaml"),
        import_name="boxmot",
    ),
    "boxmot_strongsort": TrackerSpec(
        key="boxmot_strongsort",
        framework="boxmot",
        run_script=Path("benchmarks/boxmot/src/run_tracker.py"),
        eval_script=Path("benchmarks/boxmot/src/evaluate_mot.py"),
        config_path=Path("benchmarks/boxmot/configs/tracking/strongsort.yaml"),
        import_name="boxmot",
        requires_frames=True,
        reid_weights=Path("reports/runs/tracker_suite/_weights/osnet_x0_25_msmt17.pt"),
    ),
    "boxmot_botsort": TrackerSpec(
        key="boxmot_botsort",
        framework="boxmot",
        run_script=Path("benchmarks/boxmot/src/run_tracker.py"),
        eval_script=Path("benchmarks/boxmot/src/evaluate_mot.py"),
        config_path=Path("benchmarks/boxmot/configs/tracking/botsort.yaml"),
        import_name="boxmot",
        requires_frames=True,
        reid_weights=Path("reports/runs/tracker_suite/_weights/osnet_x0_25_msmt17.pt"),
    ),
    "boxmot_deepocsort": TrackerSpec(
        key="boxmot_deepocsort",
        framework="boxmot",
        run_script=Path("benchmarks/boxmot/src/run_tracker.py"),
        eval_script=Path("benchmarks/boxmot/src/evaluate_mot.py"),
        config_path=Path("benchmarks/boxmot/configs/tracking/deepocsort.yaml"),
        import_name="boxmot",
        requires_frames=True,
        reid_weights=Path("reports/runs/tracker_suite/_weights/osnet_x0_25_msmt17.pt"),
    ),
    "boxmot_boosttrack": TrackerSpec(
        key="boxmot_boosttrack",
        framework="boxmot",
        run_script=Path("benchmarks/boxmot/src/run_tracker.py"),
        eval_script=Path("benchmarks/boxmot/src/evaluate_mot.py"),
        config_path=Path("benchmarks/boxmot/configs/tracking/boosttrack.yaml"),
        import_name="boxmot",
        requires_frames=True,
        reid_weights=Path("reports/runs/tracker_suite/_weights/osnet_x0_25_msmt17.pt"),
    ),
    "deepsort": TrackerSpec(
        key="deepsort",
        framework="deepsort",
        run_script=Path("benchmarks/deepsort/src/run_tracker.py"),
        eval_script=Path("benchmarks/deepsort/src/evaluate_mot.py"),
        config_path=Path("benchmarks/deepsort/configs/tracking/default.yaml"),
        import_name="tensorflow",
        requires_frames=True,
        model_path=Path("reports/runs/tracker_suite/_weights/mars-small128.pb"),
    ),
}


def resolve_path(path: Path) -> Path:
    """Resolve repo-relative CLI paths from the 2d-tracking root."""

    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def module_available(name: str) -> bool:
    """Return whether one import is available in the active interpreter."""

    return importlib.util.find_spec(name) is not None


def find_gt_for_detection(detection_json: Path, gt_dir: Path) -> Optional[Path]:
    """Find the GT MOT file whose stem prefixes the detection filename stem."""

    detection_stem = detection_json.stem
    matches = [
        gt_path
        for gt_path in gt_dir.glob("*.txt")
        if detection_stem.startswith(gt_path.stem)
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: len(path.stem))


def detector_name_for(detection_json: Path, gt_path: Path) -> str:
    """Extract the detector/model suffix from one detection export filename."""

    suffix = detection_json.stem[len(gt_path.stem):]
    return suffix.removeprefix("_").removesuffix("_detections") or "unknown_detector"


def output_name_for_gt_annotation(annotation_json: Path) -> str:
    """Map one GT annotation JSON to the shared MOT naming convention."""

    scenario_name = annotation_json.parent.name
    camera_name = CAMERA_SUFFIXES.get(annotation_json.stem, annotation_json.stem.removesuffix("_ann"))
    return f"{scenario_name}_{camera_name}"


def normalize_gt_annotation(annotation_json: Path, output_json: Path) -> None:
    """Convert GT annotations into detector-style JSON without identity leakage."""

    records = json.loads(annotation_json.read_text(encoding="utf-8"))
    normalized = []

    for record in records:
        labels = []
        for label in record.get("Labels", []):
            if not isinstance(label, dict):
                continue
            bbox = label.get("BoundingBoxes") or label.get("bbox") or label.get("box")
            if bbox is None or len(bbox) != 4:
                continue
            labels.append({
                "Class": "person",
                "BoundingBoxes": [float(value) for value in bbox],
                "Confidence": 1.0,
            })

        normalized.append({
            "File": record.get("File"),
            "Labels": labels,
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def infer_frames_dir(sequence_name: str, frames_root: Path) -> Optional[Path]:
    """Infer the frame directory for one sequence name under the dataset root."""

    camera_suffixes = {
        "_zed_rgb": Path("sensor_data/cam_zed_rgb"),
        "_front_fisheye": Path("sensor_data/cam_fish_front"),
    }

    for suffix, relative_camera_dir in camera_suffixes.items():
        if sequence_name.endswith(suffix):
            scenario_name = sequence_name.removesuffix(suffix)
            frames_dir = frames_root / scenario_name / relative_camera_dir
            if frames_dir.is_dir():
                return frames_dir
    return None


def iter_tracking_inputs(
    sources: List[str],
    detections_dir: Path,
    gt_annotations_dir: Path,
    gt_dir: Path,
    generated_inputs_dir: Path,
    includes: Optional[set[str]],
) -> Iterable[InputRecord]:
    """Yield tracking-ready inputs for the selected source modes."""

    if "detections" in sources:
        for detection_json in sorted(detections_dir.glob("*_detections.json")):
            if includes and detection_json.name not in includes:
                continue
            gt_mot = find_gt_for_detection(detection_json, gt_dir)
            if gt_mot is None:
                continue
            detector_name = detector_name_for(detection_json, gt_mot)
            yield InputRecord(
                input_json=detection_json,
                gt_mot=gt_mot,
                input_kind="detections",
                input_name=detector_name,
                run_name=detection_json.stem,
                source_path=detection_json,
            )

    if "gt" in sources:
        for annotation_json in sorted(gt_annotations_dir.glob("*/*_ann.json")):
            output_name = output_name_for_gt_annotation(annotation_json)
            generated_json = generated_inputs_dir / f"{output_name}_gt_as_detections.json"
            if includes and generated_json.name not in includes and annotation_json.name not in includes:
                continue
            gt_mot = gt_dir / f"{output_name}.txt"
            if not gt_mot.exists():
                continue
            normalize_gt_annotation(annotation_json, generated_json)
            yield InputRecord(
                input_json=generated_json,
                gt_mot=gt_mot,
                input_kind="gt",
                input_name="gt_as_detections",
                run_name=generated_json.stem,
                source_path=annotation_json,
            )


def run_command(command: List[str]) -> subprocess.CompletedProcess[str]:
    """Run one child command from the 2d-tracking root."""

    return subprocess.run(
        command,
        cwd=TRACKING_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def build_tracking_command(
    spec: TrackerSpec,
    detection_json: Path,
    output_mot: Path,
    runtime_json: Path,
    frames_dir: Optional[Path],
) -> List[str]:
    """Build the CLI invocation for one tracker wrapper."""

    command = [
        sys.executable,
        str(spec.run_script),
        "--config",
        str(spec.config_path),
        "--detections-json",
        str(detection_json),
        "--out-mot",
        str(output_mot),
        "--runtime-json",
        str(runtime_json),
    ]
    if frames_dir is not None:
        command.extend(["--frames-dir", str(frames_dir)])
    if spec.reid_weights is not None:
        command.extend(["--reid-weights", str(spec.reid_weights)])
    if spec.model_path is not None:
        command.extend(["--model-path", str(spec.model_path)])
    return command


def build_evaluation_command(
    spec: TrackerSpec,
    gt_mot: Path,
    pred_mot: Path,
    output_csv: Path,
    output_json: Path,
) -> List[str]:
    """Build the benchmark-specific evaluation invocation."""

    return [
        sys.executable,
        str(spec.eval_script),
        "--gt",
        str(gt_mot),
        "--pred",
        str(pred_mot),
        "--out-csv",
        str(output_csv),
        "--out-json",
        str(output_json),
    ]


def load_summary_json(path: Path) -> Dict[str, object]:
    """Load one flattened MOT summary JSON row."""

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return {}
    row = dict(rows[0])
    if "index" in row:
        row["sequence_label"] = row.pop("index")
    return row


def load_runtime_json(path: Path) -> Dict[str, object]:
    """Load one tracker runtime summary JSON row."""

    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    """Define the batch suite CLI."""

    parser = argparse.ArgumentParser(
        description="Run supported trackers on detector exports and/or GT-derived inputs.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("detections", "gt"),
        default=("detections", "gt"),
        help="Which tracking input sources to run.",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=Path("reports/runs/detections"),
        help="Directory containing detector export JSON files.",
    )
    parser.add_argument(
        "--gt-annotations-dir",
        type=Path,
        default=Path("reports/GTs"),
        help="Directory containing GT annotation JSON files.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("reports/runs/GTs_MOT"),
        help="Directory containing GT MOT txt files.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path(r"D:/AOC/datasets/aghri-val"),
        help="Dataset root used to infer per-sequence frame directories.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("reports/runs/tracker_suite"),
        help="Directory for tracker prediction artifacts.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("reports/summary/tracker_suite"),
        help="Directory for per-run metrics and aggregate summaries.",
    )
    parser.add_argument(
        "--include-inputs",
        nargs="+",
        help="Optional list of input basenames to process.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=sorted(TRACKER_SPECS),
        default=sorted(TRACKER_SPECS),
        help="Tracker suite to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose prediction MOT and metrics JSON already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the tracking benchmark suite."""

    # Resolve CLI arguments into concrete workspace paths and runtime filters.
    args = parse_args()
    detections_dir = resolve_path(args.detections_dir)
    gt_annotations_dir = resolve_path(args.gt_annotations_dir)
    gt_dir = resolve_path(args.gt_dir)
    frames_root = resolve_path(args.frames_root)
    runs_dir = resolve_path(args.runs_dir)
    summary_dir = resolve_path(args.summary_dir)
    generated_inputs_dir = runs_dir / "_inputs"
    includes = set(args.include_inputs) if args.include_inputs else None

    # Ensure the shared output locations exist before any tracker jobs start.
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    generated_inputs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "_weights").mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []

    # Discover each tracking-ready input, whether it came from detector exports
    # or from GT annotations normalized into detector-style JSON.
    for input_record in iter_tracking_inputs(
        list(args.sources),
        detections_dir,
        gt_annotations_dir,
        gt_dir,
        generated_inputs_dir,
        includes,
    ):
        # Each input gets its own run directory for predictions and metrics.
        sequence_dir = runs_dir / input_record.run_name
        sequence_summary_dir = summary_dir / input_record.run_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        sequence_summary_dir.mkdir(parents=True, exist_ok=True)

        # Run every requested tracker against the current input independently.
        for tracker_key in args.trackers:
            spec = TRACKER_SPECS[tracker_key]
            pred_mot = sequence_dir / f"{tracker_key}_tracks.txt"
            runtime_json = sequence_summary_dir / f"{tracker_key}_runtime.json"
            metrics_csv = sequence_summary_dir / f"{tracker_key}_metrics.csv"
            metrics_json = sequence_summary_dir / f"{tracker_key}_metrics.json"
            frames_dir = infer_frames_dir(input_record.gt_mot.stem, frames_root)

            # This base record captures the run metadata whether the run
            # completes, fails, or is skipped.
            base_record: Dict[str, object] = {
                "framework": spec.framework,
                "tracker": tracker_key,
                "status": "pending",
                "input_kind": input_record.input_kind,
                "input_name": input_record.input_name,
                "input_json": str(input_record.input_json),
                "source_path": str(input_record.source_path),
                "gt_mot": str(input_record.gt_mot),
                "frames_dir": str(frames_dir) if frames_dir is not None else "",
                "model_path": str(spec.model_path) if spec.model_path is not None else "",
                "pred_mot": str(pred_mot),
                "runtime_json": str(runtime_json),
                "metrics_csv": str(metrics_csv),
                "metrics_json": str(metrics_json),
                "sequence_name": input_record.gt_mot.stem,
            }

            # Skip cleanly if the tracker backend is not installed in the
            # active Python environment.
            if not module_available(spec.import_name):
                results.append(
                    {
                        **base_record,
                        "status": "skipped_missing_dependency",
                        "error": f"Python package {spec.import_name!r} is not importable.",
                    }
                )
                continue

            if spec.requires_frames and frames_dir is None:
                results.append(
                    {
                        **base_record,
                        "status": "skipped_missing_frames",
                        "error": f"No frame directory could be inferred for {input_record.gt_mot.stem!r}.",
                    }
                )
                continue

            if spec.model_path is not None and not resolve_path(spec.model_path).is_file():
                results.append(
                    {
                        **base_record,
                        "status": "skipped_missing_model",
                        "error": f"Required model file not found: {resolve_path(spec.model_path)}",
                    }
                )
                continue

            # Reuse cached outputs when requested so repeated suite runs are fast.
            if args.skip_existing and pred_mot.exists() and metrics_json.exists() and runtime_json.exists():
                metrics_record = load_summary_json(metrics_json) if metrics_json.exists() else {}
                runtime_record = load_runtime_json(runtime_json)
                results.append({**base_record, **runtime_record, **metrics_record, "status": "completed_cached"})
                print(f"Skipping existing run: {tracker_key} on {input_record.run_name}")
                continue

            # First produce the tracker prediction file in MOT format.
            track_result = run_command(build_tracking_command(spec, input_record.input_json, pred_mot, runtime_json, frames_dir))
            if track_result.returncode != 0:
                results.append(
                    {
                        **base_record,
                        "status": "tracking_failed",
                        "error": track_result.stderr.strip() or track_result.stdout.strip(),
                    }
                )
                print(f"Tracking failed: {tracker_key} on {input_record.run_name}")
                continue

            if not runtime_json.exists():
                results.append(
                    {
                        **base_record,
                        "status": "tracking_failed",
                        "error": f"Tracker run completed but runtime summary was not created: {runtime_json}",
                    }
                )
                print(f"Tracking failed: {tracker_key} on {input_record.run_name}")
                continue

            runtime_record = load_runtime_json(runtime_json)

            # Then evaluate that prediction against the matching GT MOT file.
            eval_result = run_command(build_evaluation_command(spec, input_record.gt_mot, pred_mot, metrics_csv, metrics_json))
            if eval_result.returncode != 0:
                results.append(
                    {
                        **base_record,
                        **runtime_record,
                        "status": "evaluation_failed",
                        "error": eval_result.stderr.strip() or eval_result.stdout.strip(),
                    }
                )
                print(f"Evaluation failed: {tracker_key} on {input_record.run_name}")
                continue

            # Fold the computed metrics back into the aggregate suite table.
            metrics_record = load_summary_json(metrics_json)
            results.append({**base_record, **runtime_record, **metrics_record, "status": "completed"})
            print(f"Completed: {tracker_key} on {input_record.run_name}")

    # Persist one aggregate summary in JSON and CSV for downstream analysis.
    summary_json_path = summary_dir / "tracking_suite_summary.json"
    summary_csv_path = summary_dir / "tracking_suite_summary.csv"
    summary_json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Build a union of keys so the CSV can hold both success and failure rows.
    fieldnames: List[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print a compact run summary for terminal use.
    completed = sum(1 for row in results if str(row.get("status", "")).startswith("completed"))
    failed = sum(1 for row in results if str(row.get("status", "")).endswith("failed"))
    skipped = sum(1 for row in results if str(row.get("status", "")).startswith("skipped"))
    print(f"Tracking suite finished. Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
    print(f"Aggregate summary CSV: {summary_csv_path}")
    print(f"Aggregate summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
