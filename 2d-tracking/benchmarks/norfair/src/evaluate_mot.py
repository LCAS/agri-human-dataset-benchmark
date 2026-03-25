"""
Evaluate MOT predictions against ground truth and optionally write summary files.
Relative paths in configs and CLI arguments resolve from the 2d-tracking root.

Examples:
    python src/evaluate_mot.py --config configs/evaluation/default.yaml
    python src/evaluate_mot.py --gt reports/runs/example/ground_truth.txt --pred reports/runs/example/tracks.txt --out-csv reports/summary/example_metrics.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, List, Optional

import motmetrics as mm
import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "evaluation" / "default.yaml"
DEFAULT_METRICS = [
    "num_frames",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "mota",
    "motp",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "mostly_lost",
]


@dataclass(frozen=True)
class EvaluationConfig:
    gt_mot: Path
    pred_mot: Path
    iou_threshold: float = 0.5
    output_json: Optional[Path] = None
    output_csv: Optional[Path] = None
    metrics: List[str] = field(default_factory=lambda: list(DEFAULT_METRICS))


def _resolve_path(value: Optional[Any]) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def load_config(path: Path) -> EvaluationConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    gt_mot = raw.get("gt_mot")
    pred_mot = raw.get("pred_mot")
    if not gt_mot or not pred_mot:
        raise ValueError("`gt_mot` and `pred_mot` are required in the evaluation config.")

    metrics = raw.get("metrics", DEFAULT_METRICS)
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("`metrics` must be a non-empty list when provided.")

    return EvaluationConfig(
        gt_mot=_resolve_path(gt_mot),
        pred_mot=_resolve_path(pred_mot),
        iou_threshold=float(raw.get("iou_threshold", 0.5)),
        output_json=_resolve_path(raw.get("output_json")),
        output_csv=_resolve_path(raw.get("output_csv")),
        metrics=[str(metric) for metric in metrics],
    )


def evaluate_mot(cfg: EvaluationConfig):
    # `motmetrics` reads both files into MOTChallenge tables, then performs frame
    # matching with IoU distance before computing the requested summary metrics.
    gt = mm.io.loadtxt(str(cfg.gt_mot), fmt="mot15-2D", min_confidence=1)
    pred = mm.io.loadtxt(str(cfg.pred_mot), fmt="mot15-2D", min_confidence=1)

    acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=cfg.iou_threshold)
    metrics_handler = mm.metrics.create()
    summary = metrics_handler.compute(acc, metrics=cfg.metrics, name="sequence")

    print(
        mm.io.render_summary(
            summary,
            formatters=metrics_handler.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
    )

    return summary


def write_outputs(summary, cfg: EvaluationConfig) -> None:
    if cfg.output_csv is not None:
        cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(cfg.output_csv)

    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        # Flatten the summary DataFrame for easier downstream inspection.
        rows = summary.reset_index().to_dict(orient="records")
        cfg.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MOT prediction file against ground truth.",
        epilog=(
            "Examples:\n"
            "  python src/evaluate_mot.py --config configs/evaluation/default.yaml\n"
            "  python src/evaluate_mot.py --gt reports/runs/example/ground_truth.txt "
            "--pred reports/runs/example/tracks.txt --out-csv reports/summary/example_metrics.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to an evaluation YAML config. Relative data and report paths resolve from 2d-tracking.",
    )
    parser.add_argument("--gt", type=Path, help="Ground-truth MOT txt file.")
    parser.add_argument("--pred", type=Path, help="Prediction MOT txt file.")
    parser.add_argument("--iou-threshold", type=float, help="IoU distance threshold.")
    parser.add_argument("--out-json", type=Path, help="Optional JSON summary path.")
    parser.add_argument("--out-csv", type=Path, help="Optional CSV summary path.")
    return parser.parse_args()


def merge_cli_overrides(cfg: EvaluationConfig, args: argparse.Namespace) -> EvaluationConfig:
    # Keep YAML as the baseline and use CLI overrides for ad hoc comparisons.
    updates = {}
    if args.gt is not None:
        updates["gt_mot"] = _resolve_path(args.gt)
    if args.pred is not None:
        updates["pred_mot"] = _resolve_path(args.pred)
    if args.iou_threshold is not None:
        updates["iou_threshold"] = float(args.iou_threshold)
    if args.out_json is not None:
        updates["output_json"] = _resolve_path(args.out_json)
    if args.out_csv is not None:
        updates["output_csv"] = _resolve_path(args.out_csv)
    return replace(cfg, **updates)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    summary = evaluate_mot(cfg)
    write_outputs(summary, cfg)


if __name__ == "__main__":
    main()
