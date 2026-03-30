"""
Benchmark-local wrapper for the shared MOT evaluation utility.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3]
COMMON_ROOT = TRACKING_ROOT / "common"
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "evaluation" / "default.yaml"

if str(COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMMON_ROOT))

from mot.evaluate_mot import EvaluationConfig, evaluate_mot, load_config, merge_cli_overrides, parse_args, resolve_path, write_outputs

__all__ = (
    "EvaluationConfig",
    "evaluate_mot",
    "load_config",
    "merge_cli_overrides",
    "parse_args",
    "resolve_path",
    "write_outputs",
)


def main() -> None:
    args = parse_args(DEFAULT_CONFIG)
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    summary = evaluate_mot(cfg)
    write_outputs(summary, cfg)


if __name__ == "__main__":
    main()
