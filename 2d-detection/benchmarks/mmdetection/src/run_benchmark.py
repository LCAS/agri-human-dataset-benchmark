import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import yaml

ROOT_2D = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
print(f"Using ROOT_2D: {ROOT_2D}")
print(f"Using REPO_ROOT: {REPO_ROOT}")
print(Path(__file__).resolve().parents)

SUMMARY_FIELDS = [
    "model",
    "train_time_sec",
    "inf_time_per_frame_ms",
    "preprocess_time_per_frame_ms",
    "postprocess_time_per_frame_ms",
    "total_time_per_frame_ms",
    "eval_split",
    "num_classes",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "f1",
    "per_class_ap50_95",
    "per_class_ap50",
    "per_class_precision",
    "per_class_recall",
    "per_class_f1",
    "eval_imgsz",
    "eval_batch",
    "eval_rect",
    "eval_half",
    "eval_workers",
    "eval_device",
    "eval_dnn",
    "epochs",
    "imgsz",
    "batch",
    "dataset",
    "config_device",
    "fixed_input_inference_ms",
    "fixed_input_shape",
    "fixed_input_dtype",
    "fixed_input_device",
    "fixed_input_iters",
    "fixed_input_warmup",
    "train_results_dir",
    "eval_results_dir",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_cuda_device(device_cfg: Any) -> None:
    if device_cfg is None:
        return
    if isinstance(device_cfg, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_cfg)
        return
    if isinstance(device_cfg, str):
        value = device_cfg.strip().lower()
        if value == "cpu":
            return
        if value.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = value.split(":", 1)[1]
            return
        if value.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = value
            return
    if isinstance(device_cfg, (list, tuple)):
        values = [str(x) for x in device_cfg]
        if values:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(values)


def _device_to_mmdet_str(device_cfg: Any) -> str:
    try:
        import torch
    except Exception:
        torch = None

    cuda_available = bool(torch and torch.cuda.is_available())
    if device_cfg is None:
        return "cuda:0" if cuda_available else "cpu"
    if isinstance(device_cfg, int):
        return "cuda:0" if cuda_available else "cpu"
    if isinstance(device_cfg, str):
        value = device_cfg.strip().lower()
        if value == "cpu":
            return "cpu"
        if value.startswith("cuda"):
            return value if cuda_available else "cpu"
        if value.isdigit():
            return "cuda:0" if cuda_available else "cpu"
    if isinstance(device_cfg, (list, tuple)):
        return "cuda:0" if cuda_available else "cpu"
    return "cuda:0" if cuda_available else "cpu"


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _is_remote_ref(path_str: str) -> bool:
    parsed = urlparse(str(path_str))
    return parsed.scheme in {"http", "https"}


def _resolve_checkpoint_ref(path_str: str, repo_root: Path) -> str:
    if _is_remote_ref(path_str):
        return str(path_str)
    return str(_resolve_path(path_str, repo_root))


def _unwrap_dataset(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    current = dataset_cfg
    while isinstance(current, dict) and "dataset" in current:
        current = current["dataset"]
    return current


def _normalize_hw(scale_value: Any) -> Optional[Tuple[int, int]]:
    if scale_value is None:
        return None
    if isinstance(scale_value, (int, float)):
        size = int(scale_value)
        return (size, size)
    if isinstance(scale_value, (list, tuple)) and len(scale_value) >= 2:
        width = int(scale_value[0])
        height = int(scale_value[1])
        return (height, width)
    return None


def _extract_resize_hw(dataset_cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    pipeline = dataset_cfg.get("pipeline", [])
    for transform in pipeline:
        if not isinstance(transform, dict):
            continue
        if transform.get("type") != "Resize":
            continue
        scale = transform.get("scale", transform.get("img_scale"))
        hw = _normalize_hw(scale)
        if hw is not None:
            return hw
    return None


def _format_size(hw: Optional[Tuple[int, int]]) -> Optional[Any]:
    if hw is None:
        return None
    height, width = hw
    if height == width:
        return int(height)
    return f"{width}x{height}"


def _infer_split_name(ann_file: Optional[str]) -> Optional[str]:
    if not ann_file:
        return None
    name = ann_file.lower()
    for split in ("test", "val", "train"):
        if split in name:
            return split
    return "test"


def _join_dataset_path(dataset_cfg: Dict[str, Any]) -> Optional[str]:
    ann_file = dataset_cfg.get("ann_file")
    if ann_file is None:
        return None
    ann_path = Path(str(ann_file))
    if ann_path.is_absolute():
        return str(ann_path)
    data_root = dataset_cfg.get("data_root")
    if data_root:
        return str((Path(str(data_root)) / ann_path).as_posix())
    return str(ann_path.as_posix())


def _find_metric(metrics: Dict[str, Any], key_base: str) -> Optional[float]:
    if key_base in metrics:
        return _to_float(metrics[key_base])
    for key, value in metrics.items():
        if key.endswith("/" + key_base):
            return _to_float(value)
    return None


def _extract_per_class_ap50_95(metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
    # MMDetection's CocoMetric stores classwise AP(0.5:0.95) in keys ending
    # with "_precision" when classwise=True.
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        key_name = key.split("/")[-1]
        if key_name.endswith("_precision"):
            class_name = key_name[: -len("_precision")]
            casted = _to_float(value)
            if casted is not None:
                out[class_name] = casted
    return out or None


def _enable_classwise(cfg: Any) -> None:
    evaluator = cfg.get("test_evaluator")
    if evaluator is None:
        return
    if isinstance(evaluator, list):
        for item in evaluator:
            if isinstance(item, dict) and item.get("type") == "CocoMetric":
                item["classwise"] = True
    elif isinstance(evaluator, dict):
        if evaluator.get("type") == "CocoMetric":
            evaluator["classwise"] = True


def _latest_checkpoint(work_dir: Path) -> Optional[Path]:
    latest = work_dir / "latest.pth"
    if latest.exists():
        return latest
    candidates = sorted(work_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _best_checkpoint(work_dir: Path, metric_name: Optional[str]) -> Optional[Path]:
    candidates = sorted(
        work_dir.glob("best*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    if metric_name:
        metric_token = metric_name.replace("/", "_")
        preferred = [p for p in candidates if metric_token in p.name]
        if preferred:
            return preferred[0]
    return candidates[0]


def _select_eval_checkpoint(
    work_dir: Path,
    explicit_checkpoint: Optional[Union[str, Path]],
    prefer_best: bool,
    best_metric_name: Optional[str],
) -> Optional[Union[str, Path]]:
    if explicit_checkpoint is not None:
        return explicit_checkpoint
    if prefer_best:
        best_ckpt = _best_checkpoint(work_dir, best_metric_name)
        if best_ckpt is not None:
            return best_ckpt
    return _latest_checkpoint(work_dir)


def _epoch_from_checkpoint_name(checkpoint: Union[str, Path]) -> Optional[int]:
    checkpoint_str = str(checkpoint)
    if _is_remote_ref(checkpoint_str):
        name = Path(urlparse(checkpoint_str).path).stem
    else:
        name = Path(checkpoint_str).stem
    if "_epoch_" in name:
        tail = name.split("_epoch_", 1)[1]
        if tail.isdigit():
            return int(tail)
    if name.startswith("epoch_"):
        tail = name.split("epoch_", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def _configure_checkpoint_hook(cfg: Any, bench_cfg: Dict[str, Any]) -> None:
    if "default_hooks" not in cfg or cfg.default_hooks is None:
        cfg.default_hooks = {}
    checkpoint_hook = cfg.default_hooks.get("checkpoint", {})
    if checkpoint_hook is None:
        checkpoint_hook = {}

    save_best_metric = bench_cfg.get("save_best_metric")
    if save_best_metric:
        checkpoint_hook["save_best"] = save_best_metric
        checkpoint_hook["rule"] = bench_cfg.get("save_best_rule", "greater")
    if bench_cfg.get("max_keep_ckpts") is not None:
        checkpoint_hook["max_keep_ckpts"] = int(bench_cfg["max_keep_ckpts"])

    cfg.default_hooks["checkpoint"] = checkpoint_hook


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_csvable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return value


def benchmark_fixed_input(
    cfg_path: Path,
    checkpoint_path: Optional[Union[str, Path]],
    device: str,
    shape_hw: Optional[Tuple[int, int]],
    iters: int,
    warmup: int,
) -> Dict[str, Any]:
    if checkpoint_path is None or shape_hw is None:
        return {"fixed_input_inference_ms": None}

    try:
        import numpy as np
        import torch
        from mmdet.apis import inference_detector, init_detector
    except Exception:
        return {"fixed_input_inference_ms": None}

    iters = max(int(iters), 0)
    warmup = max(int(warmup), 0)
    if iters == 0:
        return {"fixed_input_inference_ms": None}

    model = init_detector(str(cfg_path), str(checkpoint_path), device=device)
    height, width = shape_hw
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for _ in range(warmup):
        _ = inference_detector(model, image)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = inference_detector(model, image)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed_ms = (time.time() - start) * 1000.0 / iters
    first_param = next(model.parameters())
    return {
        "fixed_input_inference_ms": round(elapsed_ms, 3),
        "fixed_input_shape": f"{height}x{width}",
        "fixed_input_dtype": str(first_param.dtype),
        "fixed_input_device": str(first_param.device),
        "fixed_input_iters": iters,
        "fixed_input_warmup": warmup,
    }


def train_and_eval_model(
    model_item: Dict[str, Any],
    bench_cfg: Dict[str, Any],
    repo_root: Path,
) -> Dict[str, Any]:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import RUNNERS

    cfg_path = _resolve_path(model_item["config"], repo_root)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")

    model_name = model_item.get("name", cfg_path.stem)
    cfg = Config.fromfile(str(cfg_path))
    cfg.launcher = "none"

    if bench_cfg.get("cfg_options"):
        cfg.merge_from_dict(bench_cfg["cfg_options"])
    if model_item.get("cfg_options"):
        cfg.merge_from_dict(model_item["cfg_options"])

    seed = model_item.get("seed", bench_cfg.get("seed"))
    if seed is not None:
        cfg.randomness = dict(seed=int(seed))

    default_work_dir = (
        Path(
            bench_cfg.get(
                "project", "2d-detection/reports/benchmarks/mmdetection/runs"
            )
        )
        / bench_cfg.get("name", "mmdet-benchmark")
        / model_name
    )
    work_dir = _resolve_path(model_item.get("work_dir", str(default_work_dir)), repo_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)

    _enable_classwise(cfg)
    _configure_checkpoint_hook(cfg, bench_cfg)

    train_enabled = bool(model_item.get("train", True))
    # `checkpoint` is an optional benchmark-YAML override. If it is absent,
    # eval-only runs can still use the model config's `load_from`.
    checkpoint_path = model_item.get("checkpoint")
    if checkpoint_path:
        checkpoint = _resolve_checkpoint_ref(checkpoint_path, repo_root)
        cfg.load_from = checkpoint
    else:
        checkpoint = None

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    train_time_sec = None
    if train_enabled:
        start_train = time.time()
        runner.train()
        train_time_sec = round(time.time() - start_train, 2)
    elif checkpoint is None and cfg.get("load_from") is None:
        raise ValueError(
            f"Model '{model_name}' has train=false but no checkpoint/load_from."
        )

    prefer_best = bool(bench_cfg.get("prefer_best_checkpoint", True))
    best_metric_name = bench_cfg.get("save_best_metric")

    checkpoint_for_eval = checkpoint if not train_enabled else None
    # For eval-only runs without a YAML-level checkpoint override, fall back to
    # the checkpoint declared in the MMDetection config file.
    if checkpoint_for_eval is None and not train_enabled and cfg.get("load_from") is not None:
        checkpoint_for_eval = str(cfg.load_from)
    checkpoint = _select_eval_checkpoint(
        work_dir=work_dir,
        explicit_checkpoint=checkpoint_for_eval,
        prefer_best=prefer_best,
        best_metric_name=best_metric_name,
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found in {work_dir}. Cannot run evaluation."
        )
    eval_epoch = _epoch_from_checkpoint_name(checkpoint)
    if eval_epoch is None:
        print(f"[{model_name}] Evaluating checkpoint: {checkpoint}")
    else:
        print(f"[{model_name}] Evaluating checkpoint (epoch {eval_epoch}): {checkpoint}")

    cfg.load_from = str(checkpoint)
    cfg.resume = False
    if "runner_type" not in cfg:
        eval_runner = Runner.from_cfg(cfg)
    else:
        eval_runner = RUNNERS.build(cfg)
    test_metrics = _to_jsonable(eval_runner.test() or {})

    test_dataset_cfg = _unwrap_dataset(cfg.test_dataloader["dataset"])
    train_dataset_cfg = _unwrap_dataset(cfg.train_dataloader["dataset"])

    classes = None
    metainfo = test_dataset_cfg.get("metainfo", cfg.get("metainfo"))
    if isinstance(metainfo, dict):
        classes = metainfo.get("classes")
    num_classes = len(classes) if classes else None

    eval_hw = _extract_resize_hw(test_dataset_cfg)
    train_hw = _extract_resize_hw(train_dataset_cfg)

    map50 = _find_metric(test_metrics, "bbox_mAP_50")
    map50_95 = _find_metric(test_metrics, "bbox_mAP")
    per_class_ap50_95 = _extract_per_class_ap50_95(test_metrics)

    # CocoMetric classwise output does not provide per-class AP50 directly.
    per_class_ap50 = None
    if classes and len(classes) == 1 and map50 is not None:
        per_class_ap50 = {classes[0]: map50}
    if per_class_ap50_95 is None and classes and len(classes) == 1 and map50_95 is not None:
        per_class_ap50_95 = {classes[0]: map50_95}

    device_cfg = model_item.get("device", bench_cfg.get("device"))
    bench_device = _device_to_mmdet_str(device_cfg)
    bench = {"fixed_input_inference_ms": None}
    if bench_cfg.get("benchmark_fixed_input", False):
        bench = benchmark_fixed_input(
            cfg_path=cfg_path,
            checkpoint_path=checkpoint,
            device=bench_device,
            shape_hw=eval_hw,
            iters=bench_cfg.get("benchmark_iters", 200),
            warmup=bench_cfg.get("benchmark_warmup", 20),
        )

    inf_ms = bench.get("fixed_input_inference_ms")
    summary = {
        "model": model_name,
        "train_time_sec": train_time_sec,
        "inf_time_per_frame_ms": inf_ms,
        "preprocess_time_per_frame_ms": None,
        "postprocess_time_per_frame_ms": None,
        "total_time_per_frame_ms": inf_ms,
        "eval_split": _infer_split_name(test_dataset_cfg.get("ann_file")),
        "num_classes": num_classes,
        "map50": map50,
        "map50_95": map50_95,
        "precision": None,
        "recall": None,
        "f1": None,
        "per_class_ap50_95": per_class_ap50_95,
        "per_class_ap50": per_class_ap50,
        "per_class_precision": None,
        "per_class_recall": None,
        "per_class_f1": None,
        "eval_imgsz": _format_size(eval_hw),
        "eval_batch": cfg.test_dataloader.get("batch_size"),
        "eval_rect": None,
        "eval_half": None,
        "eval_workers": cfg.test_dataloader.get("num_workers"),
        "eval_device": device_cfg,
        "eval_dnn": None,
        "epochs": cfg.get("train_cfg", {}).get("max_epochs"),
        "imgsz": _format_size(train_hw),
        "batch": cfg.train_dataloader.get("batch_size"),
        "dataset": _join_dataset_path(test_dataset_cfg),
        "config_device": bench_cfg.get("device"),
        "train_results_dir": str(work_dir),
        "eval_results_dir": str(work_dir),
    }
    summary.update(bench)

    # Keep stable column order for EDA compatibility.
    ordered = {k: summary.get(k) for k in SUMMARY_FIELDS}
    return _to_jsonable(ordered)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = {k: _to_csvable(row.get(k)) for k in SUMMARY_FIELDS}
            writer.writerow(csv_row)


def main() -> None:
    default_config = (
        ROOT_2D
        / "benchmarks"
        / "mmdetection"
        / "configs"
        / "benchmark_mmdetection_fisheye.yaml"
    )
    default_out = (
        ROOT_2D / "reports" / "benchmarks" / "summary" / "mmdetection" / "summary_mmdetection_fisheye.csv"
    )

    parser = argparse.ArgumentParser(
        description="Train/evaluate MMDetection models and export summary CSV/JSON."
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--out",
        default=str(default_out),
        help="CSV output path.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    bench_cfg = load_config(cfg_path)

    _set_cuda_device(bench_cfg.get("device"))
    sys.path.insert(0, str(REPO_ROOT))

    model_items = bench_cfg.get("models", [])
    if not model_items:
        raise ValueError("No models configured. Add at least one entry under `models`.")

    rows = []
    for model_item in model_items:
        if isinstance(model_item, str):
            model_item = {"config": model_item}
        rows.append(train_and_eval_model(model_item, bench_cfg, REPO_ROOT))

    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = (REPO_ROOT / out_csv).resolve()
    write_csv(out_csv, rows)

    out_json = out_csv.with_suffix(".json")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out_csv}")
    print(f"Wrote JSON to {out_json}")


if __name__ == "__main__":
    main()
