"""
Ultralytics benchmark runner with an explicit config schema.

Expected config shape:

mode: train_eval | transfer_eval | eval_only
train_dataset: /workspace/data/source/data.yaml   # required for train_eval
eval_dataset: /workspace/data/target/data.yaml    # required for all modes
models:
  - name: yolov8s
    source: yolov8s.pt
  - name: yolo11s
    source: yolo11s.pt
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT_2D = Path(__file__).resolve().parents[3]
BENCH_ROOT = Path(__file__).resolve().parents[1]
BenchmarkMode = Literal["train_eval", "transfer_eval", "eval_only"]


# Keep the parsed config strongly typed so the rest of the runner can work
# with one explicit shape instead of re-checking raw YAML dictionaries.
@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: BenchmarkMode
    train_dataset: Optional[str]
    eval_dataset: str
    imgsz: Any
    epochs: Optional[int]
    batch: int
    device: Any
    seed: Optional[int]
    eval_split: str
    rect: bool
    half: bool
    dnn: bool
    workers: int
    benchmark_fixed_input: bool
    benchmark_warmup: int
    benchmark_iters: int
    project: str
    name: str
    notes: Optional[str]
    models: List[ModelSpec]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Resolve report output paths relative to the 2d-detection root so configs can
# stay portable across local and cluster environments.
def _resolve_project_path(project_value: str) -> str:
    project_path = Path(project_value)
    if project_path.is_absolute():
        return str(project_path)
    return str((ROOT_2D / project_path).resolve())


def _is_remote_ref(value: str) -> bool:
    return urlparse(str(value)).scheme in {"http", "https"}


def _resolve_local_ref(value: str) -> str:
    if _is_remote_ref(value):
        return str(value)

    path = Path(value)
    if path.is_absolute():
        return str(path)

    # Try a small set of benchmark-local roots instead of guessing broadly.
    candidates = [path, BENCH_ROOT / path, ROOT_2D / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(value)


# Eval-only and transfer-eval runs may point either at a checkpoint file or at
# a prior run directory; accept both and normalize to a concrete checkpoint.
def _resolve_eval_source(source: str) -> str:
    resolved = _resolve_local_ref(source)
    if _is_remote_ref(resolved):
        return resolved

    path = Path(resolved)
    if path.is_file():
        return str(path)

    candidate = path / "weights" / "best.pt"
    if candidate.is_file():
        return str(candidate)

    raise FileNotFoundError(
        f"Cannot evaluate source '{source}'. Expected a checkpoint file or a run directory "
        "containing weights/best.pt."
    )


def _safe_tag(value: str) -> str:
    chars = []
    for char in str(value):
        chars.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(chars).strip("_") or "run"


def _dataset_tag(dataset_ref: str) -> str:
    dataset_path = Path(str(dataset_ref))
    parent_name = dataset_path.parent.name or dataset_path.stem or "dataset"
    return _safe_tag(parent_name)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_eval_arg(eval_results, name, default=None):
    args = getattr(eval_results, "args", None)
    if args is None:
        return default
    return getattr(args, name, default)


def _parse_models(raw_models: Any) -> List[ModelSpec]:
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("`models` must be a non-empty list of {name, source} objects.")

    models: List[ModelSpec] = []
    for item in raw_models:
        if not isinstance(item, dict):
            raise TypeError("Each model entry must be a mapping with `name` and `source`.")
        name = item.get("name")
        source = item.get("source")
        if not name or not source:
            raise ValueError("Each model entry must define both `name` and `source`.")
        models.append(ModelSpec(name=str(name), source=str(source)))
    return models


def parse_benchmark_config(raw_cfg: Dict[str, Any]) -> BenchmarkConfig:
    # The YAML schema is deliberately strict so the runtime logic can stay simple.
    mode = raw_cfg.get("mode")
    if mode not in {"train_eval", "transfer_eval", "eval_only"}:
        raise ValueError("`mode` must be one of: train_eval, transfer_eval, eval_only.")

    train_dataset = raw_cfg.get("train_dataset")
    eval_dataset = raw_cfg.get("eval_dataset")
    if not eval_dataset:
        raise ValueError("`eval_dataset` is required.")
    if mode == "train_eval" and not train_dataset:
        raise ValueError("`train_dataset` is required when mode=train_eval.")
    if mode != "train_eval" and train_dataset is not None:
        raise ValueError("`train_dataset` is only valid when mode=train_eval.")

    epochs = raw_cfg.get("epochs")
    if mode == "train_eval" and epochs is None:
        raise ValueError("`epochs` is required when mode=train_eval.")

    return BenchmarkConfig(
        mode=mode,
        train_dataset=train_dataset,
        eval_dataset=str(eval_dataset),
        imgsz=raw_cfg["imgsz"],
        epochs=None if epochs is None else int(epochs),
        batch=int(raw_cfg["batch"]),
        device=raw_cfg.get("device"),
        seed=None if raw_cfg.get("seed") is None else int(raw_cfg["seed"]),
        eval_split=str(raw_cfg.get("eval_split", "val")),
        rect=bool(raw_cfg.get("rect", False)),
        half=bool(raw_cfg.get("half", False)),
        dnn=bool(raw_cfg.get("dnn", False)),
        workers=int(raw_cfg.get("workers", 8)),
        benchmark_fixed_input=bool(raw_cfg.get("benchmark_fixed_input", False)),
        benchmark_warmup=int(raw_cfg.get("benchmark_warmup", 20)),
        benchmark_iters=int(raw_cfg.get("benchmark_iters", 200)),
        project=_resolve_project_path(
            raw_cfg.get("project", "reports/benchmarks/ultralytics/runs")
        ),
        name=str(raw_cfg.get("name", "agri-human-detection")),
        notes=raw_cfg.get("notes"),
        models=_parse_models(raw_cfg.get("models")),
    )


def _train_run_name(cfg: BenchmarkConfig, model: ModelSpec) -> str:
    return f"{cfg.name}/{model.name}"


def _eval_run_name(cfg: BenchmarkConfig, model: ModelSpec) -> str:
    # In-domain train/eval uses the plain split name; cross-domain and eval-only
    # runs include the target dataset tag so output folders stay distinct.
    if cfg.mode == "train_eval":
        suffix = cfg.eval_split
    else:
        suffix = f"{cfg.mode}_{_dataset_tag(cfg.eval_dataset)}_{cfg.eval_split}"
    return f"{cfg.name}/{model.name}/{suffix}"


def _trained_checkpoint(model: YOLO, train_results: Any) -> Optional[Path]:
    # Ultralytics has changed where it exposes the save directory across versions,
    # so probe the common locations and prefer best.pt when available.
    candidate_dirs = [
        getattr(train_results, "save_dir", None),
        getattr(getattr(model, "trainer", None), "save_dir", None),
    ]
    for save_dir in candidate_dirs:
        if not save_dir:
            continue
        save_dir_path = Path(save_dir)
        best = save_dir_path / "weights" / "best.pt"
        if best.is_file():
            return best
        last = save_dir_path / "weights" / "last.pt"
        if last.is_file():
            return last
    return None


def _infer_train_results_dir(source_ref: str) -> Optional[str]:
    # Only checkpoint-backed eval runs can be mapped back to an earlier train dir.
    if _is_remote_ref(source_ref):
        return None
    path = Path(source_ref)
    if path.is_file() and path.parent.name == "weights" and path.parent.parent.name:
        return str(path.parent.parent)
    return None


def benchmark_fixed_input(model: YOLO, imgsz, iters: int, warmup: int) -> dict:
    try:
        import torch
    except Exception:
        return {"fixed_input_inference_ms": None}

    iters = max(int(iters), 0)
    warmup = max(int(warmup), 0)
    if iters == 0:
        return {"fixed_input_inference_ms": None}

    model_t = model.model
    device = next(model_t.parameters()).device
    dtype = next(model_t.parameters()).dtype

    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) >= 2:
            h, w = int(imgsz[0]), int(imgsz[1])
        else:
            h = w = int(imgsz[0])
    else:
        h = w = int(imgsz)

    x = torch.zeros((1, 3, h, w), device=device, dtype=dtype)

    model_t.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model_t(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        for _ in range(iters):
            _ = model_t(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    ms = (time.time() - start) * 1000.0 / iters
    return {
        "fixed_input_inference_ms": round(ms, 3),
        "fixed_input_shape": f"{h}x{w}",
        "fixed_input_dtype": str(dtype),
        "fixed_input_device": str(device),
        "fixed_input_iters": iters,
        "fixed_input_warmup": warmup,
    }


def _evaluate_model(model: YOLO, cfg: BenchmarkConfig, model_spec: ModelSpec) -> Any:
    # Evaluation settings are shared across all benchmark modes; only the loaded
    # model artifact differs between train/eval and eval-only paths.
    return model.val(
        data=cfg.eval_dataset,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        split=cfg.eval_split,
        project=cfg.project,
        name=_eval_run_name(cfg, model_spec),
        verbose=True,
        rect=cfg.rect,
        half=cfg.half,
        dnn=cfg.dnn,
        workers=cfg.workers,
    )


def _collect_metrics(
    cfg: BenchmarkConfig,
    model_spec: ModelSpec,
    model_source: str,
    checkpoint: Optional[str],
    train_time: Optional[float],
    train_results_dir: Optional[str],
    model: YOLO,
    eval_results: Any,
) -> Dict[str, Any]:
    # Keep all result shaping in one place so the execution paths only need to
    # worry about how the model was produced.
    box = eval_results.box
    f1 = 0.0
    if (box.mp + box.mr) > 0:
        f1 = (2 * box.mp * box.mr) / (box.mp + box.mr)

    def to_list(values):
        if values is None:
            return None
        try:
            return [float(v) for v in values]
        except TypeError:
            return None

    names = eval_results.names or {}
    ap = to_list(getattr(box, "ap", None))
    ap50 = to_list(getattr(box, "ap50", None))
    per_class_p = to_list(getattr(box, "p", None))
    per_class_r = to_list(getattr(box, "r", None))
    per_class_f1 = None
    if per_class_p is not None and per_class_r is not None:
        per_class_f1 = [
            (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            for p, r in zip(per_class_p, per_class_r)
        ]

    def by_name(values):
        if values is None or not names:
            return None
        return {names[i]: values[i] for i in range(len(values)) if i in names}

    speed = getattr(eval_results, "speed", None) or {}
    pre_time = _to_float(speed.get("preprocess"))
    inf_time = _to_float(speed.get("inference"))
    post_time = _to_float(speed.get("postprocess"))
    total_time = None
    if pre_time is not None and inf_time is not None and post_time is not None:
        total_time = pre_time + inf_time + post_time

    metrics = {
        "model": model_spec.name,
        "model_source": model_source,
        "checkpoint": checkpoint,
        "run_type": cfg.mode,
        "train_enabled": cfg.mode == "train_eval",
        "train_time_sec": train_time,
        "inf_time_per_frame_ms": None if inf_time is None else round(inf_time, 3),
        "preprocess_time_per_frame_ms": None if pre_time is None else round(pre_time, 3),
        "postprocess_time_per_frame_ms": None if post_time is None else round(post_time, 3),
        "total_time_per_frame_ms": None if total_time is None else round(total_time, 3),
        "eval_split": cfg.eval_split,
        "num_classes": len(names) if names else None,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": float(f1),
        "per_class_ap50_95": by_name(ap),
        "per_class_ap50": by_name(ap50),
        "per_class_precision": by_name(per_class_p),
        "per_class_recall": by_name(per_class_r),
        "per_class_f1": by_name(per_class_f1),
        "eval_imgsz": _get_eval_arg(eval_results, "imgsz", cfg.imgsz),
        "eval_batch": _get_eval_arg(eval_results, "batch", cfg.batch),
        "eval_rect": _get_eval_arg(eval_results, "rect", cfg.rect),
        "eval_half": _get_eval_arg(eval_results, "half", cfg.half),
        "eval_workers": _get_eval_arg(eval_results, "workers", cfg.workers),
        "eval_device": _get_eval_arg(eval_results, "device", cfg.device),
        "eval_dnn": _get_eval_arg(eval_results, "dnn", cfg.dnn),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "dataset": cfg.eval_dataset,
        "train_dataset": cfg.train_dataset,
        "eval_dataset": cfg.eval_dataset,
        "config_device": cfg.device,
        "train_results_dir": train_results_dir,
        "eval_results_dir": str(eval_results.save_dir),
    }

    if cfg.benchmark_fixed_input:
        # This is a synthetic latency measurement on a fixed zero tensor; keep it
        # separate from Ultralytics' own per-batch eval timing above.
        metrics.update(
            benchmark_fixed_input(
                model,
                cfg.imgsz,
                cfg.benchmark_iters,
                cfg.benchmark_warmup,
            )
        )

    return metrics


def run_train_eval(model_spec: ModelSpec, cfg: BenchmarkConfig) -> Dict[str, Any]:
    # `source` is an architecture or pretrained initialization here. After
    # training, reload the saved checkpoint so evaluation uses the exported artifact.
    model_source = _resolve_local_ref(model_spec.source)
    model = YOLO(model_source)

    start = time.time()
    train_results = model.train(
        data=cfg.train_dataset,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        seed=cfg.seed,
        project=cfg.project,
        name=_train_run_name(cfg, model_spec),
        verbose=True,
    )
    train_time = round(time.time() - start, 2)

    trained_checkpoint = _trained_checkpoint(model, train_results)
    train_results_dir = None
    checkpoint = None
    if trained_checkpoint is not None:
        checkpoint = str(trained_checkpoint)
        train_results_dir = str(trained_checkpoint.parent.parent)
        model = YOLO(checkpoint)
    else:
        train_results_dir = getattr(train_results, "save_dir", None) or getattr(
            getattr(model, "trainer", None), "save_dir", None
        )
        if train_results_dir is not None:
            train_results_dir = str(train_results_dir)

    eval_results = _evaluate_model(model, cfg, model_spec)
    return _collect_metrics(
        cfg=cfg,
        model_spec=model_spec,
        model_source=model_source,
        checkpoint=checkpoint,
        train_time=train_time,
        train_results_dir=train_results_dir,
        model=model,
        eval_results=eval_results,
    )


def run_eval_only(model_spec: ModelSpec, cfg: BenchmarkConfig) -> Dict[str, Any]:
    # Eval-only covers both zero-shot pretrained runs and transfer evaluation
    # from previously fine-tuned checkpoints.
    model_source = _resolve_eval_source(model_spec.source)
    model = YOLO(model_source)
    eval_results = _evaluate_model(model, cfg, model_spec)
    return _collect_metrics(
        cfg=cfg,
        model_spec=model_spec,
        model_source=model_source,
        checkpoint=model_source,
        train_time=None,
        train_results_dir=_infer_train_results_dir(model_source),
        model=model,
        eval_results=eval_results,
    )


def run_benchmark(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    # The mode switch happens once here; the per-mode helpers stay linear.
    rows = []
    for model_spec in cfg.models:
        if cfg.mode == "train_eval":
            rows.append(run_train_eval(model_spec, cfg))
        else:
            rows.append(run_eval_only(model_spec, cfg))
    return rows


def main() -> None:
    default_config = BENCH_ROOT / "configs" / "benchmark_ultralytics_zedrgb.yaml"
    default_out = (
        ROOT_2D
        / "reports"
        / "benchmarks"
        / "summary"
        / "ultralytics"
        / "summary_ultralytics_zedrgb.csv"
    )

    parser = argparse.ArgumentParser(
        description="Run Ultralytics benchmarks with an explicit config schema."
    )
    parser.add_argument("--config", default=str(default_config), help="Path to benchmark config.")
    parser.add_argument("--out", default=str(default_out), help="CSV output path.")
    args = parser.parse_args()

    # Parse and validate the full YAML up front before any training/eval work starts.
    cfg = parse_benchmark_config(load_config(Path(args.config)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = run_benchmark(cfg)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    out_path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary_columns = [
        column
        for column in ["model", "run_type", "dataset", "map50", "map50_95", "precision", "recall", "f1"]
        if column in df.columns
    ]
    print(df[summary_columns].to_string(index=False))


if __name__ == "__main__":
    main()
