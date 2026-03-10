"""
Transfer evaluation: load fine-tuned YOLO checkpoints (trained on zedrgb)
and evaluate them on a different dataset (e.g. COCO val) to measure generalisation.

Usage:
    python transfer_eval.py --config configs/transfer_eval_coco.yaml
    python transfer_eval.py \
        --checkpoints \
            /workspace/.../yolov8s/weights/best.pt \
            /workspace/.../yolo11s/weights/best.pt \
        --dataset /workspace/data/coco2017-filtered-yolo/data.yaml \
        --out reports/benchmarks/ultralytics/transfer_summary.csv
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT_2D = Path(__file__).resolve().parents[3]
BENCH_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        h, w = (int(imgsz[0]), int(imgsz[1])) if len(imgsz) >= 2 else (int(imgsz[0]),) * 2
    else:
        h = w = int(imgsz)

    import torch
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


# ---------------------------------------------------------------------------
# Core: resolve best.pt from a training run dir or accept a direct path
# ---------------------------------------------------------------------------

def resolve_checkpoint(checkpoint_path: str) -> Path:
    """
    Accept either:
      - a direct path to best.pt / last.pt
      - a training run directory (will look for weights/best.pt inside)
    """
    p = Path(checkpoint_path)
    if p.is_file():
        return p
    # Maybe it's a run dir
    candidate = p / "weights" / "best.pt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot find a checkpoint at '{checkpoint_path}'. "
        "Expected either a .pt file or a directory containing weights/best.pt"
    )


# ---------------------------------------------------------------------------
# Transfer evaluation for a single checkpoint
# ---------------------------------------------------------------------------

def transfer_eval(checkpoint_path: str, cfg: dict) -> dict:
    ckpt = resolve_checkpoint(checkpoint_path)
    print(f"\n[transfer_eval] Loading checkpoint: {ckpt}")

    model = YOLO(str(ckpt))

    eval_split = cfg.get("eval_split", "val")
    rect      = cfg.get("rect", False)
    half      = cfg.get("half", False)
    dnn       = cfg.get("dnn", False)
    workers   = cfg.get("workers", 8)

    # Use a descriptive sub-folder so runs don't collide
    ckpt_tag = ckpt.parent.parent.name  # e.g. "yolov8s5"
    run_name = f'{cfg["name"]}/{ckpt_tag}/transfer_{eval_split}'

    eval_results = model.val(
        data=cfg["transfer_dataset"],   # <-- target domain dataset
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device"),
        split=eval_split,
        project=cfg["project"],
        name=run_name,
        verbose=True,
        rect=rect,
        half=half,
        dnn=dnn,
        workers=workers,
    )

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
    ap          = to_list(getattr(box, "ap",   None))
    ap50        = to_list(getattr(box, "ap50", None))
    per_class_p = to_list(getattr(box, "p",    None))
    per_class_r = to_list(getattr(box, "r",    None))
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

    speed     = getattr(eval_results, "speed", None) or {}
    pre_time  = _to_float(speed.get("preprocess"))
    inf_time  = _to_float(speed.get("inference"))
    post_time = _to_float(speed.get("postprocess"))
    total_time = (
        pre_time + inf_time + post_time
        if None not in (pre_time, inf_time, post_time)
        else None
    )

    metrics = {
        # --- identification ---
        "checkpoint": str(ckpt),
        "source_model": ckpt.parent.parent.name,   # training run folder name
        "transfer_dataset": cfg["transfer_dataset"],
        "eval_split": eval_split,
        # --- timing ---
        "inf_time_per_frame_ms":         None if inf_time  is None else round(inf_time,  3),
        "preprocess_time_per_frame_ms":  None if pre_time  is None else round(pre_time,  3),
        "postprocess_time_per_frame_ms": None if post_time is None else round(post_time, 3),
        "total_time_per_frame_ms":       None if total_time is None else round(total_time, 3),
        # --- detection metrics ---
        "num_classes": len(names) if names else None,
        "map50":     float(box.map50),
        "map50_95":  float(box.map),
        "precision": float(box.mp),
        "recall":    float(box.mr),
        "f1":        float(f1),
        "per_class_ap50_95":  by_name(ap),
        "per_class_ap50":     by_name(ap50),
        "per_class_precision":by_name(per_class_p),
        "per_class_recall":   by_name(per_class_r),
        "per_class_f1":       by_name(per_class_f1),
        # --- eval config echo ---
        "eval_imgsz":   _get_eval_arg(eval_results, "imgsz",   cfg["imgsz"]),
        "eval_batch":   _get_eval_arg(eval_results, "batch",   cfg["batch"]),
        "eval_rect":    _get_eval_arg(eval_results, "rect",    rect),
        "eval_half":    _get_eval_arg(eval_results, "half",    half),
        "eval_workers": _get_eval_arg(eval_results, "workers", workers),
        "eval_device":  _get_eval_arg(eval_results, "device",  cfg.get("device")),
        "eval_dnn":     _get_eval_arg(eval_results, "dnn",     dnn),
        "eval_results_dir": str(eval_results.save_dir),
    }

    if cfg.get("benchmark_fixed_input", False):
        bench = benchmark_fixed_input(
            model,
            cfg["imgsz"],
            cfg.get("benchmark_iters", 200),
            cfg.get("benchmark_warmup", 20),
        )
        metrics.update(bench)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    default_config = BENCH_ROOT / "configs" / "transfer_eval_coco.yaml"
    default_out    = ROOT_2D / "reports" / "benchmarks" / "ultralytics" / "transfer_summary.csv"

    parser = argparse.ArgumentParser(
        description="Transfer evaluation: run fine-tuned YOLO checkpoints on a new dataset."
    )
    parser.add_argument("--config", default=str(default_config),
                        help="Path to transfer-eval YAML config.")
    parser.add_argument("--out",    default=str(default_out),
                        help="CSV output path.")
    # Allow overriding checkpoints / dataset directly from CLI
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="One or more checkpoint paths (overrides config).")
    parser.add_argument("--dataset", default=None,
                        help="Target domain data.yaml (overrides config transfer_dataset).")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # CLI overrides
    if args.checkpoints:
        cfg["checkpoints"] = args.checkpoints
    if args.dataset:
        cfg["transfer_dataset"] = args.dataset

    # Resolve project path
    raw_project = cfg.get("project", "reports/benchmarks/ultralytics/runs")
    project_p   = Path(raw_project)
    cfg["project"] = str(project_p if project_p.is_absolute() else (ROOT_2D / project_p).resolve())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ckpt in cfg["checkpoints"]:
        rows.append(transfer_eval(ckpt, cfg))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Transfer Evaluation Summary ===")
    print(df[["source_model", "transfer_dataset", "map50", "map50_95",
              "precision", "recall", "f1"]].to_string(index=False))
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()