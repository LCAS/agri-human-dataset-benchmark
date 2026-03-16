import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

ROOT_2D = Path(__file__).resolve().parents[3]
BENCH_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_project_path(project_value: str) -> str:
    project_path = Path(project_value)
    if project_path.is_absolute():
        return str(project_path)
    return str((ROOT_2D / project_path).resolve())


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


def train_and_eval(model_name: str, cfg: dict) -> dict:
    model = YOLO(model_name)
    start = time.time()
    # train_results = model.train(
    #     data=cfg["dataset"],
    #     imgsz=cfg["imgsz"],
    #     epochs=cfg["epochs"],
    #     batch=cfg["batch"],
    #     device=cfg.get("device"),
    #     seed=cfg["seed"],
    #     project=cfg["project"],
    #     name=f'{cfg["name"]}/{Path(model_name).stem}',
    #     verbose=True,
    # )
    train_time = time.time() - start

    eval_split = cfg.get("eval_split", "val")
    rect = cfg.get("rect", False)
    half = cfg.get("half", False)
    dnn = cfg.get("dnn", False)
    workers = cfg.get("workers", 8)

    eval_results = model.val(
        data=cfg["dataset"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg.get("device"),
        split=eval_split,
        project=cfg["project"],
        name=f'{cfg["name"]}/{Path(model_name).stem}/{eval_split}',
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
    num_classes = len(names) if names else None
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

    eval_imgsz = _get_eval_arg(eval_results, "imgsz", cfg["imgsz"])
    eval_batch = _get_eval_arg(eval_results, "batch", cfg["batch"])
    eval_rect = _get_eval_arg(eval_results, "rect", rect)
    eval_half = _get_eval_arg(eval_results, "half", half)
    eval_workers = _get_eval_arg(eval_results, "workers", workers)
    eval_device = _get_eval_arg(eval_results, "device", cfg.get("device"))
    eval_dnn = _get_eval_arg(eval_results, "dnn", dnn)

    metrics = {
        "model": model_name,
        "train_time_sec": round(train_time, 2),
        "inf_time_per_frame_ms": None if inf_time is None else round(inf_time, 3),
        "preprocess_time_per_frame_ms": None if pre_time is None else round(pre_time, 3),
        "postprocess_time_per_frame_ms": None if post_time is None else round(post_time, 3),
        "total_time_per_frame_ms": None if total_time is None else round(total_time, 3),
        "eval_split": eval_split,
        "num_classes": num_classes,
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
        "eval_imgsz": eval_imgsz,
        "eval_batch": eval_batch,
        "eval_rect": eval_rect,
        "eval_half": eval_half,
        "eval_workers": eval_workers,
        "eval_device": eval_device,
        "eval_dnn": eval_dnn,
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "dataset": cfg["dataset"],
        "config_device": cfg.get("device"),
    }

    if cfg.get("benchmark_fixed_input", False):
        bench = benchmark_fixed_input(
            model,
            cfg["imgsz"],
            cfg.get("benchmark_iters", 200),
            cfg.get("benchmark_warmup", 20),
        )
        metrics.update(bench)

    # metrics["train_results_dir"] = str(train_results.save_dir)
    metrics["eval_results_dir"] = str(eval_results.save_dir)
    return metrics


def main() -> None:
    default_config = BENCH_ROOT / "configs" / "benchmark_ultralytics_coco128.yaml"
    default_out = ROOT_2D / "reports" / "benchmarks" / "summary" / "ultralytics" / "summary.csv"

    parser = argparse.ArgumentParser(description="Run YOLOv8 vs YOLOv11 benchmark.")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to benchmark config.",
    )
    parser.add_argument(
        "--out",
        default=str(default_out),
        help="CSV output path.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    cfg["project"] = _resolve_project_path(
        cfg.get("project", "reports/benchmarks/ultralytics/runs")
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in cfg["models"]:
        rows.append(train_and_eval(model_name, cfg))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
