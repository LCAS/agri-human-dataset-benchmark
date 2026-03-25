"""
Dump YOLO detections to JSON files for 2D tracking pipeline.
Each model in the config produces one JSON file consumable by norfair_tracker.py.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from ultralytics import YOLO


# ==============================
# CONFIG
# ==============================
@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str


@dataclass(frozen=True)
class PredictConfig:
    name: str
    images_dir: Path
    output_dir: Path
    confidence: float
    target_class: int
    image_ext: str
    device: int
    workers: int
    models: List[ModelSpec]


def load_config(path: Path) -> PredictConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Validate required fields
    for field in ("images_dir", "output_dir", "models"):
        if not raw.get(field):
            raise ValueError(f"`{field}` is required in config.")

    models = []
    for item in raw["models"]:
        if not item.get("name") or not item.get("source"):
            raise ValueError("Each model entry must have `name` and `source`.")
        models.append(ModelSpec(name=str(item["name"]), source=str(item["source"])))

    return PredictConfig(
        name=str(raw.get("name", "detection")),
        images_dir=Path(raw["images_dir"]),
        output_dir=Path(raw["output_dir"]),
        confidence=float(raw.get("confidence", 0.25)),
        target_class=int(raw.get("target_class", 0)),
        image_ext=str(raw.get("image_ext", "png")),
        device=raw.get("device", 0),
        workers=int(raw.get("workers", 8)),
        models=models,
    )


# ==============================
# PREDICTION
# ==============================
def dump_detections(model_spec: ModelSpec, cfg: PredictConfig) -> str:
    print(f"\n[{model_spec.name}] Loading checkpoint: {model_spec.source}")
    model = YOLO(model_spec.source)

    # Collect and sort images (order matters for tracking!)
    image_paths = sorted(cfg.images_dir.glob(f"*.{cfg.image_ext}"))
    if not image_paths:
        raise FileNotFoundError(f"No .{cfg.image_ext} files found in {cfg.images_dir}")
    print(f"[{model_spec.name}] Found {len(image_paths)} images, running predict...")

    records = []
    for img_path in image_paths:
        results = model.predict(
            source=str(img_path),
            conf=cfg.confidence,
            device=cfg.device,
            workers=cfg.workers,
            verbose=False,
        )[0]

        labels = []
        for box in results.boxes:
            if int(box.cls) != cfg.target_class:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            labels.append({
                "Class": "person",
                "BoundingBoxes": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            })

        records.append({
            "File": img_path.name,
            "Labels": labels,
        })

    # Save JSON
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"{cfg.name}_{model_spec.name}_detections.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    total_dets = sum(len(r["Labels"]) for r in records)
    print(f"[{model_spec.name}] {len(records)} frames, {total_dets} detections → {out_path}")
    return str(out_path)


# ==============================
# MAIN
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Dump YOLO detections to JSON for tracking.")
    parser.add_argument("--config", required=True, type=Path, help="Path to predict config YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Running detection export for {len(cfg.models)} model(s)...")
    for model_spec in cfg.models:
        dump_detections(model_spec, cfg)

    print("\nDone. All detection JSONs ready for norfair_tracker.py.")


if __name__ == "__main__":
    main()