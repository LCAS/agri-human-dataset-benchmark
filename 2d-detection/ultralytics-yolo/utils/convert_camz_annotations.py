import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image


def find_image_path(root_dirs, filename):
    """Search for the given filename in the list of root directories."""
    for root in root_dirs:
        for dirpath, _, files in os.walk(root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None


def find_annotation_files(search_dir):
    """Recursively find all cam_zed_rgb_ann.json files."""
    ann_files = []
    for root, dirs, files in os.walk(search_dir):
        if "cam_zed_rgb_ann.json" in files:
            ann_files.append(os.path.join(root, "cam_zed_rgb_ann.json"))
    return sorted(ann_files)


def main():
    p = argparse.ArgumentParser(description="Convert cam_zed_rgb JSON annotations to YOLO format dataset layout")
    p.add_argument("--ann-dir", required=True, help="Directory to search recursively for cam_zed_rgb_ann.json files (e.g. Dataset/labelled_dataset)")
    p.add_argument(
        "--images-root",
        required=True,
        help="Root folder(s) to search for images. Comma-separated if multiple (e.g. Dataset/labelled_dataset)",
    )
    p.add_argument("--out", default="yolo_dataset", help="Output dataset folder (will be created)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true", help="Copy images into the output dataset (default: create symlinks when possible)")

    args = p.parse_args()

    images_root_dirs = [d.strip() for d in args.images_root.split(",")]

    # Find all annotation files
    print("\nSearching for annotation files in ", args.ann_dir)
    ann_files = find_annotation_files(args.ann_dir)
    print(f"\nFound {len(ann_files)} annotation file(s):")
    for f in ann_files:
        print(f"  - {f}")

    # Create output directories
    out_dir = Path(args.out)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"

    out_dir.mkdir(parents=True, exist_ok=True)
    (images_out / "train").mkdir(parents=True, exist_ok=True)
    (images_out / "val").mkdir(parents=True, exist_ok=True)
    (labels_out / "train").mkdir(parents=True, exist_ok=True)
    (labels_out / "val").mkdir(parents=True, exist_ok=True)

    # Load and merge all annotation files
    ann = []
    for ann_file in ann_files:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            ann.extend(data)
            print(f"\nLoaded {len(data)} annotations from {ann_file}")
    
    print(f"Total annotations: {len(ann)}")

    # Build mapping filename -> labels
    mapping = {}
    classes = set()
    for item in ann:
        fname = item.get("File")
        labels = item.get("Labels", [])
        mapping[fname] = labels
        for lab in labels:
            classes.add(lab.get("Class"))

    classes = sorted([c for c in classes if c is not None])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Find actual image paths
    image_paths = []
    missing = []
    for fname in mapping:
        path = find_image_path(images_root_dirs, fname)
        if path:
            image_paths.append(path)
        else:
            missing.append(fname)

    if missing:
        print(f"Warning: {len(missing)} images referenced in JSON were not found. Example: {missing[:5]}")

    # Split into train/val and write files
    random.seed(args.seed)
    random.shuffle(image_paths)
    print(f"{len(image_paths)} images found.")
    n_train = int(len(image_paths) * args.train_ratio)
    train_imgs = set(image_paths[:n_train])

    for img_path in image_paths:
        src = Path(img_path)
        fname = src.name
        split = "train" if img_path in train_imgs else "val"

        dst_img = images_out / split / fname
        dst_lbl = labels_out / split / (src.stem + ".txt")

        # copy or symlink image (maybe try symlink first)
        if args.copy_images:
            shutil.copy2(src, dst_img)
        else:
            try:
                if dst_img.exists():
                    dst_img.unlink()
                os.symlink(src, dst_img)
            except Exception:
                shutil.copy2(src, dst_img)

        # open image to get size
        with Image.open(src) as im:
            w, h = im.size

        # write label file
        anns = mapping.get(fname, [])
        lines = []
        for a in anns:
            cls = a.get("Class")
            bbox = a.get("BoundingBoxes")
            if bbox is None or cls is None:
                continue
            # JSON bbox is [x, y, width, height] with top-left origin
            x, y, bw, bh = bbox
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            cx_rel = cx / w
            cy_rel = cy / h
            bw_rel = bw / w
            bh_rel = bh / h
            cls_id = class_to_idx[cls]
            lines.append(f"{cls_id} {cx_rel:.6f} {cy_rel:.6f} {bw_rel:.6f} {bh_rel:.6f}")

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # write data.yaml
    data_yaml = out_dir / "data.yaml"
    content = f"path: {out_dir}\n"
    # content += f"nc: {len(classes)}\n"
    content += f"train: {Path('images') / 'train'}\n"
    content += f"val: {Path('images') / 'val'}\n"
    content += "names:\n"
    for i, c in enumerate(classes):
        content += f"  {i}: {c}\n"

    data_yaml.write_text(content)
    print(f"Wrote dataset to {out_dir}. Classes: {classes}. data.yaml at {data_yaml}")


if __name__ == "__main__":
    main()
