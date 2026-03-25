import json
from pathlib import Path

def custom_gt_to_mot(gt_json_path: str, output_path: str):
    with open(gt_json_path) as f:
        records = json.load(f)

    # Extract unique class names and assign numeric IDs
    # "human1" → 1, "human2" → 2, "human3" → 3, etc.
    def class_to_id(class_name: str) -> int:
        # Extract the number from "human1", "human2", etc.
        digits = ''.join(filter(str.isdigit, class_name))
        return int(digits) if digits else hash(class_name) % 1000

    lines = []
    for frame_idx, rec in enumerate(records, start=1):
        for label in rec.get("Labels", []):
            cls = label.get("Class", "")
            bbox = label.get("BoundingBoxes", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            track_id = class_to_id(cls)
            lines.append(f"{frame_idx},{track_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"GT MOT file written: {output_path}")
    print(f"Total frames: {len(records)}, Total lines: {len(lines)}")

custom_gt_to_mot(
    r"D:\AOC\datasets\agri-human-sensing\labelled_dataset\out_vine_4swap+walk_st_ly_11_06_2024_2_label\annotations\cam_zed_rgb_ann.json",
    r"D:\AOC\agri-human-dataset-benchmark\2d-tracking\benchmarks\norfair\outputs\out_vine_4swap+walk_st_ly_11_06_2024_2_label_gt_mot.txt"
)