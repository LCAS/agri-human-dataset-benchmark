import json

def coco_to_mot(coco_path: str, output_path: str):
    with open(coco_path) as f:
        coco = json.load(f)

    # Map image_id → frame_index (sorted by image id = frame order)
    sorted_images = sorted(coco["images"], key=lambda x: x["id"])
    image_id_to_frame = {img["id"]: idx+1 for idx, img in enumerate(sorted_images)}

    lines = []
    for ann in coco["annotations"]:
        frame_idx = image_id_to_frame[ann["image_id"]]
        track_id = ann.get("track_id", ann["id"])  # use track_id if available, else ann id
        x, y, w, h = ann["bbox"]
        lines.append((frame_idx, track_id, x, y, w, h))

    # Sort by frame
    lines.sort(key=lambda r: r[0])

    with open(output_path, "w") as f:
        for frame, tid, x, y, w, h in lines:
            f.write(f"{frame},{tid},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1\n")

    print(f"GT MOT file written: {output_path}")

# Example usage:
coco_to_mot(
    r"D:\AOC\...\cam_zed_rgb_ann.json",
    r"D:\AOC\...\gt.txt"
)