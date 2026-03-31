"""
Run a BoxMOT tracking job from explicit CLI arguments or a YAML config.
Relative paths in configs and CLI arguments resolve from the 2d-tracking root.

Examples:
    python src/run_tracker.py --config configs/tracking/default.yaml
    python src/run_tracker.py --config configs/tracking/default.yaml --tracker boosttrack --reid-weights /path/to/osnet_x0_25_msmt17.pt
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "tracking" / "default.yaml"
SUPPORTED_TRACKERS = ("boosttrack", "botsort", "bytetrack", "deepocsort", "ocsort", "strongsort")
TRACKER_CLASS_NAMES = {
    "boosttrack": "BoostTrack",
    "botsort": "BotSort",
    "bytetrack": "ByteTrack",
    "deepocsort": "DeepOcSort",
    "ocsort": "OcSort",
    "strongsort": "StrongSort",
}
TRACKER_MODULES = {
    "boosttrack": "boxmot.trackers.boosttrack.boosttrack",
    "botsort": "boxmot.trackers.botsort.botsort",
    "bytetrack": "boxmot.trackers.bytetrack.bytetrack",
    "deepocsort": "boxmot.trackers.deepocsort.deepocsort",
    "ocsort": "boxmot.trackers.ocsort.ocsort",
    "strongsort": "boxmot.trackers.strongsort.strongsort",
}
TRACKER_PARAM_ALIASES = {
    "iou_thresh": ("iou_threshold", "asso_threshold"),
    "iou_threshold": ("iou_thresh", "asso_threshold"),
    "asso_threshold": ("iou_threshold", "iou_thresh"),
}


@dataclass(frozen=True)
class TrackerConfig:
    """Normalized BoxMOT settings after YAML loading and CLI overrides."""

    detections_json: Path
    frames_dir: Optional[Path] = None
    mot_output: Optional[Path] = None
    output_video: Optional[Path] = None
    save_frames_dir: Optional[Path] = None
    frame_rate: float = 30.0
    tracker: str = "bytetrack"
    reid_weights: Optional[Union[str, Path]] = None
    device: str = "cpu"
    half: bool = False
    per_class: bool = False
    tracker_kwargs: Dict[str, Any] = field(default_factory=dict)
    box_thickness: int = 2
    id_offset: int = 10


def _resolve_path(value: Optional[Any]) -> Optional[Path]:
    """Resolve repo-relative config paths from the 2d-tracking root."""

    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def _resolve_reid_weights(value: Optional[Any]) -> Optional[Union[str, Path]]:
    """Resolve local weight paths but preserve plain model identifiers when needed."""

    if value in (None, ""):
        return None

    text = str(value)
    path = Path(text)
    if path.is_absolute():
        return path

    resolved = (TRACKING_ROOT / path).resolve()
    if resolved.exists() or "/" in text or "\\" in text:
        return resolved

    return text


def _parse_bool(value: Optional[Any]) -> Optional[bool]:
    """Accept YAML/CLI boolean values in a few human-friendly spellings."""

    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}.")


def _parse_tracker_kwargs(raw: Any) -> Dict[str, Any]:
    """Normalize the tracker-specific kwargs block from YAML."""

    if raw in (None, ""):
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("`tracker_kwargs` must be a mapping when provided.")
    return {str(key): value for key, value in raw.items()}


def load_config(path: Path) -> TrackerConfig:
    """Load one BoxMOT tracking YAML file into the internal config dataclass."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    detections_json = raw.get("detections_json")
    if not detections_json:
        raise ValueError("`detections_json` is required in the tracking config.")

    tracker = str(raw.get("tracker", "bytetrack")).strip().lower()
    if tracker not in SUPPORTED_TRACKERS:
        raise ValueError(f"`tracker` must be one of {SUPPORTED_TRACKERS}, got {tracker!r}.")

    return TrackerConfig(
        detections_json=_resolve_path(detections_json),
        frames_dir=_resolve_path(raw.get("frames_dir")),
        mot_output=_resolve_path(raw.get("mot_output")),
        output_video=_resolve_path(raw.get("output_video")),
        save_frames_dir=_resolve_path(raw.get("save_frames_dir")),
        frame_rate=float(raw.get("frame_rate", 30.0)),
        tracker=tracker,
        reid_weights=_resolve_reid_weights(raw.get("reid_weights")),
        device=str(raw.get("device", "cpu")),
        half=bool(_parse_bool(raw.get("half")) or False),
        per_class=bool(_parse_bool(raw.get("per_class")) or False),
        tracker_kwargs=_parse_tracker_kwargs(raw.get("tracker_kwargs")),
        box_thickness=int(raw.get("box_thickness", 2)),
        id_offset=int(raw.get("id_offset", 10)),
    )


def _to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert project boxes into xyxy, which is what BoxMOT expects."""

    x, y, w, h = map(float, box)
    if w > 0 and h > 0:
        return x, y, x + w, y + h
    return x, y, w, h


def _labels_iter(labels_field: Any) -> Iterable[Tuple[str, Tuple[float, float, float, float], float]]:
    """Yield `(label, xyxy_box, score)` tuples from the supported detector JSON variants."""

    if labels_field is None:
        return []

    def _extract(item: Mapping[str, Any]) -> Optional[Tuple[str, Tuple[float, float, float, float], float]]:
        label = (
            item.get("Class")
            or item.get("class")
            or item.get("label")
            or item.get("Label")
        )
        box = item.get("BoundingBoxes") or item.get("bbox") or item.get("box")
        score = item.get("Confidence")
        if score is None:
            score = item.get("confidence")
        if score is None:
            score = item.get("Score")
        if score is None:
            score = item.get("score")
        if score is None:
            score = 1.0
        if label is None or box is None:
            return None
        return str(label), _to_xyxy(box), float(score)

    if isinstance(labels_field, Mapping):
        extracted = _extract(labels_field)
        if extracted is not None:
            yield extracted
        return

    if isinstance(labels_field, list):
        for item in labels_field:
            if not isinstance(item, Mapping):
                continue
            extracted = _extract(item)
            if extracted is not None:
                yield extracted
        return

    return []


def detections_from_json_record(record: Dict[str, Any], class_ids: MutableMapping[str, int]) -> np.ndarray:
    """Adapt one project detection record into BoxMOT's `[x1,y1,x2,y2,conf,cls]` array."""

    detections = []
    dropped = 0

    for label, (x1, y1, x2, y2), score in _labels_iter(record.get("Labels")):
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue

        class_id = class_ids.setdefault(label, len(class_ids))
        detections.append([x1, y1, x2, y2, score, float(class_id)])

    if dropped:
        warnings.warn(
            f"Dropped {dropped} invalid bbox(es) in record {record.get('File', '<no-file>')}",
            stacklevel=2,
        )

    if not detections:
        return np.empty((0, 6), dtype=np.float32)

    return np.asarray(detections, dtype=np.float32)


class MotWriter:
    """Append tracked boxes to a MOTChallenge-format text file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write_frame(self, frame_idx: int, tracked_objects: np.ndarray) -> None:
        if tracked_objects.size == 0:
            return

        lines = []
        for track in tracked_objects:
            x1, y1, x2, y2 = map(float, track[:4])
            track_id = int(track[4])
            width = x2 - x1
            height = y2 - y1
            lines.append(
                f"{frame_idx},{track_id},{x1:f},{y1:f},{width:f},{height:f},1,-1,-1,-1\n"
            )

        with self.path.open("a", encoding="utf-8") as file:
            file.writelines(lines)


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate a stable visualization color from the integer track id."""

    seed = int(track_id) * 2654435761 % (1 << 24)
    return (seed & 255, (seed >> 8) & 255, (seed >> 16) & 255)


def _draw_tracks(frame: np.ndarray, tracked_objects: np.ndarray, box_thickness: int, id_offset: int) -> np.ndarray:
    """Render tracked boxes and IDs on one frame for videos or debug exports."""

    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, np.round(track[:4]))
        track_id = int(track[4])
        color = _color_for_id(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        cv2.putText(
            frame,
            f"ID{track_id}",
            (x1, max(0, y1 - id_offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def _fallback_frame_shape(detections: np.ndarray, last_shape: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Build a reasonable blank canvas when frames are unavailable on disk."""

    if detections.size > 0:
        width = max(int(np.ceil(np.max(detections[:, 2]))) + 32, 2)
        height = max(int(np.ceil(np.max(detections[:, 3]))) + 32, 2)
        return (height, width, 3)
    if last_shape is not None:
        return last_shape
    return (720, 1280, 3)


def _load_frame(
    record: Dict[str, Any],
    frames_dir: Optional[Path],
    detections: np.ndarray,
    last_shape: Optional[Tuple[int, int, int]],
) -> Tuple[np.ndarray, Tuple[int, int, int], str]:
    """Load the real frame when available, otherwise synthesize a blank one."""

    file_name = Path(record.get("File", "frame.png")).name
    if frames_dir is not None:
        frame_path = frames_dir / file_name
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            return frame, frame.shape, file_name

    shape = _fallback_frame_shape(detections, last_shape)
    return np.zeros(shape, dtype=np.uint8), shape, file_name


def _import_tracker_class(tracker_name: str):
    """Import the requested BoxMOT tracker lazily so `--help` works without the package."""

    try:
        boxmot = importlib.import_module("boxmot")
    except ImportError as exc:
        raise RuntimeError(
            "BoxMOT is not installed in the active interpreter. Create or activate the "
            "virtual environment for `2d-tracking/benchmarks/boxmot/requirements.txt` first."
        ) from exc

    class_name = TRACKER_CLASS_NAMES[tracker_name]
    if hasattr(boxmot, class_name):
        return getattr(boxmot, class_name)

    module = importlib.import_module(TRACKER_MODULES[tracker_name])
    return getattr(module, class_name)


def _normalize_tracker_kwargs(tracker_kwargs: Mapping[str, Any], accepted: Sequence[str]) -> Dict[str, Any]:
    """Map user-facing tracker kwargs onto the constructor parameters this version accepts."""

    normalized: Dict[str, Any] = {}
    accepted_set = set(accepted)

    for key, value in tracker_kwargs.items():
        target_key = key
        if target_key not in accepted_set:
            for alias in TRACKER_PARAM_ALIASES.get(key, ()):
                if alias in accepted_set:
                    target_key = alias
                    break
            else:
                raise ValueError(
                    f"Unsupported tracker kwarg {key!r}. Accepted parameters are: {sorted(accepted_set)}"
                )
        normalized[target_key] = value

    return normalized


def build_tracker(cfg: TrackerConfig):
    """Instantiate the selected BoxMOT tracker from the normalized config."""

    tracker_class = _import_tracker_class(cfg.tracker)
    signature = inspect.signature(tracker_class)
    accepted = [name for name in signature.parameters if name != "self"]

    tracker_kwargs: Dict[str, Any] = {}
    if "reid_weights" in signature.parameters:
        param = signature.parameters["reid_weights"]
        if param.default is inspect._empty and cfg.reid_weights is None:
            raise ValueError(
                f"`reid_weights` is required for tracker {cfg.tracker!r}. "
                "Point it at a local weights file or a BoxMOT-supported weights identifier."
            )
        if cfg.reid_weights is not None:
            tracker_kwargs["reid_weights"] = cfg.reid_weights

    if "device" in signature.parameters:
        tracker_kwargs["device"] = cfg.device
    if "half" in signature.parameters:
        tracker_kwargs["half"] = cfg.half
    if "per_class" in signature.parameters:
        tracker_kwargs["per_class"] = cfg.per_class
    if "frame_rate" in signature.parameters:
        tracker_kwargs["frame_rate"] = int(round(cfg.frame_rate))

    tracker_kwargs.update(_normalize_tracker_kwargs(cfg.tracker_kwargs, accepted))
    return tracker_class(**tracker_kwargs)


def _tracker_uses_blank_frame_poorly(cfg: TrackerConfig) -> bool:
    """Flag trackers that degrade sharply without real frames for ReID/CMC."""

    if cfg.tracker == "strongsort":
        return True
    if cfg.tracker == "botsort":
        return bool(cfg.tracker_kwargs.get("with_reid", True))
    if cfg.tracker == "boosttrack":
        return bool(cfg.tracker_kwargs.get("with_reid", False))
    if cfg.tracker == "deepocsort":
        return not bool(cfg.tracker_kwargs.get("embedding_off", False))
    return False


def validate_config(cfg: TrackerConfig) -> None:
    """Check input/output paths before the tracking loop starts."""

    if not cfg.detections_json.is_file():
        raise FileNotFoundError(f"Detections JSON not found: {cfg.detections_json}")
    if cfg.frames_dir is not None and not cfg.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {cfg.frames_dir}")


def _parse_cli_tracker_kwargs(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Parse repeated `NAME=VALUE` CLI overrides for tracker-specific kwargs."""

    if not items:
        return {}

    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid `--tracker-kwarg` value {item!r}. Expected NAME=VALUE."
            )
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def _normalize_tracks_output(tracked_objects: Any) -> np.ndarray:
    """Coerce tracker outputs into one predictable `NxM` float array."""

    if tracked_objects is None:
        return np.empty((0, 8), dtype=np.float32)

    tracks = np.asarray(tracked_objects)
    if tracks.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    if tracks.ndim == 1:
        tracks = tracks.reshape(1, -1)
    if tracks.ndim != 2 or tracks.shape[1] < 5:
        raise ValueError(
            f"Unexpected BoxMOT output shape {tracks.shape}. Expected NxM with M >= 5."
        )
    return tracks.astype(np.float32, copy=False)


def run(cfg: TrackerConfig) -> None:
    """Run one BoxMOT pass over the frame-ordered detection export."""

    validate_config(cfg)
    records = json.loads(cfg.detections_json.read_text(encoding="utf-8"))
    tracker = build_tracker(cfg)

    if cfg.frames_dir is None and _tracker_uses_blank_frame_poorly(cfg):
        warnings.warn(
            f"`frames_dir` is not set for tracker {cfg.tracker!r}. The run will fall back to "
            "blank frames, which is usually a bad fit for appearance-based tracking.",
            stacklevel=2,
        )

    mot_writer = MotWriter(cfg.mot_output) if cfg.mot_output else None
    video_writer = None
    last_shape: Optional[Tuple[int, int, int]] = None
    class_ids: Dict[str, int] = {}

    if cfg.output_video is not None:
        cfg.output_video.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_frames_dir is not None:
        cfg.save_frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, record in enumerate(records, start=1):
        detections = detections_from_json_record(record, class_ids)
        frame, last_shape, file_name = _load_frame(record, cfg.frames_dir, detections, last_shape)
        tracked_objects = _normalize_tracks_output(tracker.update(detections, frame))

        if mot_writer is not None:
            mot_writer.write_frame(frame_idx, tracked_objects)

        if cfg.output_video is None and cfg.save_frames_dir is None:
            continue

        annotated = _draw_tracks(frame.copy(), tracked_objects, cfg.box_thickness, cfg.id_offset)

        if cfg.output_video is not None:
            if video_writer is None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(cfg.output_video),
                    fourcc,
                    cfg.frame_rate,
                    (width, height),
                )
            video_writer.write(annotated)

        if cfg.save_frames_dir is not None:
            cv2.imwrite(str(cfg.save_frames_dir / file_name), annotated)

    if video_writer is not None:
        video_writer.release()

    if hasattr(tracker, "dump_cache"):
        tracker.dump_cache()


def parse_args() -> argparse.Namespace:
    """Define the CLI that layers ad hoc overrides on top of the YAML config."""

    parser = argparse.ArgumentParser(
        description="Run BoxMOT tracking from a config or explicit args.",
        epilog=(
            "Examples:\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml "
            "--tracker strongsort --reid-weights /path/to/osnet_x0_25_msmt17.pt\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml "
            "--tracker-kwarg track_buffer=60 --tracker-kwarg with_reid=true"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a tracking YAML config. Relative data and report paths resolve from 2d-tracking.",
    )
    parser.add_argument("--detections-json", type=Path, help="Detections JSON file to track.")
    parser.add_argument("--frames-dir", type=Path, help="Optional directory containing video frames.")
    parser.add_argument("--out-mot", type=Path, help="Output MOT file path.")
    parser.add_argument("--out-video", type=Path, help="Output MP4 path.")
    parser.add_argument("--save-frames-dir", type=Path, help="Directory for annotated output frames.")
    parser.add_argument("--frame-rate", type=float, help="Output video FPS.")
    parser.add_argument("--tracker", choices=SUPPORTED_TRACKERS, help="BoxMOT tracker to run.")
    parser.add_argument(
        "--reid-weights",
        type=str,
        help="Optional ReID weights path or BoxMOT-supported model identifier.",
    )
    parser.add_argument("--device", type=str, help="Tracker device, for example `cpu` or `cuda:0`.")
    parser.add_argument("--half", type=str, help="Whether to use half precision: true/false.")
    parser.add_argument("--per-class", type=str, help="Track each class independently: true/false.")
    parser.add_argument(
        "--tracker-kwarg",
        action="append",
        default=[],
        help="Tracker-specific override in NAME=VALUE form. May be repeated.",
    )
    parser.add_argument("--box-thickness", type=int, help="Rendered bounding box thickness.")
    parser.add_argument("--id-offset", type=int, help="Vertical text offset in pixels.")
    return parser.parse_args()


def merge_cli_overrides(cfg: TrackerConfig, args: argparse.Namespace) -> TrackerConfig:
    """Overlay any explicit CLI flags on top of the loaded YAML config."""

    updates = {}
    if args.detections_json is not None:
        updates["detections_json"] = _resolve_path(args.detections_json)
    if args.frames_dir is not None:
        updates["frames_dir"] = _resolve_path(args.frames_dir)
    if args.out_mot is not None:
        updates["mot_output"] = _resolve_path(args.out_mot)
    if args.out_video is not None:
        updates["output_video"] = _resolve_path(args.out_video)
    if args.save_frames_dir is not None:
        updates["save_frames_dir"] = _resolve_path(args.save_frames_dir)
    if args.frame_rate is not None:
        updates["frame_rate"] = float(args.frame_rate)
    if args.tracker is not None:
        updates["tracker"] = args.tracker
    if args.reid_weights is not None:
        updates["reid_weights"] = _resolve_reid_weights(args.reid_weights)
    if args.device is not None:
        updates["device"] = args.device
    if args.half is not None:
        parsed_half = _parse_bool(args.half)
        if parsed_half is None:
            raise ValueError("`--half` expects true/false.")
        updates["half"] = parsed_half
    if args.per_class is not None:
        parsed_per_class = _parse_bool(args.per_class)
        if parsed_per_class is None:
            raise ValueError("`--per-class` expects true/false.")
        updates["per_class"] = parsed_per_class
    if args.box_thickness is not None:
        updates["box_thickness"] = int(args.box_thickness)
    if args.id_offset is not None:
        updates["id_offset"] = int(args.id_offset)
    if args.tracker_kwarg:
        tracker_kwargs = dict(cfg.tracker_kwargs)
        tracker_kwargs.update(_parse_cli_tracker_kwargs(args.tracker_kwarg))
        updates["tracker_kwargs"] = tracker_kwargs
    return replace(cfg, **updates)


def main() -> None:
    """CLI entrypoint used by local scripts and SLURM launchers."""

    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
