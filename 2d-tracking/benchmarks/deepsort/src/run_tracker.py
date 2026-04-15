"""
Run a DeepSORT tracking job from explicit CLI arguments or a YAML config.
Relative paths in configs and CLI arguments resolve from the 2d-tracking root.

Examples:
    python src/run_tracker.py --config configs/tracking/default.yaml
    python src/run_tracker.py --config configs/tracking/default.yaml --model-path /path/to/mars-small128.pb
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment
import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "tracking" / "default.yaml"
INFTY_COST = 1e5


@dataclass(frozen=True)
class TrackerConfig:
    """Normalized DeepSORT settings after YAML loading and CLI overrides."""

    detections_json: Optional[Path]
    frames_dir: Optional[Path]
    model_path: Optional[Path]
    mot_output: Optional[Path] = None
    output_video: Optional[Path] = None
    save_frames_dir: Optional[Path] = None
    frame_rate: float = 30.0
    min_confidence: float = 0.3
    min_detection_height: int = 0
    nms_max_overlap: float = 1.0
    max_cosine_distance: float = 0.2
    nn_budget: Optional[int] = None
    max_iou_distance: float = 0.7
    max_age: int = 30
    n_init: int = 3
    encoder_batch_size: int = 32
    reid_input_name: str = "images"
    reid_output_name: str = "features"
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


def _optional_int(value: Optional[Any]) -> Optional[int]:
    """Parse optional integer fields that may be blank or explicitly `None`."""

    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return int(value)


def load_config(path: Path) -> TrackerConfig:
    """Load one DeepSORT tracking YAML file into the internal config dataclass."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    return TrackerConfig(
        detections_json=_resolve_path(raw.get("detections_json")),
        frames_dir=_resolve_path(raw.get("frames_dir")),
        model_path=_resolve_path(raw.get("model_path")),
        mot_output=_resolve_path(raw.get("mot_output")),
        output_video=_resolve_path(raw.get("output_video")),
        save_frames_dir=_resolve_path(raw.get("save_frames_dir")),
        frame_rate=float(raw.get("frame_rate", 30.0)),
        min_confidence=float(raw.get("min_confidence", 0.3)),
        min_detection_height=int(raw.get("min_detection_height", 0)),
        nms_max_overlap=float(raw.get("nms_max_overlap", 1.0)),
        max_cosine_distance=float(raw.get("max_cosine_distance", 0.2)),
        nn_budget=_optional_int(raw.get("nn_budget")),
        max_iou_distance=float(raw.get("max_iou_distance", 0.7)),
        max_age=int(raw.get("max_age", 30)),
        n_init=int(raw.get("n_init", 3)),
        encoder_batch_size=int(raw.get("encoder_batch_size", 32)),
        reid_input_name=str(raw.get("reid_input_name", "images")),
        reid_output_name=str(raw.get("reid_output_name", "features")),
        box_thickness=int(raw.get("box_thickness", 2)),
        id_offset=int(raw.get("id_offset", 10)),
    )


def _labels_iter(
    labels_field: Any,
) -> Iterable[Tuple[str, Tuple[float, float, float, float], float]]:
    """Yield `(label, tlwh_box, score)` tuples from the supported detector JSON variants."""

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
        if label is None or box is None or len(box) != 4:
            return None
        x, y, w, h = map(float, box)
        return str(label), (x, y, w, h), float(score)

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


@dataclass
class RawDetection:
    """One parsed detection before appearance embedding extraction."""

    tlwh: np.ndarray
    confidence: float
    label: str

    def to_xyah(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.asarray([x + w / 2.0, y + h / 2.0, w / max(h, 1e-6), h], dtype=np.float32)


@dataclass
class Detection:
    """DeepSORT detection with one appearance feature vector."""

    tlwh: np.ndarray
    confidence: float
    feature: np.ndarray
    label: str

    def to_tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.asarray([x + w / 2.0, y + h / 2.0, w / max(h, 1e-6), h], dtype=np.float32)


class TrackState:
    """Small enum-like namespace for the track lifecycle."""

    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    """One DeepSORT track with Kalman state and appearance history."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        detection: Detection,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.features: List[np.ndarray] = []
        self.label = detection.label
        self._n_init = n_init
        self._max_age = max_age
        if detection.feature.size:
            self.features.append(detection.feature)

    def to_tlwh(self) -> np.ndarray:
        x, y, a, h = self.mean[:4].copy()
        w = a * h
        return np.asarray([x - w / 2.0, y - h / 2.0, w, h], dtype=np.float32)

    def predict(self, kf: "KalmanFilter") -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: "KalmanFilter", detection: Detection) -> None:
        self.mean, self.covariance = kf.update(
            self.mean,
            self.covariance,
            detection.to_xyah(),
        )
        self.label = detection.label
        if detection.feature.size:
            self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE or self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED


class KalmanFilter:
    """Kalman filter used by DeepSORT for `(cx, cy, aspect_ratio, height)` boxes."""

    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.0700,
        6: 12.5920,
        7: 14.0670,
        8: 15.5070,
        9: 16.9190,
    }

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2.0 * self._std_weight_position * measurement[3],
            2.0 * self._std_weight_position * measurement[3],
            1e-2,
            2.0 * self._std_weight_position * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            10.0 * self._std_weight_velocity * measurement[3],
            1e-5,
            10.0 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean = mean[:2]
            covariance = covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor,
            d.T,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )
        return np.sum(z * z, axis=0)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    a = _normalize_rows(a.astype(np.float32, copy=False))
    b = _normalize_rows(b.astype(np.float32, copy=False))
    return 1.0 - np.dot(a, b.T)


class NearestNeighborDistanceMetric:
    """Cosine nearest-neighbor metric used for appearance association."""

    def __init__(self, matching_threshold: float, budget: Optional[int] = None) -> None:
        self.matching_threshold = float(matching_threshold)
        self.budget = budget
        self.samples: Dict[int, List[np.ndarray]] = {}

    def partial_fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        active_targets: Sequence[int],
    ) -> None:
        for feature, target in zip(features, targets):
            target_id = int(target)
            self.samples.setdefault(target_id, []).append(feature.astype(np.float32, copy=False))
            if self.budget is not None:
                self.samples[target_id] = self.samples[target_id][-self.budget :]

        active_target_set = {int(target) for target in active_targets}
        self.samples = {
            target_id: samples
            for target_id, samples in self.samples.items()
            if target_id in active_target_set
        }

    def distance(self, features: np.ndarray, targets: Sequence[int]) -> np.ndarray:
        cost_matrix = np.full((len(targets), len(features)), INFTY_COST, dtype=np.float32)
        if len(features) == 0 or len(targets) == 0:
            return cost_matrix

        for row, target in enumerate(targets):
            samples = self.samples.get(int(target))
            if not samples:
                continue
            sample_matrix = np.asarray(samples, dtype=np.float32)
            distances = _cosine_distance(sample_matrix, features)
            cost_matrix[row, :] = np.min(distances, axis=0)
        return cost_matrix


def _iou(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.maximum(bbox_tl, candidates_tl)
    br = np.minimum(bbox_br, candidates_br)
    wh = np.maximum(0.0, br - tl)

    intersection = wh[:, 0] * wh[:, 1]
    bbox_area = bbox[2] * bbox[3]
    candidate_area = candidates[:, 2] * candidates[:, 3]
    union = np.maximum(bbox_area + candidate_area - intersection, 1e-12)
    return intersection / union


def _non_max_suppression(boxes_tlwh: np.ndarray, scores: np.ndarray, max_overlap: float) -> np.ndarray:
    if boxes_tlwh.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes_tlwh[:, 0]
    y1 = boxes_tlwh[:, 1]
    x2 = boxes_tlwh[:, 0] + boxes_tlwh[:, 2]
    y2 = boxes_tlwh[:, 1] + boxes_tlwh[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        overlap = inter / np.maximum(areas[order[1:]], 1e-12)
        order = order[np.where(overlap <= max_overlap)[0] + 1]

    return np.asarray(keep, dtype=np.int64)


def _extract_image_patch(image: np.ndarray, bbox: np.ndarray, patch_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    bbox = np.asarray(bbox, dtype=np.float32).copy()

    target_aspect = float(patch_shape[1]) / float(patch_shape[0])
    new_width = target_aspect * bbox[3]
    bbox[0] -= (new_width - bbox[2]) / 2.0
    bbox[2] = new_width

    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None

    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, tuple(patch_shape[::-1]))


class FrozenGraphEncoder:
    """TensorFlow frozen-graph image encoder compatible with MARS `.pb` files."""

    def __init__(
        self,
        model_path: Path,
        input_name: str = "images",
        output_name: str = "features",
        batch_size: int = 32,
    ) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed in the active interpreter. "
                "Install the DeepSORT benchmark requirements before running this tracker."
            ) from exc

        tf.compat.v1.disable_eager_execution()
        self.batch_size = int(batch_size)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.io.gfile.GFile(str(model_path), "rb") as file_handle:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(file_handle.read())
                tf.compat.v1.import_graph_def(graph_def, name="")

        self.session = tf.compat.v1.Session(graph=self.graph)
        self.input_var = self._lookup_tensor(self.graph, input_name)
        self.output_var = self._lookup_tensor(self.graph, output_name)
        self.feature_dim = int(self.output_var.shape.as_list()[-1])
        image_shape = self.input_var.shape.as_list()[1:]
        if len(image_shape) != 3 or any(dim is None for dim in image_shape):
            raise ValueError(
                f"Unexpected ReID input tensor shape {image_shape!r}; expected [H, W, C]."
            )
        self.image_shape = tuple(int(dim) for dim in image_shape)
        self.input_dtype = self.input_var.dtype.as_numpy_dtype

    @staticmethod
    def _candidate_tensor_names(name: str) -> Iterable[str]:
        base = name[:-2] if name.endswith(":0") else name
        yield f"{base}:0"
        yield f"net/{base}:0"

    def _lookup_tensor(self, graph, name: str):
        last_error: Optional[Exception] = None
        for candidate in self._candidate_tensor_names(name):
            try:
                return graph.get_tensor_by_name(candidate)
            except KeyError as exc:
                last_error = exc
        raise KeyError(f"Tensor {name!r} was not found in the frozen graph.") from last_error

    def __call__(self, image_bgr: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
        if len(boxes_tlwh) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        patch_shape = self.image_shape[:2]
        image_patches = []
        for box in boxes_tlwh:
            patch = _extract_image_patch(image_bgr, box, patch_shape)
            if patch is None:
                warnings.warn(
                    f"Failed to extract image patch for bbox {box.tolist()}; using zeros instead.",
                    stacklevel=2,
                )
                patch = np.zeros(self.image_shape, dtype=np.uint8)
            image_patches.append(patch)

        data_x = np.asarray(image_patches, dtype=self.input_dtype)
        out = np.zeros((len(data_x), self.feature_dim), dtype=np.float32)

        start = 0
        while start < len(data_x):
            end = min(start + self.batch_size, len(data_x))
            out[start:end] = self.session.run(
                self.output_var,
                feed_dict={self.input_var: data_x[start:end]},
            )
            start = end

        return out


def detections_from_json_record(record: Dict[str, Any]) -> List[RawDetection]:
    """Adapt one project detection record into intermediate DeepSORT detections."""

    detections: List[RawDetection] = []
    dropped = 0

    for label, (x, y, w, h), score in _labels_iter(record.get("Labels")):
        if w <= 0 or h <= 0:
            dropped += 1
            continue
        detections.append(
            RawDetection(
                tlwh=np.asarray([x, y, w, h], dtype=np.float32),
                confidence=float(score),
                label=label,
            )
        )

    if dropped:
        warnings.warn(
            f"Dropped {dropped} invalid bbox(es) in record {record.get('File', '<no-file>')}",
            stacklevel=2,
        )

    return detections


def create_detections_with_embeddings(
    raw_detections: Sequence[RawDetection],
    image_bgr: np.ndarray,
    encoder: FrozenGraphEncoder,
    min_confidence: float,
    min_detection_height: int,
    nms_max_overlap: float,
) -> List[Detection]:
    """Filter, NMS, and embed one frame of detections."""

    filtered = [
        detection
        for detection in raw_detections
        if detection.confidence >= min_confidence and detection.tlwh[3] >= min_detection_height
    ]
    if not filtered:
        return []

    boxes_tlwh = np.asarray([detection.tlwh for detection in filtered], dtype=np.float32)
    scores = np.asarray([detection.confidence for detection in filtered], dtype=np.float32)
    keep = _non_max_suppression(boxes_tlwh, scores, nms_max_overlap)
    if len(keep) == 0:
        return []

    kept_detections = [filtered[idx] for idx in keep]
    kept_boxes = np.asarray([detection.tlwh for detection in kept_detections], dtype=np.float32)
    features = encoder(image_bgr, kept_boxes)

    return [
        Detection(
            tlwh=detection.tlwh.copy(),
            confidence=detection.confidence,
            feature=feature.astype(np.float32, copy=False),
            label=detection.label,
        )
        for detection, feature in zip(kept_detections, features)
    ]


def _gate_cost_matrix(
    kf: KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    track_indices: Sequence[int],
    detection_indices: Sequence[int],
    only_position: bool = False,
) -> np.ndarray:
    if not track_indices or not detection_indices:
        return cost_matrix

    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices], dtype=np.float32)
    threshold = KalmanFilter.chi2inv95[2 if only_position else 4]
    for row, track_idx in enumerate(track_indices):
        gating_distances = kf.gating_distance(
            tracks[track_idx].mean,
            tracks[track_idx].covariance,
            measurements,
            only_position=only_position,
        )
        cost_matrix[row, gating_distances > threshold] = INFTY_COST
    return cost_matrix


def _min_cost_matching(
    distance_metric,
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    max_distance: float,
    track_indices: Optional[Sequence[int]] = None,
    detection_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    track_indices = list(track_indices)
    detection_indices = list(detection_indices)
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = np.asarray(
        distance_metric(tracks, detections, track_indices, detection_indices),
        dtype=np.float32,
    )
    if cost_matrix.size == 0:
        return [], track_indices, detection_indices

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches: List[Tuple[int, int]] = []
    unmatched_tracks = list(track_indices)
    unmatched_detections = list(detection_indices)

    for row, col in zip(row_indices, col_indices):
        if not np.isfinite(cost_matrix[row, col]) or cost_matrix[row, col] > max_distance:
            continue
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        matches.append((track_idx, detection_idx))
        unmatched_tracks.remove(track_idx)
        unmatched_detections.remove(detection_idx)

    return matches, unmatched_tracks, unmatched_detections


def _matching_cascade(
    distance_metric,
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    max_distance: float,
    cascade_depth: int,
    track_indices: Optional[Sequence[int]] = None,
    detection_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches: List[Tuple[int, int]] = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        track_indices_l = [
            track_idx
            for track_idx in track_indices
            if tracks[track_idx].time_since_update == level + 1
        ]
        if len(track_indices_l) == 0:
            continue
        matches_l, _, unmatched_detections = _min_cost_matching(
            distance_metric,
            tracks,
            detections,
            max_distance,
            track_indices_l,
            unmatched_detections,
        )
        matches.extend(matches_l)

    matched_track_indices = {track_idx for track_idx, _ in matches}
    unmatched_tracks = [track_idx for track_idx in track_indices if track_idx not in matched_track_indices]
    return matches, unmatched_tracks, unmatched_detections


def _iou_cost(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    track_indices: Sequence[int],
    detection_indices: Sequence[int],
) -> np.ndarray:
    cost_matrix = np.full((len(track_indices), len(detection_indices)), INFTY_COST, dtype=np.float32)
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return cost_matrix

    candidate_boxes = np.asarray([detections[i].tlwh for i in detection_indices], dtype=np.float32)
    candidate_labels = [detections[i].label for i in detection_indices]
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if track.time_since_update > 1:
            continue
        ious = _iou(track.to_tlwh(), candidate_boxes)
        for col, label in enumerate(candidate_labels):
            if label != track.label:
                continue
            cost_matrix[row, col] = 1.0 - ious[col]
    return cost_matrix


class DeepSortTracker:
    """Minimal DeepSORT tracker for one video sequence."""

    def __init__(
        self,
        max_cosine_distance: float,
        nn_budget: Optional[int],
        max_iou_distance: float,
        max_age: int,
        n_init: int,
    ) -> None:
        self.metric = NearestNeighborDistanceMetric(max_cosine_distance, nn_budget)
        self.max_iou_distance = float(max_iou_distance)
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1

    def predict(self) -> None:
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: Sequence[Detection]) -> None:
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        active_targets = [track.track_id for track in self.tracks if track.is_confirmed()]
        features: List[np.ndarray] = []
        targets: List[int] = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            for feature in track.features:
                features.append(feature)
                targets.append(track.track_id)
            track.features.clear()

        feature_matrix = np.asarray(features, dtype=np.float32) if features else np.empty((0, 0), dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.int64)
        self.metric.partial_fit(feature_matrix, target_array, active_targets)

    def _match(self, detections: Sequence[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        def gated_metric(
            tracks: Sequence[Track],
            detections: Sequence[Detection],
            track_indices: Sequence[int],
            detection_indices: Sequence[int],
        ) -> np.ndarray:
            features = np.asarray([detections[i].feature for i in detection_indices], dtype=np.float32)
            targets = [tracks[i].track_id for i in track_indices]
            cost_matrix = self.metric.distance(features, targets)
            for row, track_idx in enumerate(track_indices):
                track = tracks[track_idx]
                for col, detection_idx in enumerate(detection_indices):
                    if detections[detection_idx].label != track.label:
                        cost_matrix[row, col] = INFTY_COST
            return _gate_cost_matrix(
                self.kf,
                cost_matrix,
                tracks,
                detections,
                track_indices,
                detection_indices,
            )

        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed_tracks = [i for i, track in enumerate(self.tracks) if not track.is_confirmed()]

        matches_a, unmatched_confirmed_tracks, unmatched_detections = _matching_cascade(
            gated_metric,
            self.tracks,
            detections,
            self.metric.matching_threshold,
            self.max_age,
            confirmed_tracks,
        )

        iou_track_candidates = unconfirmed_tracks + [
            track_idx
            for track_idx in unmatched_confirmed_tracks
            if self.tracks[track_idx].time_since_update == 1
        ]
        unmatched_confirmed_tracks = [
            track_idx
            for track_idx in unmatched_confirmed_tracks
            if self.tracks[track_idx].time_since_update != 1
        ]

        matches_b, unmatched_iou_tracks, unmatched_detections = _min_cost_matching(
            _iou_cost,
            self.tracks,
            detections,
            1.0 - self.max_iou_distance,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = unmatched_confirmed_tracks + unmatched_iou_tracks
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: Detection) -> None:
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean=mean,
                covariance=covariance,
                track_id=self._next_id,
                n_init=self.n_init,
                max_age=self.max_age,
                detection=detection,
            )
        )
        self._next_id += 1


class MotWriter:
    """Append DeepSORT boxes to a MOTChallenge-format text file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write_frame(self, frame_idx: int, tracks: Sequence[Track]) -> None:
        lines = []
        for track in tracks:
            x, y, w, h = map(float, track.to_tlwh())
            lines.append(f"{frame_idx},{track.track_id},{x:f},{y:f},{w:f},{h:f},1,-1,-1,-1\n")
        with self.path.open("a", encoding="utf-8") as file:
            file.writelines(lines)


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    seed = int(track_id) * 2654435761 % (1 << 24)
    return (seed & 255, (seed >> 8) & 255, (seed >> 16) & 255)


def _draw_tracks(frame: np.ndarray, tracks: Sequence[Track], box_thickness: int, id_offset: int) -> np.ndarray:
    for track in tracks:
        x, y, w, h = map(int, np.round(track.to_tlwh()))
        color = _color_for_id(track.track_id)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)
        cv2.putText(
            frame,
            f"ID{track.track_id}",
            (x, max(0, y - id_offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def validate_config(cfg: TrackerConfig) -> None:
    """Check input/output paths before the tracking loop starts."""

    if cfg.detections_json is None:
        raise ValueError("`detections_json` must be provided in the config or via `--detections-json`.")
    if cfg.frames_dir is None:
        raise ValueError("`frames_dir` must be provided in the config or via `--frames-dir`.")
    if cfg.model_path is None:
        raise ValueError("`model_path` must be provided in the config or via `--model-path`.")
    if not cfg.detections_json.is_file():
        raise FileNotFoundError(f"Detections JSON not found: {cfg.detections_json}")
    if not cfg.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {cfg.frames_dir}")
    if not cfg.model_path.is_file():
        raise FileNotFoundError(f"DeepSORT ReID graph not found: {cfg.model_path}")


def run(cfg: TrackerConfig) -> None:
    """Run one DeepSORT pass over the frame-ordered detection export."""

    validate_config(cfg)
    records = json.loads(cfg.detections_json.read_text(encoding="utf-8"))
    tracker = DeepSortTracker(
        max_cosine_distance=cfg.max_cosine_distance,
        nn_budget=cfg.nn_budget,
        max_iou_distance=cfg.max_iou_distance,
        max_age=cfg.max_age,
        n_init=cfg.n_init,
    )
    encoder = FrozenGraphEncoder(
        model_path=cfg.model_path,
        input_name=cfg.reid_input_name,
        output_name=cfg.reid_output_name,
        batch_size=cfg.encoder_batch_size,
    )

    mot_writer = MotWriter(cfg.mot_output) if cfg.mot_output else None
    video_writer = None

    if cfg.output_video is not None:
        cfg.output_video.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_frames_dir is not None:
        cfg.save_frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, record in enumerate(records, start=1):
        file_name = Path(record.get("File", f"frame_{frame_idx:05d}.png")).name
        frame_path = cfg.frames_dir / file_name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        raw_detections = detections_from_json_record(record)
        detections = create_detections_with_embeddings(
            raw_detections=raw_detections,
            image_bgr=frame,
            encoder=encoder,
            min_confidence=cfg.min_confidence,
            min_detection_height=cfg.min_detection_height,
            nms_max_overlap=cfg.nms_max_overlap,
        )

        tracker.predict()
        tracker.update(detections)

        mot_tracks = [
            track
            for track in tracker.tracks
            if track.is_confirmed() and track.time_since_update <= 1
        ]
        draw_tracks = [
            track
            for track in tracker.tracks
            if track.is_confirmed() and track.time_since_update == 0
        ]

        if mot_writer is not None:
            mot_writer.write_frame(frame_idx, mot_tracks)

        if cfg.output_video is None and cfg.save_frames_dir is None:
            continue

        annotated = _draw_tracks(frame.copy(), draw_tracks, cfg.box_thickness, cfg.id_offset)
        if cfg.output_video is not None:
            if video_writer is None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(cfg.output_video), fourcc, cfg.frame_rate, (width, height))
            video_writer.write(annotated)

        if cfg.save_frames_dir is not None:
            cv2.imwrite(str(cfg.save_frames_dir / file_name), annotated)

    if video_writer is not None:
        video_writer.release()


def parse_args() -> argparse.Namespace:
    """Define the CLI that layers ad hoc overrides on top of the YAML config."""

    parser = argparse.ArgumentParser(
        description="Run DeepSORT tracking from a config or explicit args.",
        epilog=(
            "Examples:\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml "
            "--model-path /path/to/mars-small128.pb\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml "
            "--max-cosine-distance 0.2 --nn-budget 100"
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
    parser.add_argument("--frames-dir", type=Path, help="Directory containing the source image frames.")
    parser.add_argument("--model-path", type=Path, help="Frozen TensorFlow `.pb` graph used for appearance embeddings.")
    parser.add_argument("--out-mot", type=Path, help="Output MOT file path.")
    parser.add_argument("--out-video", type=Path, help="Output MP4 path.")
    parser.add_argument("--save-frames-dir", type=Path, help="Directory for annotated output frames.")
    parser.add_argument("--frame-rate", type=float, help="Output video FPS.")
    parser.add_argument("--min-confidence", type=float, help="Detection confidence threshold.")
    parser.add_argument("--min-detection-height", type=int, help="Minimum detection bbox height in pixels.")
    parser.add_argument("--nms-max-overlap", type=float, help="Maximum allowed NMS overlap ratio.")
    parser.add_argument("--max-cosine-distance", type=float, help="Appearance metric threshold.")
    parser.add_argument("--nn-budget", type=str, help="Appearance gallery size. Use `None` for unlimited history.")
    parser.add_argument("--max-iou-distance", type=float, help="IoU matching threshold.")
    parser.add_argument("--max-age", type=int, help="Maximum missed frames before deleting a confirmed track.")
    parser.add_argument("--n-init", type=int, help="Frames required to confirm a track.")
    parser.add_argument("--encoder-batch-size", type=int, help="Batch size for the frozen-graph encoder.")
    parser.add_argument("--reid-input-name", type=str, help="Input tensor name inside the frozen graph.")
    parser.add_argument("--reid-output-name", type=str, help="Output tensor name inside the frozen graph.")
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
    if args.model_path is not None:
        updates["model_path"] = _resolve_path(args.model_path)
    if args.out_mot is not None:
        updates["mot_output"] = _resolve_path(args.out_mot)
    if args.out_video is not None:
        updates["output_video"] = _resolve_path(args.out_video)
    if args.save_frames_dir is not None:
        updates["save_frames_dir"] = _resolve_path(args.save_frames_dir)
    if args.frame_rate is not None:
        updates["frame_rate"] = float(args.frame_rate)
    if args.min_confidence is not None:
        updates["min_confidence"] = float(args.min_confidence)
    if args.min_detection_height is not None:
        updates["min_detection_height"] = int(args.min_detection_height)
    if args.nms_max_overlap is not None:
        updates["nms_max_overlap"] = float(args.nms_max_overlap)
    if args.max_cosine_distance is not None:
        updates["max_cosine_distance"] = float(args.max_cosine_distance)
    if args.nn_budget is not None:
        updates["nn_budget"] = _optional_int(args.nn_budget)
    if args.max_iou_distance is not None:
        updates["max_iou_distance"] = float(args.max_iou_distance)
    if args.max_age is not None:
        updates["max_age"] = int(args.max_age)
    if args.n_init is not None:
        updates["n_init"] = int(args.n_init)
    if args.encoder_batch_size is not None:
        updates["encoder_batch_size"] = int(args.encoder_batch_size)
    if args.reid_input_name is not None:
        updates["reid_input_name"] = args.reid_input_name
    if args.reid_output_name is not None:
        updates["reid_output_name"] = args.reid_output_name
    if args.box_thickness is not None:
        updates["box_thickness"] = int(args.box_thickness)
    if args.id_offset is not None:
        updates["id_offset"] = int(args.id_offset)
    return replace(cfg, **updates)


def main() -> None:
    """CLI entrypoint used by local scripts and SLURM launchers."""

    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
