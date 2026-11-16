from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Iterable, Optional, Sequence

import numpy as np


@dataclass
class DepthSample:
    timestamp: float
    raw_depth_m: float
    filtered_depth_m: float
    valid_fraction: float
    dispersion_m: float


@dataclass
class MotionEstimate:
    status: str = "UNKNOWN"
    closing_speed_mps: Optional[float] = None
    experimental_ttc_s: Optional[float] = None


def valid_roi(bbox: Sequence[float], width: int, height: int,
              x_margin: float = 0.15, y_top: float = 0.25,
              y_bottom: float = 0.15) -> Optional[tuple[int, int, int, int]]:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    left = max(0, min(width, int(math.ceil(x1 + bw * x_margin))))
    right = max(0, min(width, int(math.floor(x2 - bw * x_margin))))
    top = max(0, min(height, int(math.ceil(y1 + bh * y_top))))
    bottom = max(0, min(height, int(math.floor(y2 - bh * y_bottom))))
    return (left, top, right, bottom) if right > left and bottom > top else None


def estimate_roi_depth(depth_map: np.ndarray, bbox: Sequence[float], *,
                       min_depth_m: float = 0.1, max_depth_m: float = 20.0,
                       min_pixels: int = 12, min_valid_fraction: float = 0.2,
                       max_relative_mad: float = 0.35, x_margin: float = 0.15,
                       y_top: float = 0.25, y_bottom: float = 0.15) -> Optional[tuple[float, float, float]]:
    if depth_map is None or depth_map.ndim != 2 or depth_map.size == 0:
        return None
    h, w = depth_map.shape
    roi = valid_roi(bbox, w, h, x_margin=x_margin, y_top=y_top, y_bottom=y_bottom)
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    patch = np.asarray(depth_map[y1:y2, x1:x2], dtype=np.float64)
    if patch.size == 0:
        return None
    valid = np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
    fraction = float(valid.mean())
    values = patch[valid]
    if values.size < min_pixels or fraction < min_valid_fraction:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    dispersion = mad * 1.4826
    if median <= 0 or dispersion / median > max_relative_mad:
        return None
    return median, fraction, dispersion


def person_foot_point(bbox: Sequence[float], width: int, height: int) -> Optional[tuple[int, int]]:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    x = int(round((x1 + x2) / 2))
    y = int(math.floor(y2)) - 1
    if x < 0 or x >= width or y < 0 or y >= height:
        return None
    return x, y


def estimate_foot_depth(depth_map: np.ndarray, bbox: Sequence[float], *,
                        min_depth_m: float = 0.1, max_depth_m: float = 20.0,
                        min_pixels: int = 12, min_valid_fraction: float = 0.2,
                        max_relative_mad: float = 0.35) -> Optional[tuple[float, float, float]]:
    return estimate_roi_depth(
        depth_map, bbox, min_depth_m=min_depth_m, max_depth_m=max_depth_m,
        min_pixels=min_pixels, min_valid_fraction=min_valid_fraction,
        max_relative_mad=max_relative_mad, x_margin=0.35, y_top=0.82, y_bottom=0.0,
    )


class DepthFilter:
    def __init__(self, size: int = 5) -> None:
        self.values: Deque[float] = deque(maxlen=max(1, size))

    def update(self, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("depth must be finite and positive")
        self.values.append(value)
        return float(np.median(self.values))


class MotionHistory:
    def __init__(self, *, max_samples: int = 20, timeout_s: float = 3.0,
                 min_samples: int = 3, speed_threshold_mps: float = 0.3,
                 stable_threshold_mps: float = 0.15, min_delta_s: float = 0.05) -> None:
        self.max_samples = max(2, max_samples)
        self.timeout_s = timeout_s
        self.min_samples = max(2, min_samples)
        self.speed_threshold_mps = speed_threshold_mps
        self.stable_threshold_mps = stable_threshold_mps
        self.min_delta_s = min_delta_s
        self.samples: dict[int, Deque[tuple[float, float]]] = {}

    def clear(self) -> None:
        self.samples.clear()

    def cleanup(self, now: float) -> None:
        for ident in list(self.samples):
            history = self.samples[ident]
            while history and now - history[0][0] > self.timeout_s:
                history.popleft()
            if not history:
                del self.samples[ident]

    def update(self, ident: int, timestamp: float, depth_m: Optional[float]) -> MotionEstimate:
        self.cleanup(timestamp)
        if depth_m is None or not math.isfinite(timestamp) or not math.isfinite(depth_m) or depth_m <= 0:
            return MotionEstimate()
        history = self.samples.setdefault(ident, deque(maxlen=self.max_samples))
        if history and timestamp <= history[-1][0]:
            return MotionEstimate()
        history.append((timestamp, depth_m))
        if len(history) < self.min_samples:
            return MotionEstimate()
        times = np.array([p[0] for p in history], dtype=float)
        depths = np.array([p[1] for p in history], dtype=float)
        deltas = np.diff(times)
        if np.any(deltas < self.min_delta_s) or not np.all(np.isfinite(deltas)):
            return MotionEstimate()
        slopes = np.diff(depths) / deltas
        if not np.all(np.isfinite(slopes)):
            return MotionEstimate()
        # Robust median slope: positive means the camera-person range is increasing.
        closing = float(-np.median(slopes))
        if closing >= self.speed_threshold_mps:
            status = "CLOSING"
        elif closing <= -self.speed_threshold_mps:
            status = "RECEDING"
        elif abs(closing) <= self.stable_threshold_mps:
            status = "STABLE"
        else:
            status = "UNKNOWN"
        ttc = float(depth_m / closing) if closing > self.speed_threshold_mps else None
        return MotionEstimate(status, closing, ttc)


@dataclass
class WarningState:
    status: str = "UNKNOWN"
    since: Optional[float] = None


class WarningMachine:
    def __init__(self, warning_distance_m: float = 4.0, critical_distance_m: float = 2.0,
                 hysteresis_m: float = 0.3, debounce_s: float = 0.25) -> None:
        self.warning_distance_m = warning_distance_m
        self.critical_distance_m = critical_distance_m
        self.hysteresis_m = hysteresis_m
        self.debounce_s = debounce_s
        self.state = WarningState()
        self.candidate: Optional[str] = None
        self.candidate_since: Optional[float] = None

    def update(self, timestamp: float, depth_m: Optional[float], *, valid: bool,
               closing: bool = False, degraded: bool = False) -> str:
        if degraded or not valid or depth_m is None or not math.isfinite(depth_m):
            self.state = WarningState("UNKNOWN", timestamp)
            self.candidate = None
            self.candidate_since = None
            return self.state.status
        if depth_m < self.critical_distance_m:
            target = "CRITICAL"
        elif depth_m <= self.warning_distance_m or closing:
            target = "WARNING"
        elif self.state.status == "CRITICAL" and depth_m < self.critical_distance_m + self.hysteresis_m:
            target = "CRITICAL"
        elif self.state.status == "WARNING" and depth_m <= self.warning_distance_m + self.hysteresis_m:
            target = "WARNING"
        else:
            target = "NORMAL"
        if target == self.state.status:
            self.candidate = None
            self.candidate_since = None
            return target
        if target != self.candidate:
            self.candidate = target
            self.candidate_since = timestamp
        elif self.candidate_since is not None and timestamp - self.candidate_since >= self.debounce_s:
            self.state = WarningState(target, timestamp)
            self.candidate = None
            self.candidate_since = None
        return self.state.status


def homography_from_points(image_points: Iterable[Sequence[float]],
                          ground_points: Iterable[Sequence[float]]) -> np.ndarray:
    import cv2
    image = np.asarray(list(image_points), dtype=np.float32)
    ground = np.asarray(list(ground_points), dtype=np.float32)
    if image.shape != (4, 2) or ground.shape != (4, 2) or not np.isfinite(image).all() or not np.isfinite(ground).all():
        raise ValueError("exactly four finite image and ground points are required")
    matrix = cv2.getPerspectiveTransform(image, ground)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("degenerate floor calibration")
    return matrix


def project_ground_point(matrix: np.ndarray, point: Sequence[float]) -> Optional[tuple[float, float]]:
    import cv2
    if matrix.shape != (3, 3) or len(point) != 2 or not np.isfinite(matrix).all():
        return None
    projected = cv2.perspectiveTransform(np.asarray([[point]], dtype=np.float32), matrix)[0, 0]
    if not np.isfinite(projected).all():
        return None
    return float(projected[0]), float(projected[1])


def build_ground_zone_overlay(frame_shape: Sequence[int], matrix: np.ndarray,
                              floor_image_points: Sequence[Sequence[float]],
                              warning_distance_m: float, critical_distance_m: float,
                              *, alpha: float = 0.24) -> tuple[np.ndarray, np.ndarray]:
    """Build a cached BGR zone tint and mask for a calibrated floor polygon.

    Ground coordinates are assumed to use the forklift reference point as (0, 0).
    Colors are BGR: safe green, warning yellow, danger red.
    """
    import cv2
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    points = np.asarray(floor_image_points, dtype=np.float32)
    if height <= 0 or width <= 0 or matrix.shape != (3, 3) or points.shape != (4, 2):
        raise ValueError("invalid frame size, homography, or floor polygon")
    if not np.isfinite(matrix).all() or not np.isfinite(points).all():
        raise ValueError("calibration values must be finite")
    if critical_distance_m <= 0 or warning_distance_m <= critical_distance_m:
        raise ValueError("zone distances must be positive and warning must exceed critical")

    polygon = np.rint(points).astype(np.int32)
    x0 = max(0, int(np.min(polygon[:, 0])))
    y0 = max(0, int(np.min(polygon[:, 1])))
    x1 = min(width, int(np.max(polygon[:, 0])) + 1)
    y1 = min(height, int(np.max(polygon[:, 1])) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("calibrated floor polygon is outside the frame")
    polygon_roi = polygon - np.asarray([x0, y0], dtype=np.int32)
    floor_mask_roi = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(floor_mask_roi, [polygon_roi], 255)

    inverse = np.linalg.inv(matrix)
    rows, cols = np.indices(floor_mask_roi.shape, dtype=np.float32)
    image_x, image_y = cols + x0, rows + y0
    denominator = inverse[2, 0] * image_x + inverse[2, 1] * image_y + inverse[2, 2]
    valid = (floor_mask_roi > 0) & np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
    ground_x = np.zeros(floor_mask_roi.shape, dtype=np.float32)
    ground_y = np.zeros(floor_mask_roi.shape, dtype=np.float32)
    ground_x[valid] = (inverse[0, 0] * image_x[valid] + inverse[0, 1] * image_y[valid] + inverse[0, 2]) / denominator[valid]
    ground_y[valid] = (inverse[1, 0] * image_x[valid] + inverse[1, 1] * image_y[valid] + inverse[1, 2]) / denominator[valid]
    distance = np.hypot(ground_x, ground_y)

    overlay_roi = np.zeros((*floor_mask_roi.shape, 3), dtype=np.uint8)
    safe = valid & (distance > warning_distance_m)
    warning = valid & (distance >= critical_distance_m) & (distance <= warning_distance_m)
    danger = valid & (distance < critical_distance_m)
    overlay_roi[safe] = (40, 160, 40)
    overlay_roi[warning] = (0, 190, 255)
    overlay_roi[danger] = (0, 0, 220)

    # Add clear boundaries around the two distance thresholds.
    for region, color in ((distance < critical_distance_m, (0, 0, 255)),
                          (distance <= warning_distance_m, (0, 220, 255))):
        region_mask = (region & valid).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_roi, contours, -1, color, 2)
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    floor_mask = np.zeros((height, width), dtype=bool)
    overlay[y0:y1, x0:x1] = overlay_roi
    floor_mask[y0:y1, x0:x1] = floor_mask_roi > 0
    return overlay, floor_mask > 0


def blend_ground_zone_overlay(frame: np.ndarray, overlay: np.ndarray,
                              floor_mask: np.ndarray, *, alpha: float = 0.24) -> np.ndarray:
    if frame.shape != overlay.shape or frame.shape[:2] != floor_mask.shape:
        raise ValueError("zone overlay dimensions do not match frame")
    result = frame.copy()
    blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)
    result[floor_mask] = blended[floor_mask]
    return result


def data_is_stale(observed_at: Optional[float], now: float, max_age_s: float) -> bool:
    return observed_at is None or not all(math.isfinite(v) for v in (observed_at, now, max_age_s)) or now < observed_at or now - observed_at > max_age_s
