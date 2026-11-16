from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np


FORMAT = "forklift-vision-depth-calibration"
VERSION = 1


def sample_depth_at_point(depth_map: np.ndarray, point: Sequence[int], *,
                          min_depth_m: float = 0.1, max_depth_m: float = 80.0,
                          radius: int = 4, min_valid_pixels: int = 5,
                          min_valid_fraction: float = 0.25,
                          max_relative_mad: float = 0.75) -> dict[str, float] | None:
    if depth_map is None or depth_map.ndim != 2 or depth_map.size == 0 or len(point) != 2:
        return None
    if not all(isinstance(v, (int, np.integer)) for v in point):
        return None
    x, y = int(point[0]), int(point[1])
    height, width = depth_map.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    left, right = max(0, x - radius), min(width, x + radius + 1)
    top, bottom = max(0, y - radius), min(height, y + radius + 1)
    patch = np.asarray(depth_map[top:bottom, left:right], dtype=np.float64)
    valid = np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
    values = patch[valid]
    fraction = float(valid.mean())
    if values.size < min_valid_pixels or fraction < min_valid_fraction:
        return None
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    dispersion = mad * 1.4826
    if median <= 0 or dispersion / median > max_relative_mad:
        return None
    return {"raw_depth_m": median, "valid_fraction": fraction, "dispersion_m": dispersion}


def fit_depth_scale(samples: Iterable[dict[str, Any]]) -> dict[str, float | int | str]:
    items = list(samples)
    if not items:
        raise ValueError("At least one valid distance sample is required")
    ratios = []
    actuals = []
    raw_depths = []
    for sample in items:
        raw_depth = float(sample["raw_depth_m"])
        actual = float(sample["actual_distance_m"])
        if not math.isfinite(raw_depth) or not math.isfinite(actual) or raw_depth <= 0 or actual <= 0:
            raise ValueError("Sample depths and actual distances must be finite and positive")
        raw_depths.append(raw_depth)
        actuals.append(actual)
        ratios.append(actual / raw_depth)
    scale = float(np.median(np.asarray(ratios, dtype=np.float64)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Fitted distance scale must be finite and positive")
    errors = np.abs(np.asarray(raw_depths) * scale - np.asarray(actuals))
    relative_errors = errors / np.asarray(actuals)
    return {
        "method": "median_scale_ratio",
        "scale": scale,
        "sample_count": len(items),
        "mae_m": float(np.mean(errors)),
        "mean_relative_error": float(np.mean(relative_errors)),
    }


def create_distance_calibration(source: str, frame_size: tuple[int, int], model_id: str,
                                samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame width and height must be positive")
    items = []
    for sample in samples:
        pixel = sample.get("pixel")
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            raise ValueError("Each sample requires a pixel coordinate [x, y]")
        x, y = int(pixel[0]), int(pixel[1])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Distance sample point lies outside the calibration frame")
        items.append({
            "pixel": [x, y],
            "raw_depth_m": float(sample["raw_depth_m"]),
            "actual_distance_m": float(sample["actual_distance_m"]),
            "valid_fraction": float(sample.get("valid_fraction", 1.0)),
            "dispersion_m": float(sample.get("dispersion_m", 0.0)),
            "frame_id": int(sample["frame_id"]),
        })
    fit = fit_depth_scale(items)
    return {
        "format": FORMAT,
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "frame_size": {"width": width, "height": height},
        "model_id": model_id,
        "samples": items,
        "fit": fit,
    }


def save_distance_calibration(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=path.suffix,
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_distance_calibration(path: Path, frame_size: tuple[int, int], model_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read distance calibration {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError("Unsupported distance calibration format/version")
    expected_width, expected_height = frame_size
    if payload.get("frame_size") != {"width": expected_width, "height": expected_height}:
        raise ValueError("Distance calibration frame resolution does not match current camera frame")
    legacy_model_id = model_id.removesuffix(":input518") if model_id.endswith(":input518") else None
    if payload.get("model_id") != model_id and not (legacy_model_id is not None and payload.get("model_id") == legacy_model_id):
        raise ValueError("Distance calibration was made for a different depth model/checkpoint/input size")
    if payload.get("model_id") == legacy_model_id and legacy_model_id is not None:
        payload["model_id"] = model_id
        payload["legacy_input_size_assumed"] = 518
    fit = payload.get("fit")
    if not isinstance(fit, dict):
        raise ValueError("Distance calibration is missing fit parameters")
    scale = float(fit.get("scale", float("nan")))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Distance calibration scale must be finite and positive")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Distance calibration must include at least one sample")
    payload["fit"] = fit_depth_scale(samples)
    return payload
