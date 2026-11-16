from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .logic import estimate_roi_depth

LOGGER = logging.getLogger(__name__)


def environment_report(torch: Any, device: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "os": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(), "pytorch": getattr(torch, "__version__", "unavailable"),
        "cuda_available": False, "gpu": None, "device_used": device,
    }
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["gpu"] = torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError) as exc:
        LOGGER.warning("CUDA inspection failed: %s", exc)
    if sys.platform.startswith("linux"):
        for filename in ("/etc/nv_tegra_release",):
            try:
                text = Path(filename).read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    report["jetson_release"] = text
                    break
            except OSError:
                pass
    return report


def resolve_device(requested: str, torch: Any) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        available = bool(torch.cuda.is_available())
        if requested == "cuda" and not available:
            raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is false")
        if available:
            torch.cuda.init()
            return "cuda"
    except (RuntimeError, AssertionError) as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA initialization failed: {exc}") from exc
        LOGGER.warning("CUDA unavailable; using CPU: %s", exc)
    return "cpu"


class VideoSource:
    def __init__(self, source: str, *, gstreamer: bool = False, retries: int = 5) -> None:
        self.source_text = source
        self.gstreamer = gstreamer
        self.retries = retries
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_live = source.lower().startswith("rtsp://") or source.startswith("/dev/video") or source.isdigit()
        if not source.isdigit() and not source.startswith("/dev/video") and not source.lower().startswith("rtsp://") and not Path(source).is_file():
            raise FileNotFoundError(f"Video file not found: {source}")
        self.source: Any = int(source) if source.isdigit() else source
        self.frame_index = 0
        self.fps = 0.0
        self.reconnected = False
        self._stop_reader = threading.Event()
        self._frame_condition = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_observed: Optional[float] = None
        self._frame_sequence = 0
        self._delivered_sequence = 0
        self._reader_error: Optional[RuntimeError] = None
        self._reader_thread: Optional[threading.Thread] = None
        self.open()
        if self.is_live:
            self._reader_thread = threading.Thread(target=self._read_live_frames, name="video-source-latest-frame", daemon=True)
            self._reader_thread.start()

    def open(self) -> None:
        backend = cv2.CAP_GSTREAMER if self.gstreamer else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(self.source, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Cannot open video source {self.source_text!r}; check camera permissions, URL, OpenCV backend, or GStreamer support")
        if self.is_live:
            # Backends that support this setting can discard buffered frames sooner.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if math.isfinite(fps) and fps > 0 else 0.0

    def _read_live_frames(self) -> None:
        """Continuously drain live input and retain only the newest decoded frame."""
        while not self._stop_reader.is_set():
            capture = self.capture
            if capture is None:
                break
            ok, frame = capture.read()
            if self._stop_reader.is_set():
                return
            if ok and frame is not None and frame.size:
                with self._frame_condition:
                    self._frame_sequence += 1
                    self._latest_frame = frame
                    self._latest_observed = time.monotonic()
                    self._frame_condition.notify_all()
                continue

            capture.release()
            if self.capture is capture:
                self.capture = None
            with self._frame_condition:
                self._latest_frame = None
                self._latest_observed = None
                self._frame_condition.notify_all()

            reconnected = False
            for attempt in range(self.retries):
                if self._stop_reader.wait(min(0.5 * (2 ** attempt), 8.0)):
                    return
                try:
                    self.open()
                    if self._stop_reader.is_set():
                        if self.capture is not None:
                            self.capture.release()
                            self.capture = None
                        return
                    self.frame_index = 0
                    self.reconnected = True
                    reconnected = True
                    LOGGER.warning("Reconnected to live source %s", self.source_text)
                    with self._frame_condition:
                        self._frame_condition.notify_all()
                    break
                except RuntimeError as exc:
                    LOGGER.warning("Reconnect attempt %d/%d failed: %s", attempt + 1, self.retries, exc)
            if not reconnected and not self._stop_reader.is_set():
                with self._frame_condition:
                    self._reader_error = RuntimeError(
                        f"Live source {self.source_text!r} disconnected and exhausted {self.retries} retries"
                    )
                    self._frame_condition.notify_all()
                return

    def read(self) -> tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
        if self.is_live:
            with self._frame_condition:
                if self.reconnected:
                    return None, None, None
                if self._reader_error is not None:
                    raise self._reader_error
                if self._frame_sequence == self._delivered_sequence:
                    self._frame_condition.wait(timeout=0.5)
                if self.reconnected:
                    return None, None, None
                if self._reader_error is not None:
                    raise self._reader_error
                if self._latest_frame is None or self._frame_sequence == self._delivered_sequence:
                    return None, None, None
                self._delivered_sequence = self._frame_sequence
                return self._latest_frame.copy(), None, self._latest_observed
        if self.capture is None:
            return None, None, None
        ok, frame = self.capture.read()
        if ok and frame is not None and frame.size:
            self.frame_index += 1
            media_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            media_t = media_ms / 1000.0 if not self.is_live and math.isfinite(media_ms) and media_ms > 0 else (self.frame_index / self.fps if not self.is_live and self.fps > 0 else None)
            observed_t = time.monotonic()
            return frame, media_t, observed_t
        return None, None, None

    def close(self) -> None:
        self._stop_reader.set()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        with self._frame_condition:
            self._latest_frame = None
            self._latest_observed = None
            self._frame_condition.notify_all()
        if self._reader_thread is not None and self._reader_thread is not threading.current_thread():
            self._reader_thread.join(timeout=1.0)


class DepthEstimator:
    CONFIGS = {
        "indoor": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": 20},
        "outdoor": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "max_depth": 80},
    }

    def __init__(self, model_kind: str, checkpoint: Path, source_dir: Path, device: str, size: int) -> None:
        if model_kind not in self.CONFIGS:
            raise ValueError("depth model must be 'indoor' or 'outdoor'")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Metric depth checkpoint not found: {checkpoint}. Download the matching official Hypersim/VKITTI vits checkpoint as described in README.md")
        source_dir = source_dir.resolve()
        if not (source_dir / "depth_anything_v2" / "dpt.py").is_file():
            raise FileNotFoundError(f"Official Depth Anything V2 metric_depth source is missing: {source_dir}. Clone the official repository and use its metric_depth directory")
        sys.path.insert(0, str(source_dir))
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        self.torch = torch
        self.device = device
        self.size = size
        self.model = DepthAnythingV2(**self.CONFIGS[model_kind])
        try:
            weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        try:
            self.model.load_state_dict(weights, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"Depth checkpoint does not match the {model_kind} vits architecture/configuration: {exc}") from exc
        self.model = self.model.to(device).eval()

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        # Official infer_image handles the model's resize, normalization and BGR convention.
        with self.torch.inference_mode():
            depth = self.model.infer_image(bgr, input_size=self.size)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.shape != bgr.shape[:2]:
            raise ValueError(f"Metric depth output shape {depth.shape} does not match input frame {bgr.shape[:2]}")
        return depth


def load_calibration(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        matrix = np.asarray(content["homography"], dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("homography must be a finite, invertible 3x3 matrix")
        return matrix
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid floor calibration {path}: {exc}") from exc


def load_calibration_image_points(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        points = np.asarray(content["image_points"], dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("image_points must contain exactly four finite (x, y) points")
        return points
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid floor calibration {path}: {exc}") from exc


def run_calibration(source: str, output_path: Path) -> None:
    from .logic import homography_from_points
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open calibration source {source!r}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read a frame for floor calibration")
    points: list[tuple[int, int]] = []
    window = "Floor calibration - click 4 floor points in order, press Enter"
    print("Click four visible floor corners in order around the floor area (clockwise or counterclockwise).")
    print("For coordinates, use the forklift's front reference point as (0, 0), X across the forklift, and Y forward.")
    print("Measure each clicked floor point in meters relative to that reference point.")
    shown = frame.copy()
    def on_click(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        shown = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(shown, pt, 7, (0, 255, 255), -1)
            cv2.putText(shown, str(i + 1), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window, shown)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) == 4:
            break
        if key == ord("q"):
            cv2.destroyWindow(window)
            raise RuntimeError("Floor calibration cancelled")
    cv2.destroyWindow(window)
    physical = []
    for index in range(4):
        try:
            x = float(input(f"Floor point {index + 1} X in meters: "))
            y = float(input(f"Floor point {index + 1} Y in meters: "))
        except ValueError as exc:
            raise ValueError("Ground coordinates must be numeric") from exc
        physical.append((x, y))
    matrix = homography_from_points(points, physical)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"homography": matrix.tolist(), "image_points": points, "ground_points_m": physical}, indent=2), encoding="utf-8")
    LOGGER.info("Saved floor calibration to %s", output_path)
