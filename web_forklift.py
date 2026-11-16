"""Local Forklift Vision AI web UI. No external services or assets are used."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify, make_response, render_template_string, request
import numpy as np

from calibrate_zones import (
    CalibrationError, ResolutionMismatchError, ZONE_COLORS, create_calibration, find_overlaps, safe_source_metadata,
    load_calibration as load_zone_json, make_zone, render_frame,
)
from forklift_vision.distance_calibration import (
    create_distance_calibration, fit_depth_scale, load_distance_calibration,
    sample_depth_at_point, save_distance_calibration,
)
from forklift_vision.logic import DepthFilter, MotionHistory, WarningMachine, data_is_stale, estimate_foot_depth, person_foot_point
from forklift_vision.runtime import DepthEstimator, VideoSource, resolve_device

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("web_forklift")
DEFAULT_SNAPSHOT_MAX_COUNT = 8
DEFAULT_SNAPSHOT_TTL_S = 900.0
DISTANCE_SNAPSHOT_MAX_COUNT = 6
RECENT_FRAME_BUFFER_SIZE = 8
AI_RESOLUTION_PRESETS = (320, 480, 640, 800, 960)


class SnapshotCapacityError(RuntimeError):
    """The configured number of live calibration snapshots is already in use."""


def load_settings(config: Path | None, *, source: str | None = None, model: str | None = None) -> dict[str, Any]:
    if config is None:
        config = Path("config.jetson.json" if Path("/etc/nv_tegra_release").is_file() else "config.json")
    path = config.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    if source is not None:
        settings["source"] = source
    if model is not None:
        settings["yolo_model"] = model
    for key in ("yolo_model", "depth_checkpoint", "depth_source", "calibration_file", "distance_calibration_file"):
        value = settings.get(key)
        if value:
            item = Path(value).expanduser()
            settings[key] = item if item.is_absolute() else ROOT / item
    settings.setdefault("zone_calibration_file", ROOT / "calibration" / "zone_calibration.json")
    settings.setdefault("distance_calibration_file", ROOT / "calibration" / "distance_calibration.json")
    settings["web_settings_file"] = ROOT / "calibration" / "web_settings.json"
    if settings["web_settings_file"].is_file():
        try:
            web_preferences = json.loads(settings["web_settings_file"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read web settings {settings['web_settings_file']}: {exc}") from exc
        mode = web_preferences.get("ai_resolution_mode") if isinstance(web_preferences, dict) else None
        if mode not in {"native", *(str(value) for value in AI_RESOLUTION_PRESETS)}:
            raise ValueError("web_settings.json ai_resolution_mode must be native or a supported AI long edge")
        settings["ai_resolution_mode"] = mode
    else:
        yolo_size = int(settings.get("yolo_size", 640))
        depth_size = int(settings.get("depth_size", yolo_size))
        settings["ai_resolution_mode"] = str(yolo_size) if yolo_size == depth_size else "custom"
    if not isinstance(settings.get("source"), (str, int)):
        raise ValueError("Config must contain source")
    return settings


def validate_zones_payload(payload: Any, width: int, height: int, source: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CalibrationError("JSON body must be an object")
    zones = payload.get("zones")
    if not isinstance(zones, list) or not zones:
        raise CalibrationError("Provide at least one zone")
    checked = []
    for item in zones:
        if not isinstance(item, dict):
            raise CalibrationError("Each zone must be an object")
        name = item.get("name")
        if item.get("color", ZONE_COLORS.get(name)) != ZONE_COLORS.get(name):
            raise CalibrationError(f"Color does not match zone type {name}")
        checked.append(make_zone(name, item.get("points", []), width, height))
    return create_calibration(source, width, height, checked)


def render_clean_reference(frame: np.ndarray, zones: list[dict[str, Any]], people: list[dict[str, Any]] | None = None) -> np.ndarray:
    """Render camera pixels and zones, optionally detector boxes; never browser controls."""
    image = frame.copy()
    overlay = image.copy()
    colors = {"hazard": (0, 0, 255), "warning": (0, 255, 255), "safe": (0, 255, 0)}
    for zone in zones:
        points = np.asarray(zone["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [points], colors[zone["name"]])
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    for zone in zones:
        points = np.asarray(zone["points"], dtype=np.int32).reshape((-1, 1, 2))
        color = colors[zone["name"]]
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        p = tuple(points[0, 0].tolist())
        cv2.putText(image, zone["name"].upper(), p, cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    for person in people or []:
        x1, y1, x2, y2 = person["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return image


def classify_pixel_zone(point: list[int] | tuple[int, int] | None,
                        calibration: dict[str, Any] | None) -> str:
    if calibration is None:
        return "UNCALIBRATED"
    if point is None:
        return "UNKNOWN"
    matches = []
    for zone in calibration.get("zones", []):
        polygon = np.asarray(zone["points"], dtype=np.int32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0:
            matches.append(zone["name"])
    for name in ("hazard", "warning", "safe"):
        if name in matches:
            return name.upper()
    return "OUTSIDE_ZONES"


def render_depth_view(frame: np.ndarray, depth_map: np.ndarray | None, people: list[dict[str, Any]],
                      min_depth_m: float, max_depth_m: float, *, stale: bool = False) -> np.ndarray:
    height, width = frame.shape[:2]
    if depth_map is None or depth_map.shape != (height, width):
        view = np.zeros_like(frame)
        message = "DEPTH MAP UNAVAILABLE"
    else:
        valid = np.isfinite(depth_map) & (depth_map >= min_depth_m) & (depth_map <= max_depth_m)
        normalized = np.zeros((height, width), dtype=np.uint8)
        scaled = (depth_map[valid] - min_depth_m) / max(max_depth_m - min_depth_m, 1e-6)
        normalized[valid] = np.clip(scaled * 255, 0, 255).astype(np.uint8)
        view = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        view[~valid] = (0, 0, 0)
        message = f"RAW DEPTH MAP · camera-axis estimate · {min_depth_m:g}-{max_depth_m:g} m"
        if stale:
            message += " · STALE"
    cv2.putText(view, message, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    for person in people:
        x1, y1, x2, y2 = person["bbox"]
        color = (0, 255, 255) if person.get("depth_m") is not None else (190, 190, 190)
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        foot = person.get("foot_point_px") if person.get("depth_m") is not None else person.get("stale_depth_foot_point_px") or person.get("foot_point_px")
        if foot is not None:
            cv2.drawMarker(view, tuple(foot), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.circle(view, tuple(foot), 5, color, -1)
            if person.get("depth_m") is not None:
                depth_label = f"FOOT ~{person['depth_m']:.2f} m"
            elif person.get("stale_depth_m") is not None:
                depth_label = f"LAST ~{person['stale_depth_m']:.2f} m · STALE {person['stale_depth_age_s']:.1f}s"
            else:
                depth_label = "FOOT UNKNOWN"
            cv2.putText(view, depth_label, (max(4, foot[0] + 8), max(48, foot[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return view


def staged_pair_write(zone_path: Path, calibration: dict[str, Any], reference: np.ndarray) -> Path:
    """Stage and validate both files; rollback the old pair if either replace fails.

    Each rename is atomic on a single filesystem, but two renames are not one
    filesystem transaction. Backups are retained until both replacements finish.
    """
    zone_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path = zone_path.with_name(f"{zone_path.stem}_reference.jpg")
    staged: list[Path] = []
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []

    def stage(suffix: str, data: bytes) -> Path:
        fd, name = tempfile.mkstemp(prefix=f".{zone_path.stem}.", suffix=suffix, dir=zone_path.parent)
        path = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        staged.append(path)
        return path

    try:
        json_bytes = (json.dumps(calibration, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        staged_json = stage(".json.tmp", json_bytes)
        encoded_ok, encoded_image = cv2.imencode(".jpg", reference)
        if not encoded_ok:
            raise OSError("Could not encode reference image")
        staged_ref = stage(".jpg.tmp", encoded_image.tobytes())
        # Validate staged outputs before touching either final file.
        staged_data = load_zone_json(staged_json, (calibration["frame_size"]["width"], calibration["frame_size"]["height"]))
        if staged_data != calibration:
            raise CalibrationError("Staged calibration JSON changed during validation")
        verified_image = cv2.imread(str(staged_ref), cv2.IMREAD_COLOR)
        expected = calibration["frame_size"]
        if verified_image is None or verified_image.shape[1] != expected["width"] or verified_image.shape[0] != expected["height"]:
            raise OSError("Staged reference image failed validation")
        for final in (zone_path, ref_path):
            if final.exists():
                fd, backup_name = tempfile.mkstemp(prefix=f".{zone_path.stem}.backup.", suffix=final.suffix, dir=zone_path.parent)
                os.close(fd)
                backup = Path(backup_name)
                backups[final] = backup
                shutil.copy2(final, backup)
        for temp, final in ((staged_json, zone_path), (staged_ref, ref_path)):
            os.replace(temp, final)
            replaced.append(final)
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                LOG.warning("Could not remove transaction backup %s", backup)
        return ref_path
    except Exception as exc:
        rollback_errors = []
        for final in reversed(replaced):
            try:
                if final in backups:
                    os.replace(backups[final], final)
                else:
                    final.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{final}: {rollback_exc}")
        detail = f"Calibration save failed; previous files were restored where possible: {exc}"
        if rollback_errors:
            detail += "; rollback also failed: " + "; ".join(rollback_errors)
        raise OSError(detail) from exc
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
        for final, backup in backups.items():
            if backup.exists() and final not in replaced:
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    LOG.warning("Could not remove staging backup %s", backup)


def scaled_ai_resolution(width: int, height: int, long_edge: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or long_edge <= 0:
        raise ValueError("Camera dimensions and AI long edge must be positive")
    scale = min(1.0, long_edge / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def ai_resolution_choices(width: int, height: int) -> list[dict[str, Any]]:
    if width <= 0 or height <= 0:
        return []
    native_long_edge = max(width, height)
    choices = []
    for target in AI_RESOLUTION_PRESETS:
        if target <= native_long_edge:
            w, h = scaled_ai_resolution(width, height, target)
            choices.append({"mode": str(target), "width": w, "height": h, "label": f"{w}×{h}"})
    choices.append({"mode": "native", "width": width, "height": height, "label": f"Asli {width}×{height}"})
    return choices


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


class LazyViewDict(dict):
    """Dictionary that lazily renders and encodes camera and depth views on demand."""
    def __init__(self, raw_frame: np.ndarray, display_frame: np.ndarray,
                 depth_map: np.ndarray | None, detections: list[dict[str, Any]],
                 min_depth_m: float, max_depth_m: float, depth_stale: bool,
                 quality: int = 85):
        super().__init__()
        self._raw = raw_frame
        self._ai = display_frame
        self._depth_map = depth_map
        self._detections = detections
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m
        self._depth_stale = depth_stale
        self._quality = quality

        # Eagerly encode AI view (primary stream requested by web clients)
        ok, jpeg = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Could not encode ai view as JPEG")
        super().__setitem__("ai", jpeg.tobytes())

    def __getitem__(self, key: str) -> bytes:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key == "camera":
            ok, jpeg = cv2.imencode(".jpg", self._raw, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                raise RuntimeError("Could not encode camera view as JPEG")
            b = jpeg.tobytes()
            super().__setitem__("camera", b)
            return b
        elif key == "depth":
            depth_frame = render_depth_view(
                self._raw, self._depth_map, self._detections,
                self._min_depth_m, self._max_depth_m,
                stale=self._depth_stale,
            )
            ok, jpeg = cv2.imencode(".jpg", depth_frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                raise RuntimeError("Could not encode depth view as JPEG")
            b = jpeg.tobytes()
            super().__setitem__("depth", b)
            return b
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        return iter(["camera", "ai", "depth"])

    def __len__(self):
        return 3

    def __contains__(self, key: object) -> bool:
        return key in ("camera", "ai", "depth")

    def items(self):
        return [("camera", self["camera"]), ("ai", self["ai"]), ("depth", self["depth"])]

    def keys(self):
        return ["camera", "ai", "depth"]

    def values(self):
        return [self["camera"], self["ai"], self["depth"]]


class CameraPipeline:
    """One worker owns one source and one instance of each model."""
    def __init__(self, settings: dict[str, Any], zone_path: Path, distance_calibration_path: Path | None = None):
        self.settings = settings
        self.ai_resolution_mode = str(settings.get("ai_resolution_mode", "custom"))
        if self.ai_resolution_mode not in ("custom", "native"):
            initial_ai_size = int(self.ai_resolution_mode)
            self.settings["yolo_size"] = initial_ai_size
            self.settings["depth_size"] = initial_ai_size
        self.zone_path = zone_path
        configured_distance_path = settings.get("distance_calibration_file")
        self.distance_calibration_path = distance_calibration_path or (Path(configured_distance_path) if configured_distance_path else ROOT / "calibration" / "distance_calibration.json")
        self.depth_model_id = f"{settings.get('depth_model', 'unknown')}:{Path(settings.get('depth_checkpoint', 'unknown')).name}:input{int(settings.get('depth_size', 518))}"
        self.distance_calibration: dict[str, Any] | None = None
        self.distance_calibration_status = "not_available"
        self.distance_calibration_message = "Distance calibration has not been created"
        self.distance_calibration_snapshots: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.distance_snapshot_ttl_s = float(settings.get("distance_snapshot_ttl_s", 900.0))
        self.distance_snapshot_max_count = int(settings.get("distance_snapshot_max_count", DISTANCE_SNAPSHOT_MAX_COUNT))
        if self.distance_snapshot_ttl_s <= 0 or self.distance_snapshot_max_count < 1:
            raise ValueError("distance snapshot TTL must be > 0 and max count must be >= 1")
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.ai_resolution_changed = threading.Event()
        self.thread: threading.Thread | None = None
        self.source: VideoSource | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_views: dict[str, bytes | None] = {"camera": None, "ai": None, "depth": None}
        self.recent_frame_views: OrderedDict[int, dict[str, bytes]] = OrderedDict()
        self.latest_frame: np.ndarray | None = None
        self.latest_depth_map: np.ndarray | None = None
        self.latest_depth_reference_frame: np.ndarray | None = None
        self.latest_depth_frame_id: int | None = None
        self.latest_depth_reference_people: list[dict[str, Any]] = []
        self.latest_detections: list[dict[str, Any]] = []
        self.updated_at: float | None = None
        self.status: dict[str, Any] = {"camera": "starting", "model": "starting", "state": "starting", "error": None,
                                      "zone_status": "not_available", "zone_message": "Kalibrasi belum tersedia"}
        self.people: list[dict[str, Any]] = []
        self.zones: dict[str, Any] | None = None
        self.calibration_snapshots: dict[str, dict[str, Any]] = {}
        self.snapshot_max_count = int(settings.get("calibration_snapshot_max_count", DEFAULT_SNAPSHOT_MAX_COUNT))
        self.snapshot_ttl_s = float(settings.get("calibration_snapshot_ttl_s", DEFAULT_SNAPSHOT_TTL_S))
        self._last_snapshot_expiry = 0.0
        if self.snapshot_max_count < 1 or self.snapshot_ttl_s <= 0:
            raise ValueError("calibration_snapshot_max_count must be >= 1 and calibration_snapshot_ttl_s must be > 0")
        self._source_close_lock = threading.Lock()
        self._source_closed = False
        self._camera_reader_stuck = False
        self.depth_filters: dict[int, DepthFilter] = {}
        self.last_valid_depth_by_id: dict[int, dict[str, Any]] = {}
        self.motion = MotionHistory(timeout_s=float(settings.get("tracking_timeout_s", 3)), speed_threshold_mps=float(settings.get("closing_speed_threshold_mps", 0.3)))
        self.warning = WarningMachine(float(settings.get("warning_distance_m", 4)), float(settings.get("critical_distance_m", 3)), hysteresis_m=0)
        self.frame_id = 0
        self.device = "unknown"
        self._depth_request_event = threading.Event()
        self._depth_input_lock = threading.Lock()
        self._pending_depth_frame: np.ndarray | None = None
        self._pending_depth_observed: float | None = None
        self._pending_depth_frame_id: int | None = None
        self._pending_depth_size: int = int(settings.get("depth_size", 518))
        self._depth_output_lock = threading.Lock()
        self._cached_depth: np.ndarray | None = None
        self._cached_depth_at: float | None = None
        self._cached_depth_reference_frame: np.ndarray | None = None
        self._cached_depth_frame_id: int | None = None
        self._cached_depth_people: list[dict[str, Any]] = []
        self._depth_thread: threading.Thread | None = None
        self._fps_history: list[float] = []
        self._latest_clean_jpeg: bytes | None = None
        self._latest_depth_jpeg: bytes | None = None

    @property
    def latest_clean_jpeg(self) -> bytes | None:
        if self._latest_clean_jpeg is not None:
            return self._latest_clean_jpeg
        return self.latest_views.get("camera") if self.latest_views else None

    @latest_clean_jpeg.setter
    def latest_clean_jpeg(self, val: bytes | None) -> None:
        self._latest_clean_jpeg = val

    @property
    def latest_depth_jpeg(self) -> bytes | None:
        if self._latest_depth_jpeg is not None:
            return self._latest_depth_jpeg
        return self.latest_views.get("depth") if self.latest_views else None

    @latest_depth_jpeg.setter
    def latest_depth_jpeg(self, val: bytes | None) -> None:
        self._latest_depth_jpeg = val

    def start(self) -> None:
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("Camera pipeline is already running")
            if self.stop_event.is_set():
                raise RuntimeError("Camera pipeline has been shut down and cannot be restarted")
            self.thread = threading.Thread(target=self._run, name="forklift-ai-pipeline", daemon=True)
            self.thread.start()

    def ai_resolution_info(self) -> dict[str, Any]:
        with self.lock:
            frame = self.latest_frame
            camera_height, camera_width = frame.shape[:2] if frame is not None else (None, None)
            yolo_size = int(self.settings.get("yolo_size", 640))
            depth_size = int(self.settings.get("depth_size", 518))
            if camera_width and camera_height:
                yolo_width, yolo_height = scaled_ai_resolution(camera_width, camera_height, min(yolo_size, max(camera_width, camera_height)))
                depth_width, depth_height = scaled_ai_resolution(camera_width, camera_height, min(depth_size, max(camera_width, camera_height)))
                choices = ai_resolution_choices(camera_width, camera_height)
            else:
                yolo_width = yolo_height = depth_width = depth_height = None
                choices = []
            return {
                "mode": self.ai_resolution_mode,
                "camera_size": {"width": camera_width, "height": camera_height} if camera_width else None,
                "yolo_size": {"width": yolo_width, "height": yolo_height, "long_edge": yolo_size} if yolo_width else {"long_edge": yolo_size},
                "depth_size": {"width": depth_width, "height": depth_height, "long_edge": depth_size} if depth_width else {"long_edge": depth_size},
                "choices": choices,
            }

    def set_ai_resolution(self, mode: str) -> dict[str, Any]:
        if mode != "native" and (not mode.isdigit() or int(mode) not in AI_RESOLUTION_PRESETS):
            raise ValueError("AI resolution mode must be native or one of the supported long-edge presets")
        with self.lock:
            frame = self.latest_frame
            camera_height, camera_width = frame.shape[:2] if frame is not None else (None, None)
            if mode == "native":
                target = max(camera_width, camera_height) if camera_width and camera_height else max(int(self.settings.get("yolo_size", 640)), int(self.settings.get("depth_size", 518)))
            else:
                target = int(mode)
                if camera_width and camera_height and target > max(camera_width, camera_height):
                    raise ValueError("AI long edge cannot exceed the original camera resolution")
            web_settings_file = Path(self.settings.get("web_settings_file", ROOT / "calibration" / "web_settings.json"))
            atomic_write_json(web_settings_file, {"ai_resolution_mode": mode})
            self.ai_resolution_mode = mode
            self.settings["ai_resolution_mode"] = mode
            self.settings["yolo_size"] = target
            self.settings["depth_size"] = target
            self.depth_model_id = f"{self.settings.get('depth_model', 'unknown')}:{Path(self.settings.get('depth_checkpoint', 'unknown')).name}:input{target}"
            self.depth_filters.clear()
            self.last_valid_depth_by_id.clear()
            self.motion.clear()
            with self._depth_output_lock:
                self._cached_depth = None
                self._cached_depth_at = None
                self._cached_depth_reference_frame = None
                self._cached_depth_frame_id = None
            self.latest_depth_map = None
            self.latest_depth_reference_frame = None
            self.latest_depth_frame_id = None
            self.latest_depth_reference_people = []
            self.ai_resolution_changed.set()
            self.distance_calibration = None
            if camera_width and camera_height and self.distance_calibration_path.is_file():
                try:
                    self.distance_calibration = load_distance_calibration(self.distance_calibration_path, (camera_width, camera_height), self.depth_model_id)
                    self.distance_calibration_status = "loaded"
                    self.distance_calibration_message = "Loaded legacy distance calibration, assumed Depth Anything input size 518" if self.distance_calibration.get("legacy_input_size_assumed") else "Matching distance calibration loaded"
                except ValueError as exc:
                    self.distance_calibration_status = "invalid"
                    self.distance_calibration_message = str(exc)
            else:
                self.distance_calibration_status = "not_available"
                self.distance_calibration_message = "Recalibrate distance for this AI resolution"
        return self.ai_resolution_info()

    def _publish(self, raw_frame: np.ndarray, display_frame: np.ndarray, detections: list[dict[str, Any]],
                 status: dict[str, Any], depth_map: np.ndarray | None = None,
                 depth_reference_frame: np.ndarray | None = None, depth_frame_id: int | None = None,
                 depth_reference_people: list[dict[str, Any]] | None = None) -> None:
        encoded = LazyViewDict(
            raw_frame, display_frame, depth_map, detections,
            float(self.settings.get("min_depth_m", .1)), float(self.settings.get("max_depth_m", 20)),
            bool(status.get("depth_stale")),
        )
        with self.lock:
            now_monotonic = time.monotonic()
            if now_monotonic - self._last_snapshot_expiry >= 1.0:
                self._expire_calibration_snapshots_locked(now_monotonic)
                self._expire_distance_snapshots_locked(now_monotonic)
                self._last_snapshot_expiry = now_monotonic
            self.latest_frame = raw_frame
            self.latest_depth_map = depth_map
            self.latest_depth_reference_frame = depth_reference_frame
            self.latest_depth_frame_id = depth_frame_id
            self.latest_depth_reference_people = [dict(p) for p in depth_reference_people or []]
            self.latest_views = encoded
            self.latest_jpeg = encoded["ai"]
            self._latest_clean_jpeg = None
            self._latest_depth_jpeg = None
            frame_id = int(status["frame_id"])
            self.recent_frame_views[frame_id] = encoded
            while len(self.recent_frame_views) > RECENT_FRAME_BUFFER_SIZE:
                self.recent_frame_views.popitem(last=False)
            self.updated_at = time.time()
            self.people = detections
            self.latest_detections = detections
            status["available_views"] = list(encoded)
            self.status = status

    def _depth_worker(self, depth: DepthEstimator) -> None:
        depth_interval_s = float(self.settings.get("depth_interval_s", 0.8))
        while not self.stop_event.is_set():
            if not self._depth_request_event.wait(timeout=0.05):
                continue
            self._depth_request_event.clear()
            with self._depth_input_lock:
                req_frame = self._pending_depth_frame
                req_observed = self._pending_depth_observed
                req_frame_id = self._pending_depth_frame_id
                req_size = self._pending_depth_size
                self._pending_depth_frame = None
            if req_frame is None or self.stop_event.is_set():
                continue
            try:
                if depth.size != req_size:
                    depth.size = req_size
                depth_map = depth.infer(req_frame)
                with self._depth_output_lock:
                    self._cached_depth = depth_map
                    self._cached_depth_at = req_observed
                    self._cached_depth_reference_frame = req_frame
                    self._cached_depth_frame_id = req_frame_id
            except Exception as exc:
                LOG.warning("Asynchronous depth inference failed: %s", exc)
            time.sleep(depth_interval_s)

    def view_snapshot(self, view: str, frame_id: int | None = None) -> tuple[bytes | None, dict[str, Any]]:
        with self.lock:
            status = dict(self.status)
            status["people"] = list(self.people)
            status["updated_at"] = self.updated_at
            if self.updated_at is None or time.time() - self.updated_at > 3:
                status.update(state="DEGRADED", message="No fresh frame is available")
                return None, status
            views = self.latest_views if frame_id is None else self.recent_frame_views.get(frame_id)
            if views is None:
                status["requested_frame_id"] = frame_id
                return None, status
            return views.get(view), status

    def _run(self) -> None:
        source = None
        try:
            source_text = str(self.settings["source"]).strip()
            source = VideoSource(source_text, gstreamer=bool(self.settings.get("gstreamer")), retries=int(self.settings.get("reconnect_retries", 5)))
            self.source = source
            self._source_closed = False
            self._camera_reader_stuck = False
            self.status.update(camera="connected", source=safe_source_metadata(source_text), state="running")
            if self.stop_event.is_set():
                return
            yolo = depth = None
            model_messages = []
            try:
                import torch
                from ultralytics import YOLO
                self.device = resolve_device(str(self.settings.get("device", "auto")), torch)
                model_path = Path(self.settings["yolo_model"])
                if not model_path.is_file():
                    raise FileNotFoundError(f"Detector weights missing: {model_path}")
                yolo = YOLO(str(model_path))
            except Exception as exc:
                model_messages.append(f"Detector unavailable: {exc}")
                LOG.warning(model_messages[-1])
            try:
                if self.device == "unknown":
                    import torch
                    self.device = resolve_device(str(self.settings.get("device", "auto")), torch)
                depth = DepthEstimator(self.settings["depth_model"], self.settings["depth_checkpoint"], self.settings["depth_source"], self.device, int(self.settings["depth_size"]))
            except Exception as exc:
                model_messages.append(f"Depth unavailable: {exc}")
                LOG.warning(model_messages[-1])
            self.status.update(model="ready" if yolo else "partial", device=self.device, message="; ".join(model_messages) or None)
            if depth is not None:
                self._depth_thread = threading.Thread(target=self._depth_worker, args=(depth,), name="forklift-depth-worker", daemon=True)
                self._depth_thread.start()
            cached_depth = None
            cached_depth_at = None
            cached_depth_reference_frame = None
            cached_depth_frame_id = None
            cached_depth_people: list[dict[str, Any]] = []
            started = time.monotonic()
            zones_checked_size: tuple[int, int] | None = None
            zone_layer: np.ndarray | None = None
            zone_mask_bool: np.ndarray | None = None
            tracker_name = str(self.settings.get("tracker", "bytetrack.yaml"))
            while not self.stop_event.is_set():
                frame, _, observed = source.read()
                if frame is None:
                    if source.is_live:
                        if source.reconnected:
                            source.reconnected = False
                            self.motion.clear()
                            self.depth_filters.clear()
                            self.last_valid_depth_by_id.clear()
                            cached_depth = cached_depth_at = None
                            cached_depth_reference_frame = None
                            cached_depth_frame_id = None
                            cached_depth_people = []
                            with self.lock:
                                self.latest_jpeg = self.latest_clean_jpeg = self.latest_depth_jpeg = None
                                self.latest_views = {"camera": None, "ai": None, "depth": None}
                                self.recent_frame_views.clear()
                                self.updated_at = None
                                self.status.update(camera="reconnected", state="warming", message="Camera reconnected; waiting for a fresh frame")
                        continue
                    with self.lock:
                        self.status.update(camera="ended", state="ended", message="Video reached end of file")
                    return
                tick = time.perf_counter()
                self.frame_id += 1
                frame_height, frame_width = frame.shape[:2]
                if zones_checked_size != (frame_width, frame_height):
                    zones_checked_size = (frame_width, frame_height)
                    self.zones = None
                    zone_layer = None
                    zone_mask_bool = None
                    if self.zone_path.is_file():
                        try:
                            self.zones = load_zone_json(self.zone_path, (frame_width, frame_height))
                            types = sorted({z["name"] for z in self.zones["zones"]})
                            self.status.update(zone_status="loaded", zone_message="Kalibrasi dimuat", zone_types=types)
                            zone_layer = np.zeros((frame_height, frame_width, 3), np.uint8)
                            zone_mask = np.zeros((frame_height, frame_width), np.uint8)
                            colors = {"hazard": (0, 0, 255), "warning": (0, 255, 255), "safe": (0, 255, 0)}
                            for zone in self.zones["zones"]:
                                pts = np.asarray(zone["points"], np.int32)
                                cv2.fillPoly(zone_layer, [pts], colors[zone["name"]])
                                cv2.fillPoly(zone_mask, [pts], 255)
                            zone_mask_bool = zone_mask > 0
                        except ResolutionMismatchError as exc:
                            self.zones = None
                            self.status.update(zone_status="resolution_mismatch", zone_message=str(exc), zone_types=[])
                        except (CalibrationError, OSError) as exc:
                            self.zones = None
                            self.status.update(zone_status="invalid", zone_message=str(exc), zone_types=[])
                    else:
                        self.status.update(zone_status="not_available", zone_message="Kalibrasi belum tersedia", zone_types=[])
                    self.distance_calibration = None
                    if self.distance_calibration_path.is_file():
                        try:
                            loaded_distance_calibration = load_distance_calibration(self.distance_calibration_path, (frame_width, frame_height), self.depth_model_id)
                            self.distance_calibration = loaded_distance_calibration
                            self.depth_filters.clear()
                            self.last_valid_depth_by_id.clear()
                            self.motion.clear()
                            self.distance_calibration_status = "loaded"
                            self.distance_calibration_message = "Distance calibration loaded"
                        except ValueError as exc:
                            self.distance_calibration_status = "invalid"
                            self.distance_calibration_message = str(exc)
                    else:
                        self.distance_calibration_status = "not_available"
                        self.distance_calibration_message = "No distance calibration file"
                with self.lock:
                    force_depth_inference = self.ai_resolution_changed.is_set()
                    self.ai_resolution_changed.clear()
                    if self.ai_resolution_mode == "native":
                        ai_long_edge = max(frame_width, frame_height)
                        self.settings["yolo_size"] = ai_long_edge
                        self.settings["depth_size"] = ai_long_edge
                    elif self.ai_resolution_mode != "custom":
                        ai_long_edge = min(int(self.ai_resolution_mode), max(frame_width, frame_height))
                        self.settings["yolo_size"] = ai_long_edge
                        self.settings["depth_size"] = ai_long_edge
                    yolo_size = int(self.settings.get("yolo_size", 640))
                    depth_size = int(self.settings.get("depth_size", 518))
                    updated_model_id = f"{self.settings.get('depth_model', 'unknown')}:{Path(self.settings.get('depth_checkpoint', 'unknown')).name}:input{depth_size}"
                    model_input_changed = updated_model_id != self.depth_model_id
                    if model_input_changed:
                        self.depth_model_id = updated_model_id
                        self.distance_calibration = None
                        if self.distance_calibration_path.is_file():
                            try:
                                self.distance_calibration = load_distance_calibration(self.distance_calibration_path, (frame_width, frame_height), self.depth_model_id)
                                self.distance_calibration_status = "loaded"
                                self.distance_calibration_message = "Loaded legacy distance calibration, assumed Depth Anything input size 518" if self.distance_calibration.get("legacy_input_size_assumed") else "Matching distance calibration loaded"
                            except ValueError as exc:
                                self.distance_calibration_status = "invalid"
                                self.distance_calibration_message = str(exc)
                        else:
                            self.distance_calibration_status = "not_available"
                            self.distance_calibration_message = "Recalibrate distance for this AI resolution"
                    if depth is not None and depth.size != depth_size:
                        force_depth_inference = True
                with self._depth_output_lock:
                    if self._cached_depth is not None:
                        cached_depth = self._cached_depth
                        cached_depth_at = self._cached_depth_at
                        cached_depth_reference_frame = self._cached_depth_reference_frame
                        cached_depth_frame_id = self._cached_depth_frame_id

                if force_depth_inference:
                    with self._depth_output_lock:
                        self._cached_depth = self._cached_depth_at = None
                        self._cached_depth_reference_frame = None
                        self._cached_depth_frame_id = None
                    cached_depth = cached_depth_at = None
                    cached_depth_reference_frame = None
                    cached_depth_frame_id = None
                    cached_depth_people = []
                    self.depth_filters.clear()
                    self.last_valid_depth_by_id.clear()
                    self.motion.clear()

                if depth is not None and not self._depth_request_event.is_set():
                    with self._depth_input_lock:
                        self._pending_depth_frame = frame.copy()
                        self._pending_depth_observed = observed
                        self._pending_depth_frame_id = self.frame_id
                        self._pending_depth_size = depth_size
                        self._depth_request_event.set()
                result = yolo.track(frame, persist=True, tracker=tracker_name, classes=[0], conf=float(self.settings.get("confidence", 0.35)), imgsz=yolo_size, device=self.device, verbose=False)[0] if yolo is not None else None
                annotated = frame.copy()
                people: list[dict[str, Any]] = []
                stale = data_is_stale(cached_depth_at, observed, float(self.settings.get("max_depth_age_s", 1)))
                depth_age_s = max(0.0, float(observed - cached_depth_at)) if cached_depth_at is not None and observed is not None else None
                depth_history_timeout = float(self.settings.get("tracking_timeout_s", 3))
                for track_id in list(self.last_valid_depth_by_id):
                    if observed - self.last_valid_depth_by_id[track_id]["observed_at"] > depth_history_timeout:
                        self.last_valid_depth_by_id.pop(track_id, None)
                frame_distance_calibration = self.distance_calibration
                boxes = result.boxes if result is not None else None
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    scores = boxes.conf.cpu().numpy()
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                    for box, confidence, ident in zip(xyxy, scores, ids):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1, x2 = max(0, x1), min(frame_width - 1, x2)
                        y1, y2 = max(0, y1), min(frame_height - 1, y2)
                        foot_point = person_foot_point([x1, y1, x2, y2], frame_width, frame_height)
                        depth_value = None
                        raw_depth_value = None
                        if depth is None:
                            depth_status = "DEPTH_MODEL_UNAVAILABLE"
                        elif stale or cached_depth is None:
                            depth_status = "DEPTH_STALE_OR_NOT_READY"
                        elif foot_point is None:
                            depth_status = "FOOT_POINT_INVALID"
                        else:
                            measured = estimate_foot_depth(cached_depth, [x1, y1, x2, y2], min_depth_m=float(self.settings.get("min_depth_m", .1)), max_depth_m=float(self.settings.get("max_depth_m", 20)))
                            if measured:
                                raw_depth_value = measured[0]
                                calibration_scale = float(frame_distance_calibration["fit"]["scale"]) if frame_distance_calibration else 1.0
                                calibrated_depth = raw_depth_value * calibration_scale
                                depth_value = self.depth_filters.setdefault(int(ident), DepthFilter()).update(calibrated_depth) if ident is not None else calibrated_depth
                                depth_status = "VALID_CALIBRATED" if frame_distance_calibration else "VALID_UNCALIBRATED"
                                if ident is not None:
                                    self.last_valid_depth_by_id[int(ident)] = {
                                        "depth_m": depth_value, "raw_depth_m": raw_depth_value,
                                        "observed_at": float(observed), "frame_id": self.frame_id,
                                        "foot_point_px": list(foot_point),
                                    }
                            else:
                                depth_status = "FOOT_DEPTH_INVALID_OR_UNSTABLE"
                        stale_estimate = self.last_valid_depth_by_id.get(int(ident)) if ident is not None and depth_value is None else None
                        stale_estimate_age_s = max(0.0, float(observed) - stale_estimate["observed_at"]) if stale_estimate is not None else None
                        motion = self.motion.update(int(ident), float(observed), depth_value) if ident is not None and depth_value is not None else None
                        motion_status = motion.status if motion else "UNKNOWN"
                        warning = self.warning.update(float(observed), depth_value, valid=depth_value is not None, closing=motion_status == "CLOSING")
                        zone_status = classify_pixel_zone(foot_point, self.zones)
                        zone_warning = {"HAZARD": "CRITICAL", "WARNING": "WARNING"}.get(zone_status)
                        if zone_warning == "CRITICAL" or (zone_warning == "WARNING" and warning not in ("CRITICAL",)):
                            warning = zone_warning
                        person = {"id": ident, "confidence": float(confidence), "bbox": [x1, y1, x2, y2], "foot_point_px": list(foot_point) if foot_point else None, "zone": zone_status, "depth_m": depth_value, "depth_status": depth_status, "stale_depth_m": stale_estimate["depth_m"] if stale_estimate else None, "stale_depth_age_s": stale_estimate_age_s, "stale_depth_frame_id": stale_estimate["frame_id"] if stale_estimate else None, "stale_depth_foot_point_px": stale_estimate["foot_point_px"] if stale_estimate else None, "motion": motion_status, "closing_speed_mps": motion.closing_speed_mps if motion else None, "ttc_s": motion.experimental_ttc_s if motion else None, "warning": warning}
                        people.append(person)
                        color = {"CRITICAL": (0, 0, 255), "WARNING": (0, 220, 255), "NORMAL": (0, 200, 0), "UNKNOWN": (128, 128, 128)}[warning]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        if foot_point is not None:
                            cv2.drawMarker(annotated, foot_point, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                            cv2.circle(annotated, foot_point, 4, color, -1)
                        distance = f"KAKI ~{depth_value:.2f}m" if depth_value is not None else f"KAKI UNKNOWN ({depth_status})"
                        cv2.putText(annotated, f"ID {ident if ident is not None else '?'} {confidence:.2f} {distance} {zone_status} {motion_status}/{warning}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 2)
                if cached_depth_frame_id == self.frame_id:
                    cached_depth_people = [dict(person) for person in people]
                elapsed_ms = (time.perf_counter() - tick) * 1000
                stale_frame = (time.monotonic() - (observed or time.monotonic())) > 5
                state = "DEGRADED" if stale or stale_frame or elapsed_ms > float(self.settings.get("latency_limit_ms", 1500)) or yolo is None or depth is None else "running"
                if zone_mask_bool is not None and zone_layer is not None:
                    blended = cv2.addWeighted(annotated, .7, zone_layer, .3, 0)
                    annotated[zone_mask_bool] = blended[zone_mask_bool]
                model_message = "; ".join(model_messages) or ("Depth measurement is stale" if stale else None)
                now_mono = time.monotonic()
                self._fps_history.append(now_mono)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                if len(self._fps_history) > 1:
                    dt = self._fps_history[-1] - self._fps_history[0]
                    current_fps = (len(self._fps_history) - 1) / dt if dt > 0 else 0.0
                else:
                    current_fps = self.frame_id / max(now_mono - started, 0.001)
                status = {"camera": "connected", "model": "ready" if yolo is not None and depth is not None else "partial", "detector_status": "ready" if yolo is not None else "unavailable", "depth_model_status": "ready" if depth is not None else "unavailable", "model_message": "; ".join(model_messages) or None, "device": self.device, "state": state, "error": None, "fps": round(current_fps, 1), "latency_ms": round(elapsed_ms, 1), "depth_stale": stale, "depth_age_s": depth_age_s, "max_depth_age_s": float(self.settings.get("max_depth_age_s", 1)), "frame_size": {"width": frame_width, "height": frame_height}, "frame_id": self.frame_id, "message": model_message,
                          "zone_status": self.status.get("zone_status", "not_available"), "zone_message": self.status.get("zone_message", "Kalibrasi belum tersedia"), "zone_types": self.status.get("zone_types", [])}
                status["distance_calibration_status"] = self.distance_calibration_status
                status["distance_calibration_message"] = self.distance_calibration_message
                status["distance_calibration_scale"] = self.distance_calibration["fit"]["scale"] if self.distance_calibration else None
                status["distance_calibration_samples"] = len(self.distance_calibration["samples"]) if self.distance_calibration else 0
                status["distance_calibration_file"] = str(self.distance_calibration_path.resolve())
                self._publish(frame, annotated, people, status, cached_depth, cached_depth_reference_frame,
                              cached_depth_frame_id, cached_depth_people)
        except Exception as exc:
            LOG.exception("Pipeline failed")
            with self.lock:
                self.status.update(state="ERROR", error=str(exc), message="Pipeline failed; no live frame is being served")
                self.latest_jpeg = self.latest_clean_jpeg = self.latest_depth_jpeg = None
                self.latest_views = {"camera": None, "ai": None, "depth": None}
                self.recent_frame_views.clear()
                self.updated_at = None
        finally:
            self._depth_request_event.set()
            if self._depth_thread and self._depth_thread.is_alive():
                self._depth_thread.join(timeout=1.0)
            if source is not None:
                self._close_source(source)
            self.source = None

    def _close_source(self, source: VideoSource) -> None:
        with self._source_close_lock:
            if self._source_closed:
                return
            self._source_closed = True
        try:
            source.close()
        except Exception:
            LOG.exception("Camera source cleanup failed")
        reader = getattr(source, "_reader_thread", None)
        if reader is not None and reader.is_alive():
            self._camera_reader_stuck = True
            message = "Camera reader thread is still blocked after VideoSource.close; it was not force-stopped."
            LOG.error(message)
            with self.lock:
                self.status.update(shutdown_status="camera_reader_still_running", shutdown_message=message)

    def _expire_distance_snapshots_locked(self, now: float) -> None:
        expired = [sid for sid, snapshot in self.distance_calibration_snapshots.items()
                   if now - snapshot.get("_created_monotonic", float("-inf")) >= self.distance_snapshot_ttl_s]
        for sid in expired:
            self.distance_calibration_snapshots.pop(sid, None)

    @staticmethod
    def _depth_map_from_snapshot(snapshot: dict[str, Any]) -> np.ndarray:
        width = int(snapshot["frame_size"]["width"])
        height = int(snapshot["frame_size"]["height"])
        raw = zlib.decompress(snapshot["depth_map_zlib"])
        return np.frombuffer(raw, dtype=np.dtype(snapshot["depth_dtype"])).reshape(height, width)

    def add_distance_calibration_sample(self, snapshot_id: str, point: Sequence[int], actual_distance_m: float) -> dict[str, Any]:
        if not math.isfinite(actual_distance_m) or actual_distance_m <= 0:
            raise ValueError("Actual distance must be a finite positive number of meters")
        snapshot = self.get_distance_calibration_snapshot(snapshot_id)
        if snapshot is None:
            raise LookupError("Distance calibration frame expired; capture a new depth frame")
        result = sample_depth_at_point(
            self._depth_map_from_snapshot(snapshot), point,
            min_depth_m=float(self.settings.get("min_depth_m", .1)),
            max_depth_m=float(self.settings.get("max_depth_m", 20)),
        )
        if result is None:
            raise ValueError("No stable valid depth around that pixel; choose a clear point on the person’s feet or floor")
        return {
            "snapshot_id": snapshot_id, "frame_id": snapshot["frame_id"],
            "pixel": [int(point[0]), int(point[1])],
            "raw_depth_m": result["raw_depth_m"], "actual_distance_m": actual_distance_m,
            "valid_fraction": result["valid_fraction"], "dispersion_m": result["dispersion_m"],
        }

    def snapshot(self) -> tuple[bytes | None, dict[str, Any]]:
        with self.lock:
            status = dict(self.status)
            status["people"] = list(self.people)
            status["updated_at"] = self.updated_at
            if self.updated_at is not None and time.time() - self.updated_at > 3:
                status.update(state="DEGRADED", message="No fresh frame for over 3 seconds")
                return None, status
            return self.latest_jpeg, status

    def capture_calibration_snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            now = time.monotonic()
            self._expire_calibration_snapshots_locked(now)
            if len(self.calibration_snapshots) >= self.snapshot_max_count:
                raise SnapshotCapacityError(
                    f"Snapshot limit reached ({self.snapshot_max_count}); save or cancel an existing calibration session first"
                )
            if self.latest_frame is None or self.updated_at is None or time.time() - self.updated_at > 3:
                return None
            raw = self.latest_frame
            ok, encoded = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return None
            height, width = raw.shape[:2]
            snapshot_id = uuid.uuid4().hex
            self.calibration_snapshots[snapshot_id] = {
                "jpeg": encoded.tobytes(), "people": [dict(p) for p in self.latest_detections],
                "captured_at": datetime.now(timezone.utc).isoformat(), "frame_id": self.status.get("frame_id"),
                "frame_size": {"width": width, "height": height}, "source": str(self.settings["source"]), "snapshot_id": snapshot_id,
                "_created_monotonic": now,
            }
            return {k: v for k, v in self.calibration_snapshots[snapshot_id].items() if k not in {"frame", "jpeg", "_created_monotonic"}}

    def _expire_calibration_snapshots_locked(self, now: float) -> int:
        expired = [sid for sid, snap in self.calibration_snapshots.items()
                   if now - snap.get("_created_monotonic", float("-inf")) >= self.snapshot_ttl_s]
        for sid in expired:
            self.calibration_snapshots.pop(sid, None)
        return len(expired)

    def _get_calibration_snapshot_locked(self, snapshot_id: str | None) -> dict[str, Any] | None:
        now = time.monotonic()
        self._expire_calibration_snapshots_locked(now)
        return self.calibration_snapshots.get(snapshot_id or "")

    def get_calibration_snapshot(self, snapshot_id: str | None) -> tuple[bytes | None, dict[str, Any] | None]:
        with self.lock:
            snapshot = self._get_calibration_snapshot_locked(snapshot_id)
            if snapshot is None:
                return None, None
            return snapshot["jpeg"], {k: v for k, v in snapshot.items() if k not in {"frame", "jpeg", "_created_monotonic"}}

    def cancel_calibration_snapshot(self, snapshot_id: str | None) -> None:
        with self.lock:
            self._expire_calibration_snapshots_locked(time.monotonic())
            self.calibration_snapshots.pop(snapshot_id or "", None)

    def capture_distance_calibration_snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            now = time.monotonic()
            self._expire_distance_snapshots_locked(now)
            if len(self.distance_calibration_snapshots) >= self.distance_snapshot_max_count:
                raise SnapshotCapacityError("Distance calibration snapshot limit reached; save samples or wait for an old session to expire")
            frame = self.latest_depth_reference_frame
            depth_map = self.latest_depth_map
            frame_id = self.latest_depth_frame_id
            if frame is None or depth_map is None or frame_id is None or depth_map.shape != frame.shape[:2]:
                return None
            ok, frame_jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("Could not encode frame used by depth inference")
            depth_preview = render_depth_view(
                frame, depth_map, self.latest_depth_reference_people,
                float(self.settings.get("min_depth_m", .1)), float(self.settings.get("max_depth_m", 20)),
            )
            ok, depth_jpeg = cv2.imencode(".jpg", depth_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Could not encode distance calibration depth preview")
            height, width = frame.shape[:2]
            snapshot_id = uuid.uuid4().hex
            snapshot = {
                "depth_map_zlib": zlib.compress(memoryview(np.ascontiguousarray(depth_map)).cast("B"), level=1),
                "depth_dtype": depth_map.dtype.str, "jpeg": frame_jpeg.tobytes(), "depth_jpeg": depth_jpeg.tobytes(),
                "frame_id": int(frame_id), "frame_size": {"width": width, "height": height},
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "source": safe_source_metadata(str(self.settings["source"])),
                "model_id": self.depth_model_id,
                "people": [dict(person) for person in self.latest_depth_reference_people],
                "snapshot_id": snapshot_id, "_created_monotonic": now,
            }
            self.distance_calibration_snapshots[snapshot_id] = snapshot
            return {key: value for key, value in snapshot.items() if key not in {"depth_map_zlib", "depth_dtype", "jpeg", "depth_jpeg", "_created_monotonic"}}

    def get_distance_calibration_snapshot(self, snapshot_id: str | None) -> dict[str, Any] | None:
        with self.lock:
            self._expire_distance_snapshots_locked(time.monotonic())
            return self.distance_calibration_snapshots.get(snapshot_id or "")

    def sample_distance_calibration_point(self, snapshot_id: str, point: Sequence[int], actual_distance_m: float) -> dict[str, Any]:
        if not math.isfinite(actual_distance_m) or actual_distance_m <= 0:
            raise ValueError("Actual distance must be finite and greater than zero")
        snapshot = self.get_distance_calibration_snapshot(snapshot_id)
        if snapshot is None:
            raise LookupError("Distance calibration frame expired or is unavailable; capture a new frame")
        sample = sample_depth_at_point(
            self._depth_map_from_snapshot(snapshot), point,
            min_depth_m=float(self.settings.get("min_depth_m", .1)),
            max_depth_m=float(self.settings.get("max_depth_m", 20)),
        )
        if sample is None:
            raise ValueError("No stable depth at this point; select a visible person foot or another clear point")
        return {
            **sample, "snapshot_id": snapshot_id, "frame_id": snapshot["frame_id"],
            "model_id": snapshot["model_id"], "frame_size": snapshot["frame_size"],
            "pixel": [int(point[0]), int(point[1])], "actual_distance_m": float(actual_distance_m),
        }

    def save_distance_calibration(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Add at least one measured calibration point")
        with self.lock:
            self._expire_distance_snapshots_locked(time.monotonic())
            verified_samples = []
            decoded_depth_maps: dict[str, np.ndarray] = {}
            reference_snapshot = None
            for item in samples:
                snapshot_id = item.get("snapshot_id")
                snapshot = self.distance_calibration_snapshots.get(snapshot_id)
                if snapshot is None:
                    raise LookupError("A sample frame expired; capture it again and re-enter its measured point")
                if snapshot["model_id"] != self.depth_model_id:
                    raise ValueError("Sample depth model does not match the active model")
                if reference_snapshot is None:
                    reference_snapshot = snapshot
                elif (snapshot["frame_size"] != reference_snapshot["frame_size"]
                      or snapshot["model_id"] != reference_snapshot["model_id"]):
                    raise ValueError("All calibration samples must use the same resolution and depth model")
                actual = float(item["actual_distance_m"])
                point = item["pixel"]
                if snapshot_id not in decoded_depth_maps:
                    decoded_depth_maps[snapshot_id] = self._depth_map_from_snapshot(snapshot)
                measured = sample_depth_at_point(
                    decoded_depth_maps[snapshot_id], point,
                    min_depth_m=float(self.settings.get("min_depth_m", .1)),
                    max_depth_m=float(self.settings.get("max_depth_m", 20)),
                )
                if measured is None:
                    raise ValueError(f"Sample at pixel {point} no longer has valid depth")
                verified_samples.append({
                    **measured, "pixel": [int(point[0]), int(point[1])],
                    "actual_distance_m": actual, "frame_id": snapshot["frame_id"],
                })
            calibration = create_distance_calibration(
                reference_snapshot["source"],
                (reference_snapshot["frame_size"]["width"], reference_snapshot["frame_size"]["height"]),
                self.depth_model_id, verified_samples,
            )
            save_distance_calibration(self.distance_calibration_path, calibration)
            self.distance_calibration = calibration
            self.depth_filters.clear()
            self.last_valid_depth_by_id.clear()
            self.motion.clear()
            self.distance_calibration_status = "loaded"
            self.distance_calibration_message = "Distance correction saved and applied"
            return calibration

    def shutdown(self, timeout: float = 5.0) -> bool:
        self.stop_event.set()
        self._depth_request_event.set()
        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join(timeout=1.0)
        if self.source:
            self._close_source(self.source)
        if self.thread and self.thread.ident is not None:
            self.thread.join(timeout=max(0.0, timeout))
            if self.thread.is_alive():
                message = "Pipeline worker did not stop before timeout; camera inference/read may be blocked. Thread was not force-stopped."
                LOG.error(message)
                with self.lock:
                    self.status.update(shutdown_status="worker_still_running", shutdown_message=message)
                return False
        with self.lock:
            if self._camera_reader_stuck:
                self.status.update(shutdown_status="camera_reader_still_running")
                return False
            self.status.update(shutdown_status="stopped")
        return True


PAGE = r'''<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Forklift Vision AI</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --panel2: #f1f5f9;
      --line: #e2e8f0;
      --line-strong: #cbd5e1;
      --text: #0f172a;
      --muted: #64748b;
      --blue: #2563eb;
      --red: #dc2626;
      --yellow: #d97706;
      --green: #16a34a;
      --focus: #3b82f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      gap: 20px;
      padding: 0 clamp(16px, 3vw, 36px);
      height: 56px;
      background: rgba(255, 255, 255, 0.95);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: -0.01em;
      color: var(--text);
      text-decoration: none;
    }
    .brand-mark {
      display: inline-grid;
      place-items: center;
      width: 28px;
      height: 28px;
      border-radius: 6px;
      background: #0f172a;
      color: #ffffff;
      font-size: 0.8rem;
      font-weight: 800;
    }
    nav {
      display: flex;
      gap: 4px;
      background: #f1f5f9;
      padding: 3px;
      border-radius: 8px;
      border: 1px solid var(--line);
    }
    nav a {
      padding: 5px 12px;
      color: var(--muted);
      text-decoration: none;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      transition: all 0.12s ease;
    }
    nav a:hover {
      color: var(--text);
    }
    nav a:focus-visible {
      background: #ffffff;
      color: var(--text);
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    #global {
      margin-left: auto;
      padding: 4px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 600;
      background: #f8fafc;
    }
    main {
      width: min(1560px, 100%);
      margin: auto;
      padding: clamp(16px, 2.5vw, 28px);
    }
    h1, h2, h3, p { margin-top: 0; }
    h1 {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
      margin-bottom: 2px;
    }
    h2 {
      font-size: 0.95rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: var(--text);
      margin-bottom: 12px;
    }
    h3 {
      font-size: 0.88rem;
      font-weight: 600;
      margin-bottom: 6px;
    }
    .subtle, .small {
      color: var(--muted);
      font-size: 0.82rem;
    }
    .card {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: clamp(14px, 1.8vw, 20px);
      box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
    }
    .stack { display: grid; gap: 14px; }
    .row { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
    .spread { justify-content: space-between; }
    
    .grid-monitor {
      display: grid;
      grid-template-columns: minmax(0, 2.2fr) minmax(300px, 0.9fr);
      gap: 16px;
      align-items: start;
    }
    .grid-cal {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(320px, 0.85fr);
      gap: 16px;
      align-items: start;
    }
    .distance-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.8fr);
      gap: 16px;
      align-items: start;
    }
    .distance-grid canvas {
      display: block;
      width: 100%;
      height: auto;
      touch-action: none;
      cursor: crosshair;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
    }
    .distance-samples {
      padding: 0;
      margin: 0;
      list-style: none;
      max-height: 260px;
      overflow-y: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .distance-samples li {
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 0.82rem;
      background: #ffffff;
      line-height: 1.4;
    }
    .distance-samples li:last-child { border-bottom: 0; }
    .distance-sample-remove {
      min-height: 26px;
      padding: 2px 8px;
      margin-left: 8px;
      font-size: 0.75rem;
    }
    
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
    }
    .metric {
      padding: 9px 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.05em;
      margin-bottom: 3px;
    }
    .metric strong {
      display: block;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
      font-size: 1.08rem;
      font-weight: 700;
      color: var(--text);
    }
    
    .state-pill, .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 9px;
      border-radius: 6px;
      background: #f1f5f9;
      border: 1px solid var(--line);
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--muted);
    }
    .state-pill::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--muted);
    }
    .state-pill.running {
      background: #f0fdf4;
      border-color: #bbf7d0;
      color: #166534;
    }
    .state-pill.running::before { background: var(--green); }
    .state-pill.degraded, .state-pill.error {
      background: #fef2f2;
      border-color: #fecaca;
      color: #991b1b;
    }
    .state-pill.degraded::before, .state-pill.error::before { background: var(--red); }
    
    .snapshot-badge { color: #1e40af; border-color: #bfdbfe; background: #eff6ff; }
    .dirty-badge { color: #92400e; border-color: #fde68a; background: #fffbeb; }
    .clean-badge { color: #166534; border-color: #bbf7d0; background: #f0fdf4; }
    
    .camera {
      display: block;
      width: 100%;
      max-height: 74vh;
      aspect-ratio: 4/3;
      object-fit: contain;
      background: #0f172a;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
    }
    .camera-empty {
      display: grid;
      place-items: center;
      min-height: 320px;
      color: var(--muted);
      background: #f8fafc;
      border: 2px dashed #cbd5e1;
      border-radius: 8px;
      text-align: center;
      padding: 24px;
      font-size: 0.88rem;
    }
    
    .person {
      padding: 10px 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      font-size: 0.84rem;
      line-height: 1.45;
      margin-bottom: 8px;
    }
    .person strong { font-variant-numeric: tabular-nums; color: var(--text); }
    
    .zone-chip {
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      margin: 3px 0;
      border-radius: 4px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.03em;
      border: 1px solid transparent;
    }
    .zone-chip.zone-hazard { background: #fef2f2; border-color: #fecaca; color: #dc2626; }
    .zone-chip.zone-warning { background: #fffbeb; border-color: #fde68a; color: #d97706; }
    .zone-chip.zone-safe { background: #f0fdf4; border-color: #bbf7d0; color: #16a34a; }
    .zone-chip.zone-outside_zones, .zone-chip.zone-uncalibrated, .zone-chip.zone-unknown {
      background: #f1f5f9; border-color: var(--line); color: var(--muted);
    }
    
    .notice {
      padding: 10px 14px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-left: 4px solid var(--muted);
      color: var(--text);
      font-size: 0.84rem;
      line-height: 1.45;
    }
    .notice.error {
      background: #fef2f2;
      border-color: #fecaca;
      border-left-color: var(--red);
      color: var(--red);
    }
    .notice.success {
      background: #f0fdf4;
      border-color: #bbf7d0;
      border-left-color: var(--green);
      color: var(--green);
    }
    
    .path {
      overflow-wrap: anywhere;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.78rem;
    }
    
    button, select, input[type=number] {
      min-height: 36px;
      padding: 7px 12px;
      border-radius: 6px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      touch-action: manipulation;
      transition: all 0.12s ease;
    }
    button:hover:not(:disabled) {
      background: #f8fafc;
      border-color: #94a3b8;
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    button:focus-visible, select:focus-visible, input:focus-visible {
      outline: none;
      border-color: var(--focus);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
    }
    
    button.primary, button.save-action {
      background: #0f172a;
      border-color: #0f172a;
      color: #ffffff;
    }
    button.primary:hover:not(:disabled), button.save-action:hover:not(:disabled) {
      background: #1e293b;
      border-color: #1e293b;
      color: #ffffff;
    }
    
    button.muted-button {
      background: #ffffff;
      border-color: var(--line);
      color: var(--muted);
    }
    button.muted-button:hover:not(:disabled) {
      background: #f8fafc;
      color: var(--text);
    }
    
    .danger-action {
      color: var(--red);
      border-color: #fecaca;
      background: #fef2f2;
    }
    .danger-action:hover:not(:disabled) {
      background: #fee2e2;
      border-color: var(--red);
    }
    
    button[data-zone] {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-weight: 600;
    }
    button[data-zone][aria-pressed="true"] {
      box-shadow: 0 0 0 2px #ffffff, 0 0 0 4px #0f172a;
    }
    button[data-zone="hazard"][aria-pressed="true"] { background: var(--red); border-color: var(--red); color: #fff; }
    button[data-zone="warning"][aria-pressed="true"] { background: var(--yellow); border-color: var(--yellow); color: #fff; }
    button[data-zone="safe"][aria-pressed="true"] { background: var(--green); border-color: var(--green); color: #fff; }
    
    .checklist {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0;
    }
    .check-item {
      padding: 9px 10px;
      border: 1px solid var(--line);
      background: #f8fafc;
      border-radius: 8px;
      min-width: 0;
      font-size: 0.8rem;
    }
    .check-item.complete {
      border-color: #bbf7d0;
      background: #f0fdf4;
      color: #166534;
    }
    .check-item.missing {
      border-color: #fde68a;
      background: #fffbeb;
      color: #92400e;
    }
    .check-item strong { display: block; }
    .zone-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 7px;
      border-radius: 50%;
      vertical-align: middle;
    }
    .dot-hazard { background: var(--red); }
    .dot-warning { background: var(--yellow); }
    .dot-safe { background: var(--green); }
    
    .zone-list {
      display: grid;
      gap: 6px;
      max-height: 38vh;
      overflow-y: auto;
    }
    .zone-entry {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      width: 100%;
      text-align: left;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 0.82rem;
    }
    .zone-entry[aria-pressed="true"] {
      border-color: var(--blue);
      background: #eff6ff;
      font-weight: 600;
    }
    .zone-entry-label { min-width: 0; overflow-wrap: anywhere; }
    .point-count { white-space: nowrap; color: var(--muted); font-size: 0.75rem; font-variant-numeric: tabular-nums; }
    
    .canvas-wrap {
      position: relative;
      background: #0f172a;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      overflow: hidden;
    }
    .canvas-wrap canvas {
      display: block;
      width: 100%;
      height: auto;
      touch-action: none;
      cursor: crosshair;
    }
    .canvas-caption {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 6px 4px;
      color: var(--muted);
      font-size: 0.78rem;
    }
    
    .save-panel {
      position: sticky;
      bottom: 12px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid var(--line);
      border-radius: 10px;
      backdrop-filter: blur(8px);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .save-panel button { width: 100%; min-height: 44px; font-size: 0.95rem; }
    .keyboard-hint {
      padding: 9px 12px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      font-size: 0.8rem;
    }
    
    dialog {
      width: min(600px, calc(100% - 28px));
      padding: 20px;
      background: #ffffff;
      color: var(--text);
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    dialog::backdrop {
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(2px);
    }
    dialog h2 { font-size: 1.15rem; }
    .dialog-actions { display: grid; gap: 8px; margin-top: 16px; }
    .dialog-actions button { text-align: left; min-height: 44px; }
    .dialog-actions .close-only { justify-self: end; min-width: 100px; text-align: center; }
    
    @media (max-width: 1050px) {
      .grid-monitor, .grid-cal, .distance-grid { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(3, minmax(100px, 1fr)); }
      .save-panel { position: static; }
    }
    @media (max-width: 600px) {
      header { gap: 8px; flex-wrap: wrap; height: auto; padding: 10px 14px; }
      header nav { order: 3; width: 100%; }
      nav a { flex: 1; text-align: center; }
      #global { font-size: 0.75rem; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .checklist { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand"><span class="brand-mark" aria-hidden="true">FV</span>Forklift Vision AI</div>
    <nav aria-label="Navigasi utama"><a href="/">Monitoring</a><a href="/calibration">Settings</a></nav>
    <span id="global" class="state-pill" role="status">Menghubungkan…</span>
  </header>
  <main id="app"></main>
  <dialog id="snapshotDialog" aria-labelledby="snapshotDialogTitle">
    <h2 id="snapshotDialogTitle">Ganti frame kalibrasi?</h2>
    <p id="snapshotDialogText"></p>
    <div class="dialog-actions">
      <button type="button" class="keep" data-snapshot-choice="keep">Lanjutkan tanpa mengganti snapshot</button>
      <button type="button" class="discard" data-snapshot-choice="discard">Mulai kalibrasi baru dan buang perubahan</button>
      <button type="button" class="preserve" data-snapshot-choice="preserve">Ganti frame, pertahankan semua poligon</button>
      <button type="button" class="close-only muted-button" data-snapshot-choice="keep">Batal</button>
    </div>
  </dialog>
  <script>
    const app=document.querySelector('#app');
    let pollTimer;
    const zoneKinds={
      hazard:{label:'Bahaya',color:'#ef5350',json:'#FF0000',dot:'dot-hazard'},
      warning:{label:'Waspada',color:'#f6c445',json:'#FFFF00',dot:'dot-warning'},
      safe:{label:'Aman',color:'#4dc28a',json:'#00FF00',dot:'dot-safe'}
    };
    function esc(value){return String(value).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
    function zoneText(s){
      const labels={hazard:'Bahaya',warning:'Waspada',safe:'Aman'};
      const counts=s.zone_counts||{hazard:0,warning:0,safe:0};
      const complete=Object.values(counts).every(n=>n>0);
      const state={not_available:'Kalibrasi belum tersedia',invalid:'File kalibrasi invalid',resolution_mismatch:'Resolusi kalibrasi tidak cocok',loaded:'Kalibrasi dimuat'}[s.zone_status]||'Status kalibrasi belum diketahui';
      const tally=Object.keys(labels).map(k=>`${labels[k]} ${counts[k]||0}`).join(' · ');
      return `<div class="notice ${s.zone_status==='invalid'||s.zone_status==='resolution_mismatch'?'error':''}"><strong>${state}${s.zone_status==='loaded'&&!complete?' · BELUM LENGKAP':''}</strong><br>${esc(s.zone_message||'')}<br>Jumlah poligon: ${esc(tally)}<br><span class="path">File kalibrasi aktif: ${esc(s.zone_calibration||'—')}</span></div>${s.lan_warning?'<div class="notice error" role="alert"><strong>Peringatan LAN:</strong> siapa pun yang dapat menjangkau server ini dapat melihat video dan mengubah kalibrasi.</div>':''}`;
    }
    function monitoring(){
      app.innerHTML=`<div class="stack">
        <section class="card stack" aria-labelledby="monitorTitle"><div class="row spread"><div><h1 id="monitorTitle">Monitoring lingkungan</h1><p class="subtle">Status kamera, pipeline, dan estimasi deteksi terbaru.</p></div><span id="pipelineState" class="state-pill">Menunggu status</span></div>
          <div class="metrics" id="metrics"></div><div id="message" aria-live="polite"></div><div id="zonestatus"></div>
        </section>
        <div class="grid-monitor"><section class="card"><div class="row spread"><h2>Tampilan video</h2><label class="row" for="frameView">Pilih view <select id="frameView" aria-label="Pilih tampilan kamera atau AI"><option value="camera">Kamera asli</option><option value="ai" selected>Deteksi AI</option><option value="depth">Peta kedalaman AI</option></select></label><span id="frameStamp" class="badge">Belum ada frame</span></div><img class="camera" id="camera" alt="Tampilan live Forklift Vision AI"><div id="cameraEmpty" class="camera-empty">Menunggu frame kamera terbaru…</div><p id="viewHint" class="small">Deteksi AI menampilkan bounding box, zona, dan estimasi di area kaki.</p></section>
          <section class="card"><div class="row spread"><h2>Orang terdeteksi</h2><div class="row"><span id="personCount" class="badge">0</span><span id="personFrameStamp" class="badge">Belum ada frame</span></div></div><div id="people" aria-live="polite"></div></section></div>
        <section class="notice" role="note">Depth, motion, warning, dan TTC adalah estimasi prototipe, bukan pengukuran keselamatan tervalidasi. Jangan gunakan untuk pengereman atau interlock.</section>
      </div>`;
      let camera=document.querySelector('#camera');const viewSelect=document.querySelector('#frameView');
      const viewHints={camera:'Kamera asli tanpa anotasi AI.',ai:'Deteksi AI: bounding box, zona, estimasi depth area kaki.',depth:'Peta kedalaman AI: warna hanya visualisasi; tanda silang menunjukkan titik kaki yang dirujuk.'};
      let updateSequence=0,updateInProgress=false,refreshQueued=false,displayedImageUrl=null,displayedFrameId=null,displayedView=null;
      camera.style.display='none';
      viewSelect.addEventListener('change',()=>{document.querySelector('#viewHint').textContent=viewHints[viewSelect.value];if(updateInProgress)updateSequence++;update();});
      function renderStatus(s){
        const global=document.querySelector('#global'),state=document.querySelector('#pipelineState');
        const pipelineState=s.error?'error':(s.state==='running'?'running':'degraded');
        global.textContent=s.state||'unknown';global.className=`state-pill ${pipelineState}`;
        state.textContent=s.error?'ERROR':(s.state||'UNKNOWN');state.className=`state-pill ${pipelineState}`;
        const dims=s.frame_size?`${s.frame_size.width}×${s.frame_size.height}`:'—';
        const metrics={Kamera:s.camera||'unknown',Model:s.model||'unknown',DepthAI:s.depth_model_status||'unknown',UmurDepth:s.depth_age_s==null?'—':`${Number(s.depth_age_s).toFixed(1)} / ${Number(s.max_depth_age_s||1).toFixed(1)} s${s.depth_stale?' · STALE':''}`,Perangkat:s.device||'—',FPS:s.fps==null?'—':Number(s.fps).toFixed(1),Latensi:s.latency_ms==null?'—':`${s.latency_ms} ms`,Resolusi:dims};
        document.querySelector('#metrics').innerHTML=Object.entries(metrics).map(([k,v])=>`<div class="metric"><label>${esc(k)}</label><strong>${esc(v)}</strong></div>`).join('');
        const message=document.querySelector('#message');
        message.innerHTML=s.error?`<div class="notice error" role="alert">${esc(s.error)}</div>`:(s.message?`<div class="notice">${esc(s.message)}</div>`:'');
        document.querySelector('#zonestatus').innerHTML=zoneText(s);
        const persons=s.people||[],zoneNames={HAZARD:'BAHAYA',WARNING:'WASPADA',SAFE:'AMAN',OUTSIDE_ZONES:'DI LUAR ZONA',UNCALIBRATED:'BELUM DIKALIBRASI',UNKNOWN:'UNKNOWN'};
        const distanceText=person=>person.depth_m!=null?`~${Number(person.depth_m).toFixed(2)} m`:person.stale_depth_m!=null?`~${Number(person.stale_depth_m).toFixed(2)} m · STALE ${Number(person.stale_depth_age_s).toFixed(1)} s (frame ${person.stale_depth_frame_id})`:'UNKNOWN';
        document.querySelector('#personCount').textContent=String(persons.length);document.querySelector('#personFrameStamp').textContent=`Deteksi frame ${s.frame_id??'—'}`;
        document.querySelector('#people').innerHTML=persons.map(p=>{const zoneKey=(p.zone||'unknown').toLowerCase();return `<div class="person"><strong>Person ${esc(p.id??'?')}</strong> · confidence ${Number(p.confidence||0).toFixed(2)}<br><span class="zone-chip zone-${esc(zoneKey)}">ZONA ${esc(zoneNames[p.zone]||p.zone||'UNKNOWN')}</span><br><strong>Estimasi jarak:</strong> ${distanceText(p)}<br>Titik kaki (piksel): ${p.foot_point_px?`${p.foot_point_px[0]}, ${p.foot_point_px[1]}`:'UNKNOWN'}</div>`;}).join('')||'<p class="subtle">Belum ada person terdeteksi.</p>';
        return dims;
      }
      function clearDisplayedFrame(text){
        camera.style.display='none';camera.removeAttribute('src');
        if(displayedImageUrl){URL.revokeObjectURL(displayedImageUrl);displayedImageUrl=null;}
        displayedFrameId=null;displayedView=null;
        const empty=document.querySelector('#cameraEmpty');empty.style.display='grid';empty.textContent=text;
      }
      async function update(){
        if(updateInProgress){refreshQueued=true;return;}
        updateInProgress=true;
        const sequence=++updateSequence;
        try{
          const s=await(await fetch('/api/status',{cache:'no-store'})).json();
          if(sequence!==updateSequence)return;
          if(!s.frame_available||s.frame_id==null){const dims=renderStatus({...s,people:[]});clearDisplayedFrame(s.error||s.message||'Tidak ada frame live terbaru');document.querySelector('#frameStamp').textContent='Tidak ada frame';return;}
          const selectedView=viewSelect.value,frameId=Number(s.frame_id);
          if(displayedFrameId===frameId&&displayedView===selectedView){const dims=renderStatus(s);document.querySelector('#frameStamp').textContent=`Sinkron · ${viewSelect.options[viewSelect.selectedIndex].text} · frame ${frameId} · ${dims}`;return;}
          const response=await fetch(`/api/frame.jpg?view=${encodeURIComponent(selectedView)}&frame_id=${encodeURIComponent(frameId)}`,{cache:'no-store'});
          if(sequence!==updateSequence)return;
          if(!response.ok){const detail=await response.json().catch(()=>({}));throw new Error(detail.error||`Frame ${frameId} tidak tersedia`);}
          const responseFrameId=Number(response.headers.get('X-Frame-ID'));
          if(responseFrameId!==frameId)throw new Error(`ID frame tidak cocok: status ${frameId}, gambar ${responseFrameId}`);
          const imageUrl=URL.createObjectURL(await response.blob());
          const nextImage=new Image();nextImage.className='camera';nextImage.id='camera';nextImage.alt='Tampilan live Forklift Vision AI';nextImage.src=imageUrl;
          try{await nextImage.decode();}catch(error){URL.revokeObjectURL(imageUrl);throw new Error(`Gambar frame ${frameId} gagal didekode`);}
          if(sequence!==updateSequence){URL.revokeObjectURL(imageUrl);return;}
          const oldImageUrl=displayedImageUrl;
          camera.replaceWith(nextImage);camera=nextImage;displayedImageUrl=imageUrl;displayedFrameId=frameId;displayedView=selectedView;
          const dims=renderStatus(s);camera.style.display='block';document.querySelector('#cameraEmpty').style.display='none';
          document.querySelector('#frameStamp').textContent=`Sinkron · ${viewSelect.options[viewSelect.selectedIndex].text} · frame ${frameId} · ${dims}`;
          if(oldImageUrl)URL.revokeObjectURL(oldImageUrl);
        }catch(error){
          if(sequence!==updateSequence)return;
          document.querySelector('#message').innerHTML=`<div class="notice error" role="alert">${esc(error.message||'Sinkronisasi frame gagal')}</div>`;
          if(displayedFrameId===null){const empty=document.querySelector('#cameraEmpty');empty.style.display='grid';empty.textContent='Belum ada frame yang tersinkron dengan data deteksi.';}
        }finally{
          updateInProgress=false;
          if(refreshQueued){refreshQueued=false;update();}
          else if(!document.hidden){setTimeout(update,30);}
        }
      }
      update();pollTimer=setInterval(update,1000);
    }
    function distanceCalibrationUI(){
      const canvas=document.querySelector('#distanceCanvas'),ctx=canvas.getContext('2d');
      const actualInput=document.querySelector('#actualDistance'),addButton=document.querySelector('#distanceAdd'),viewSelect=document.querySelector('#distanceView');
      let activeSnapshot=null,cameraImage=new Image(),depthImage=new Image(),width=0,height=0,selectedPoint=null,samples=[];
      const setMessage=(text,kind='')=>{const element=document.querySelector('#distanceMessage');element.textContent=text;element.className=`notice ${kind}`;};
      function fitPreview(){
        const target=document.querySelector('#distanceFitPreview');
        if(!samples.length){target.textContent='Belum cukup sampel';document.querySelector('#distanceSave').disabled=true;return;}
        const ratios=samples.map(sample=>sample.actual_distance_m/sample.raw_depth_m).sort((a,b)=>a-b);
        const mid=Math.floor(ratios.length/2),scale=ratios.length%2?ratios[mid]:(ratios[mid-1]+ratios[mid])/2;
        const errors=samples.map(sample=>Math.abs(sample.raw_depth_m*scale-sample.actual_distance_m));
        const mae=errors.reduce((sum,value)=>sum+value,0)/errors.length;
        target.textContent=`×${scale.toFixed(3)} · ${samples.length} titik · MAE sampel ${mae.toFixed(2)} m`;
        document.querySelector('#distanceSave').disabled=false;
      }
      function renderSamples(){
        const list=document.querySelector('#distanceSampleList');
        list.innerHTML=samples.map((sample,index)=>`<li><b>Titik ${index+1}</b> · frame ${sample.frame_id} · piksel ${sample.pixel[0]},${sample.pixel[1]}<br>Depth mentah ${sample.raw_depth_m.toFixed(2)} m · aktual ${sample.actual_distance_m.toFixed(2)} m · valid ${(sample.valid_fraction*100).toFixed(0)}% <button type="button" class="distance-sample-remove danger-action" data-distance-remove="${index}" aria-label="Hapus titik ${index+1}">Hapus</button></li>`).join('')||'<li class="subtle">Belum ada titik ukur.</li>';
        fitPreview();draw();
      }
      function draw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        const image=viewSelect.value==='depth'?depthImage:cameraImage;
        if(!activeSnapshot||!image.complete||!width||!height)return;
        ctx.drawImage(image,0,0,width,height);
        samples.forEach((sample,index)=>{if(sample.snapshot_id!==activeSnapshot.snapshot_id)return;const [x,y]=sample.pixel;ctx.fillStyle='#3ee28e';ctx.strokeStyle='#06120b';ctx.lineWidth=Math.max(2,width/500);ctx.beginPath();ctx.arc(x,y,Math.max(7,width/140),0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font=`700 ${Math.max(16,width/55)}px sans-serif`;ctx.fillText(String(index+1),x+9,y-8);});
        if(selectedPoint){ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(2,width/500);ctx.beginPath();ctx.arc(selectedPoint[0],selectedPoint[1],Math.max(8,width/100),0,Math.PI*2);ctx.stroke();ctx.fillStyle='#fff';ctx.fillText('PILIHAN',selectedPoint[0]+10,selectedPoint[1]-8);}
      }
      function canvasPoint(event){const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(width-1,Math.round((event.clientX-rect.left)*width/rect.width))),Math.max(0,Math.min(height-1,Math.round((event.clientY-rect.top)*height/rect.height)))];}
      function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Gambar kalibrasi tidak dapat dimuat'));image.src=url;});}
      async function refreshStatus(){
        try{const status=await(await fetch('/api/status',{cache:'no-store'})).json();const scale=status.distance_calibration_scale;document.querySelector('#distanceCalibrationStatus').textContent=status.distance_calibration_status==='loaded'?`Aktif · faktor ×${Number(scale).toFixed(3)} · ${status.distance_calibration_samples} sampel`:status.distance_calibration_message||'Belum dikalibrasi';document.querySelector('#distanceCalibrationPath').textContent=`File: ${status.distance_calibration_file||'—'}`;}
        catch(error){document.querySelector('#distanceCalibrationStatus').textContent='Status kalibrasi tidak tersedia';}
      }
      async function captureFrame(){
        const response=await fetch('/api/distance-calibration/snapshot',{method:'POST'}),data=await response.json();
        if(!response.ok)return setMessage(data.error||'Depth map belum tersedia. Pastikan Depth AI aktif.','error');
        if(samples.length&&(samples[0].model_id!==data.model_id||samples[0].frame_size.width!==data.frame_size.width||samples[0].frame_size.height!==data.frame_size.height)){
          if(!confirm('Frame baru memakai resolusi/model berbeda. Hapus semua sampel lama dan mulai kalibrasi baru?')){await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(data.snapshot_id)}`,{method:'DELETE'});return;}
          const ids=[...new Set(samples.map(sample=>sample.snapshot_id))];for(const id of ids)await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(id)}`,{method:'DELETE'});
          samples=[];renderSamples();
        }
        const suffix=`snapshot_id=${encodeURIComponent(data.snapshot_id)}&t=${Date.now()}`;
        try{[cameraImage,depthImage]=await Promise.all([loadImage(`/api/distance-calibration/snapshot.jpg?${suffix}`),loadImage(`/api/distance-calibration/depth.jpg?${suffix}`)]);}catch(error){await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(data.snapshot_id)}`,{method:'DELETE'});return setMessage(error.message,'error');}
        activeSnapshot=data;width=data.frame_size.width;height=data.frame_size.height;canvas.width=width;canvas.height=height;selectedPoint=null;addButton.disabled=true;
        document.querySelector('#distanceFrameInfo').textContent=`Depth input frame ${data.frame_id} · ${width}×${height} · ${data.model_id}`;
        document.querySelector('#distanceCalibrationStatus').textContent='Snapshot depth aktif';draw();setMessage('Klik titik kaki/person yang jaraknya sudah diukur, lalu masukkan nilai aktual.','success');
      }
      async function addSample(){
        if(!activeSnapshot)return setMessage('Ambil frame depth terlebih dahulu.','error');
        if(!selectedPoint)return setMessage('Pilih titik pada frame terlebih dahulu.','error');
        const actual=Number(actualInput.value);
        if(!Number.isFinite(actual)||actual<=0)return setMessage('Masukkan jarak aktual positif dalam meter.','error');
        const response=await fetch('/api/distance-calibration/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snapshot_id:activeSnapshot.snapshot_id,pixel:selectedPoint,actual_distance_m:actual})});
        const sample=await response.json();if(!response.ok)return setMessage(sample.error||'Titik tidak memiliki depth valid.','error');
        samples.push(sample);selectedPoint=null;actualInput.value='';addButton.disabled=true;renderSamples();setMessage(`Titik ${samples.length} ditambahkan dari frame ${sample.frame_id}.`,'success');
      }
      async function saveCalibration(){
        if(!samples.length)return setMessage('Tambahkan minimal satu titik ukur.','error');
        const response=await fetch('/api/distance-calibration/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({samples})});
        const result=await response.json();if(!response.ok)return setMessage(result.error||'Kalibrasi jarak gagal disimpan.','error');
        setMessage(`Tersimpan · faktor ×${Number(result.fit.scale).toFixed(3)} · ${result.sample_count} titik · MAE sampel ${Number(result.fit.mae_m).toFixed(2)} m · ${result.path}`,'success');
        await refreshStatus();
      }
      canvas.onclick=event=>{if(!activeSnapshot)return setMessage('Ambil frame depth terlebih dahulu.','error');selectedPoint=canvasPoint(event);addButton.disabled=false;draw();setMessage(`Titik (${selectedPoint[0]}, ${selectedPoint[1]}) dipilih. Masukkan jarak aktual lalu tambahkan.`);};
      viewSelect.onchange=draw;
      actualInput.oninput=()=>{addButton.disabled=!activeSnapshot||!selectedPoint||!Number.isFinite(Number(actualInput.value))||Number(actualInput.value)<=0;};
      document.querySelector('#distanceCapture').onclick=()=>captureFrame().catch(error=>setMessage(`Gagal mengambil depth frame: ${error.message}`,'error'));
      addButton.onclick=()=>addSample().catch(error=>setMessage(`Gagal menambah titik: ${error.message}`,'error'));
      document.querySelector('#distanceSave').onclick=()=>saveCalibration().catch(error=>setMessage(`Gagal menyimpan kalibrasi: ${error.message}`,'error'));
      document.querySelector('#distanceSampleList').onclick=event=>{const button=event.target.closest('[data-distance-remove]');if(!button)return;samples.splice(Number(button.dataset.distanceRemove),1);renderSamples();};
      renderSamples();refreshStatus();
    }

    function distanceCalibrationUI(){
      const canvas=document.querySelector('#distanceCanvas'),ctx=canvas.getContext('2d');
      const actualInput=document.querySelector('#actualDistance'),addButton=document.querySelector('#distanceAdd'),viewSelect=document.querySelector('#distanceView');
      let activeSnapshot=null,cameraImage=new Image(),depthImage=new Image(),width=0,height=0,selectedPoint=null,samples=[];
      const setMessage=(text,kind='')=>{const element=document.querySelector('#distanceMessage');element.textContent=text;element.className=`notice ${kind}`;};
      function refreshFit(){
        const target=document.querySelector('#distanceFitPreview'),save=document.querySelector('#distanceSave');
        if(!samples.length){target.textContent='Belum ada sampel';save.disabled=true;return;}
        const ratios=samples.map(sample=>sample.actual_distance_m/sample.raw_depth_m).sort((a,b)=>a-b);
        const middle=Math.floor(ratios.length/2),scale=ratios.length%2?ratios[middle]:(ratios[middle-1]+ratios[middle])/2;
        const mae=samples.reduce((sum,sample)=>sum+Math.abs(sample.raw_depth_m*scale-sample.actual_distance_m),0)/samples.length;
        target.textContent=`×${scale.toFixed(3)} · ${samples.length} titik · MAE pada sampel ${mae.toFixed(2)} m`;
        save.disabled=false;
      }
      function draw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        const base=viewSelect.value==='depth'?depthImage:cameraImage;
        if(!activeSnapshot||!base.complete||!width||!height)return;
        ctx.drawImage(base,0,0,width,height);
        if(viewSelect.value==='camera')for(const person of activeSnapshot.people||[]){const [x1,y1,x2,y2]=person.bbox;ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(2,width/500);ctx.strokeRect(x1,y1,x2-x1,y2-y1);}
        samples.forEach((sample,index)=>{if(sample.snapshot_id!==activeSnapshot.snapshot_id)return;const [x,y]=sample.pixel;ctx.fillStyle='#38e58c';ctx.strokeStyle='#07130b';ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,Math.max(7,width/140),0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font=`700 ${Math.max(16,width/55)}px sans-serif`;ctx.fillText(String(index+1),x+9,y-8);});
        if(selectedPoint){ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(2,width/500);ctx.beginPath();ctx.arc(selectedPoint[0],selectedPoint[1],Math.max(8,width/100),0,Math.PI*2);ctx.stroke();}
      }
      function refreshSamples(){
        document.querySelector('#distanceSampleList').innerHTML=samples.map((sample,index)=>`<li><b>Titik ${index+1}</b> · frame ${sample.frame_id} · piksel ${sample.pixel[0]}, ${sample.pixel[1]}<br>Depth mentah ${sample.raw_depth_m.toFixed(2)} m · aktual ${sample.actual_distance_m.toFixed(2)} m · valid ${(sample.valid_fraction*100).toFixed(0)}% <button type="button" class="distance-sample-remove danger-action" data-distance-remove="${index}">Hapus</button></li>`).join('')||'<li class="subtle">Belum ada titik ukur.</li>';
        refreshFit();draw();
      }
      function framePoint(event){const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(width-1,Math.round((event.clientX-rect.left)*width/rect.width))),Math.max(0,Math.min(height-1,Math.round((event.clientY-rect.top)*height/rect.height)))];}
      function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Gagal memuat gambar kalibrasi'));image.src=url;});}
      async function refreshDistanceStatus(){
        try{const status=await(await fetch('/api/status',{cache:'no-store'})).json();const scale=status.distance_calibration_scale;document.querySelector('#distanceCalibrationStatus').textContent=status.distance_calibration_status==='loaded'?`Koreksi aktif ×${Number(scale).toFixed(3)} · ${status.distance_calibration_samples} sampel`:status.distance_calibration_message||'Belum dikalibrasi';document.querySelector('#distanceCalibrationPath').textContent=`File: ${status.distance_calibration_file||'—'}`;}
        catch(error){document.querySelector('#distanceCalibrationStatus').textContent='Status kalibrasi tidak tersedia';}
      }
      async function captureDistanceFrame(){
        const response=await fetch('/api/distance-calibration/snapshot',{method:'POST'}),candidate=await response.json();
        if(!response.ok)return setMessage(candidate.error||'Depth map belum tersedia. Pastikan Depth AI berjalan.','error');
        if(samples.length){const first=samples[0],same=first.model_id===candidate.model_id&&first.frame_size.width===candidate.frame_size.width&&first.frame_size.height===candidate.frame_size.height;if(!same){if(!confirm('Resolusi atau model depth berbeda. Hapus sampel sebelumnya dan mulai kalibrasi baru?')){await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(candidate.snapshot_id)}`,{method:'DELETE'});return;}for(const id of new Set(samples.map(sample=>sample.snapshot_id)))await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(id)}`,{method:'DELETE'});samples=[];refreshSamples();}}
        const oldSnapshot=activeSnapshot;
        const query=`snapshot_id=${encodeURIComponent(candidate.snapshot_id)}&t=${Date.now()}`;
        try{[cameraImage,depthImage]=await Promise.all([loadImage(`/api/distance-calibration/snapshot.jpg?${query}`),loadImage(`/api/distance-calibration/depth.jpg?${query}`)]);}catch(error){await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(candidate.snapshot_id)}`,{method:'DELETE'});return setMessage(error.message,'error');}
        if(oldSnapshot&&!samples.some(sample=>sample.snapshot_id===oldSnapshot.snapshot_id))await fetch(`/api/distance-calibration/snapshot?snapshot_id=${encodeURIComponent(oldSnapshot.snapshot_id)}`,{method:'DELETE'});
        activeSnapshot=candidate;width=candidate.frame_size.width;height=candidate.frame_size.height;canvas.width=width;canvas.height=height;selectedPoint=null;addButton.disabled=true;
        document.querySelector('#distanceFrameInfo').textContent=`Depth input frame ${candidate.frame_id} · ${width}×${height} · ${candidate.model_id}`;
        draw();setMessage('Frame dan depth map cocok. Klik titik kaki/person, masukkan jarak ukur aktual, lalu tambah titik.','success');
      }
      async function addDistanceSample(){
        if(!activeSnapshot)return setMessage('Ambil frame depth terlebih dahulu.','error');
        if(!selectedPoint)return setMessage('Klik titik pada frame terlebih dahulu.','error');
        const actual=Number(actualInput.value);
        if(!Number.isFinite(actual)||actual<=0)return setMessage('Jarak aktual harus lebih besar dari nol meter.','error');
        const response=await fetch('/api/distance-calibration/sample',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snapshot_id:activeSnapshot.snapshot_id,pixel:selectedPoint,actual_distance_m:actual})});
        const sample=await response.json();if(!response.ok)return setMessage(sample.error||'Depth di titik ini invalid atau tidak stabil.','error');
        samples.push(sample);selectedPoint=null;actualInput.value='';addButton.disabled=true;refreshSamples();setMessage(`Titik ${samples.length} berhasil ditambahkan dari frame ${sample.frame_id}.`,'success');
      }
      async function saveDistanceCalibration(){
        if(!samples.length)return setMessage('Tambahkan minimal satu titik kalibrasi.','error');
        const response=await fetch('/api/distance-calibration/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({samples})});
        const result=await response.json();if(!response.ok)return setMessage(result.error||'Kalibrasi jarak gagal disimpan.','error');
        setMessage(`Koreksi diterapkan · faktor ×${Number(result.fit.scale).toFixed(3)} · ${result.sample_count} titik · MAE sampel ${Number(result.fit.mae_m).toFixed(2)} m · ${result.path}`,'success');
        await refreshDistanceStatus();
      }
      canvas.onclick=event=>{if(!activeSnapshot)return setMessage('Ambil frame depth terlebih dahulu.','error');selectedPoint=framePoint(event);addButton.disabled=false;draw();setMessage(`Titik (${selectedPoint[0]}, ${selectedPoint[1]}) dipilih. Masukkan jarak aktual lalu tambah.`);};
      viewSelect.onchange=draw;
      actualInput.oninput=()=>{addButton.disabled=!activeSnapshot||!selectedPoint||!Number.isFinite(Number(actualInput.value))||Number(actualInput.value)<=0;};
      document.querySelector('#distanceCapture').onclick=()=>captureDistanceFrame().catch(error=>setMessage(`Gagal mengambil depth frame: ${error.message}`,'error'));
      addButton.onclick=()=>addDistanceSample().catch(error=>setMessage(`Gagal menambah titik: ${error.message}`,'error'));
      document.querySelector('#distanceSave').onclick=()=>saveDistanceCalibration().catch(error=>setMessage(`Gagal menyimpan koreksi: ${error.message}`,'error'));
      document.querySelector('#distanceSampleList').onclick=event=>{const button=event.target.closest('[data-distance-remove]');if(!button)return;samples.splice(Number(button.dataset.distanceRemove),1);refreshSamples();};
      refreshSamples();refreshDistanceStatus();
    }

    function aiResolutionUI(){
      const select=document.querySelector('#aiResolutionMode'),apply=document.querySelector('#applyAiResolution'),infoElement=document.querySelector('#aiResolutionInfo');
      let current=null;
      function renderInfo(){
        if(!current)return;
        const choice=current.choices.find(item=>item.mode===select.value);
        if(select.value==='custom'){
          const y=current.yolo_size,d=current.depth_size;
          infoElement.textContent=`Kustom aktif · YOLO ${y.width??'—'}×${y.height??'—'} · Depth ${d.width??'—'}×${d.height??'—'} · kamera ${current.camera_size?`${current.camera_size.width}×${current.camera_size.height}`:'menunggu frame'}`;
        }else if(choice&&current.camera_size){
          infoElement.textContent=`Kamera asli ${current.camera_size.width}×${current.camera_size.height} → AI ${choice.width}×${choice.height}. Rasio aspek dipertahankan; YOLO melakukan letterbox dan Depth Anything resize internal.`;
        }else{
          infoElement.textContent='Menunggu resolusi asli dari kamera.';
        }
        apply.disabled=select.value==='custom'||select.value===current.mode;
      }
      async function loadInfo(){
        const response=await fetch('/api/settings/ai-resolution',{cache:'no-store'}),data=await response.json();
        if(!response.ok)throw new Error(data.error||'Tidak dapat membaca settings resolusi AI');
        current=data;select.innerHTML='';
        if(!data.camera_size){const waiting=document.createElement('option');waiting.value='';waiting.textContent='Menunggu resolusi kamera…';select.appendChild(waiting);select.disabled=true;apply.disabled=true;infoElement.textContent='Pilihan resolusi muncul setelah kamera mengirim frame pertama.';setTimeout(()=>loadInfo().catch(error=>{infoElement.textContent=error.message;}),1000);return;}
        select.disabled=false;
        if(data.mode==='custom'){
          const option=document.createElement('option');option.value='custom';option.textContent='Kustom saat ini';option.disabled=true;select.appendChild(option);
        }
        for(const choice of data.choices){const option=document.createElement('option');option.value=choice.mode;option.textContent=choice.mode==='native'?choice.label:`${choice.label} (sisi panjang ${choice.mode}px)`;select.appendChild(option);}
        select.value=data.mode;renderInfo();
      }
      select.onchange=renderInfo;
      apply.onclick=async()=>{
        apply.disabled=true;infoElement.textContent='Menerapkan resolusi AI…';
        try{const response=await fetch('/api/settings/ai-resolution',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:select.value})}),data=await response.json();if(!response.ok)throw new Error(data.error||'Gagal menerapkan resolusi');current=data;select.value=data.mode;renderInfo();infoElement.textContent=`Diterapkan. Kamera ${data.camera_size?`${data.camera_size.width}×${data.camera_size.height}`:'menunggu frame'} → YOLO ${data.yolo_size.width}×${data.yolo_size.height}, Depth ${data.depth_size.width}×${data.depth_size.height}.`;}
        catch(error){infoElement.textContent=error.message;renderInfo();}
      };
      loadInfo().catch(error=>{infoElement.textContent=error.message;apply.disabled=true;});
    }

    function calibration(){
      app.innerHTML=`<div class="stack">
        <section class="card stack" aria-labelledby="calTitle"><div class="row spread"><div><h1 id="calTitle">Settings kamera dan kalibrasi</h1><p class="subtle">Atur resolusi inferensi, kalibrasi jarak Depth AI, dan zona piksel kamera.</p></div><span id="dirtyBadge" class="badge clean-badge">Belum ada perubahan</span></div>
          <div class="toolbar row spread"><div class="row"><button type="button" id="capture" class="primary">Ambil frame untuk kalibrasi</button><button type="button" id="release" class="muted-button">Batalkan snapshot</button></div><span id="snapshotState" class="badge">Snapshot belum diambil</span></div>
          <div id="calmsg" role="status" aria-live="polite"></div><div id="zonestatus"></div><div class="row spread"><span id="snapshotmeta" class="small">Waktu dan resolusi snapshot akan tampil di sini.</span><span id="calPath" class="path"></span></div>
        </section>
        <section class="card stack" aria-labelledby="aiResolutionTitle">
          <div><h2 id="aiResolutionTitle">Resolusi inferensi AI</h2><p class="small">Pilih sisi panjang input model; lebar/tinggi dihitung dari resolusi asli kamera dengan rasio aspek tetap. Frame monitoring dan koordinat zona tetap pada resolusi asli.</p></div>
          <div class="row mobile-stack"><label for="aiResolutionMode">Mode resolusi AI <select id="aiResolutionMode"></select></label><button type="button" id="applyAiResolution" class="primary">Terapkan resolusi</button></div>
          <div id="aiResolutionInfo" class="notice" role="status" aria-live="polite">Membaca resolusi kamera…</div>
        </section>
        <section class="card stack" aria-labelledby="distanceTitle">
          <div><h2 id="distanceTitle">Kalibrasi jarak Depth AI</h2><p class="small">Ambil frame yang sudah diproses depth, klik titik kaki/person, lalu masukkan kedalaman aktual camera-axis dalam meter. Ukur nilai dengan definisi yang sama; jarak mendatar di lantai tidak setara jika kamera miring. Gunakan titik pada beberapa rentang jarak bila tersedia. Koreksi ini hanya post-processing, bukan fine-tuning model atau validasi keselamatan.</p></div>
          <div class="row mobile-stack"><button type="button" id="distanceCapture" class="primary">Ambil frame depth</button><label for="actualDistance">Jarak aktual (meter) <input id="actualDistance" type="number" min="0.01" step="0.01" inputmode="decimal" aria-label="Jarak aktual dalam meter"></label><button type="button" id="distanceAdd" disabled>Tambah titik ukur</button><button type="button" id="distanceSave" disabled>Simpan koreksi jarak</button><label for="distanceView">Tampilan sampel <select id="distanceView"><option value="camera">Kamera</option><option value="depth">Depth map</option></select></label></div>
          <div class="row spread"><span id="distanceCalibrationStatus" class="badge">Memeriksa koreksi jarak…</span><span id="distanceCalibrationPath" class="path"></span></div>
          <div id="distanceMessage" class="notice" role="status" aria-live="polite">Kalibrasi jarak belum dimuat. Ambil frame depth untuk mulai menambahkan titik ukur.</div>
          <div class="distance-grid"><div><div class="canvas-wrap"><canvas id="distanceCanvas" tabindex="0" aria-label="Pilih titik kaki atau titik referensi pada frame kalibrasi jarak."></canvas></div><div class="canvas-caption"><span>Klik titik kaki atau titik ukur pada gambar.</span><span id="distanceFrameInfo">Belum ada frame</span></div></div><aside class="stack"><div class="metric"><label>Koreksi skala pratinjau</label><strong id="distanceFitPreview">Belum cukup sampel</strong></div><p class="small">Satu titik menghitung faktor skala tunggal; beberapa titik memakai median rasio aktual/prediksi. Residual sampel bukan pengujian akurasi independen.</p><ol id="distanceSampleList" class="distance-samples"></ol></aside></div>
        </section>
        <div class="grid-cal"><section class="card" aria-label="Area kerja poligon"><div class="row spread"><h2>Frame referensi</h2><label class="row"><input type="checkbox" id="boxes" checked> Tampilkan bounding box person</label></div><div class="canvas-wrap"><canvas id="canvas" tabindex="0" aria-label="Canvas kalibrasi. Klik untuk menambah titik. Keyboard: Enter selesai, Backspace undo, Escape batal draft, Delete hapus poligon terpilih."></canvas></div><div class="canvas-caption"><span>Koordinat editor mengikuti dimensi piksel frame asli.</span><span id="canvasDimensions">—</span></div></section>
          <aside class="stack" aria-label="Kontrol kalibrasi"><section class="card stack"><div><h2>Jenis poligon berikutnya</h2><p class="small">Mengganti jenis hanya berlaku untuk poligon baru; poligon selesai tetap utuh.</p></div><div class="row mobile-stack" id="zoneTools"><button type="button" class="zone-tool hazard" data-zone="hazard" aria-pressed="true">● Bahaya</button><button type="button" class="zone-tool warning" data-zone="warning" aria-pressed="false">● Waspada</button><button type="button" class="zone-tool safe" data-zone="safe" aria-pressed="false">● Aman</button></div>
              <div class="row mobile-stack"><button type="button" id="finish">Selesaikan poligon</button><button type="button" id="undo">Undo titik</button><button type="button" id="cancelpoly">Batal draft</button><button type="button" id="delete" class="danger-action">Hapus poligon terpilih</button><button type="button" id="reset" class="muted-button">Hapus semua poligon</button><button type="button" id="load" class="muted-button">Muat dari file</button></div>
              <div id="zoneChecklist" class="checklist" aria-live="polite"></div><p id="completionMessage" class="small"></p>
            </section>
            <section class="card stack"><div class="row spread"><h2>Poligon di editor</h2><span id="editorTotal" class="badge">0 total</span></div><div id="zonelist" class="zone-list"></div><div class="keyboard-hint">Keyboard: <b>Enter</b> selesaikan · <b>Backspace</b> undo titik · <b>Esc</b> batal draft · <b>Delete</b> hapus pilihan.</div></section>
            <div class="save-panel"><button type="button" id="save" class="primary">Simpan kalibrasi</button><p class="small">Menyimpan seluruh poligon editor ke file JSON yang ditampilkan di atas.</p></div>
          </aside>
        </div>
        <section id="saveResult" class="notice" role="status" aria-live="polite" hidden></section>
      </div>`;
      const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d'),dialog=document.querySelector('#snapshotDialog');
      let image=new Image(),zones=[],draft=[],people=[],selected='hazard',selectedIndex=-1,w=0,h=0,drag=null,snapshot=null,dirty=false;
      const msg=(text,kind='')=>{const el=document.querySelector('#calmsg');el.className=kind?`notice ${kind}`:'notice';el.textContent=text;};
      const counts=()=>Object.fromEntries(Object.keys(zoneKinds).map(name=>[name,zones.filter(z=>z.name===name).length]));
      function refreshSummary(){
        const c=counts(),complete=Object.values(c).every(n=>n>0);
        document.querySelector('#zoneChecklist').innerHTML=Object.entries(zoneKinds).map(([name,info])=>`<div class="check-item ${c[name]?'complete':'missing'}"><strong><span class="zone-dot ${info.dot}"></span>${info.label}</strong><span>${c[name]} poligon ${c[name]===1?'tersedia':'tersedia'}</span></div>`).join('');
        document.querySelector('#completionMessage').textContent=complete?'Ketiga jenis zona sudah memiliki poligon.':'Kalibrasi belum lengkap: buat minimal satu poligon Bahaya, Waspada, dan Aman. Tidak ada zona yang dibuat otomatis.';
        document.querySelector('#completionMessage').className=complete?'notice success':'notice';
        document.querySelector('#editorTotal').textContent=`${zones.length} poligon`;
        document.querySelector('#dirtyBadge').textContent=dirty?'Perubahan belum disimpan':'Belum ada perubahan';
        document.querySelector('#dirtyBadge').className=`badge ${dirty?'dirty-badge':'clean-badge'}`;
        document.querySelector('#snapshotState').textContent=snapshot?'Snapshot aktif':'Snapshot belum diambil';
        document.querySelector('#canvasDimensions').textContent=w&&h?`Frame asli ${w}×${h} px`:'—';
        document.querySelector('#zonelist').innerHTML=zones.map((z,i)=>{const k=zoneKinds[z.name];return `<button type="button" class="zone-entry" data-index="${i}" aria-pressed="${i===selectedIndex}" aria-label="Pilih poligon ${i+1}, ${esc(k.label)}, ${z.points.length} titik"><span class="zone-entry-label"><span class="zone-dot ${k.dot}"></span>${esc(k.label)} ${i+1}</span><span class="point-count">${z.points.length} titik</span></button>`;}).join('')||'<p class="subtle">Belum ada poligon.</p>';
      }
      function draw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);
        if(!w||!h||!image.complete)return;
        ctx.drawImage(image,0,0,w,h);
        if(document.querySelector('#boxes').checked)people.forEach(person=>{const [x1,y1,x2,y2]=person.bbox;ctx.strokeStyle='#f7fbff';ctx.lineWidth=Math.max(2,w/500);ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.fillStyle='#fff';ctx.font=`${Math.max(16,w/60)}px sans-serif`;ctx.fillText(`Person ${person.id??'?'} ${Number(person.confidence||0).toFixed(2)}`,x1,Math.max(18,y1-4));});
        zones.forEach((zone,index)=>{const kind=zoneKinds[zone.name],points=zone.points;ctx.fillStyle=kind.color+'55';ctx.strokeStyle=kind.color;ctx.lineWidth=Math.max(2,w/600);ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();ctx.stroke();ctx.fillStyle='#fff';ctx.font=`700 ${Math.max(16,w/55)}px sans-serif`;ctx.fillText(`${kind.label.toUpperCase()} ${index+1}`,points[0][0]+5,Math.max(18,points[0][1]-6));if(index===selectedIndex)points.forEach(p=>{ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(p[0],p[1],Math.max(5,w/180),0,Math.PI*2);ctx.fill();});});
        if(draft.length){ctx.strokeStyle='#f4f8fc';ctx.lineWidth=Math.max(2,w/600);ctx.beginPath();draft.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();draft.forEach(p=>{ctx.fillStyle='#fff';ctx.fillRect(p[0]-4,p[1]-4,8,8);});}
      }
      function render(){draw();refreshSummary();}
      function setDirty(){dirty=true;refreshSummary();}
      function canvasPoint(event){const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(w-1,Math.round((event.clientX-rect.left)*w/rect.width))),Math.max(0,Math.min(h-1,Math.round((event.clientY-rect.top)*h/rect.height)))];}
      function deleteSnapshot(id){if(id)fetch('/api/calibration/snapshot?snapshot_id='+encodeURIComponent(id),{method:'DELETE'}).catch(()=>{});}
      function chooseReplacement(sameResolution){
        document.querySelector('#snapshotDialogText').textContent=sameResolution?'Ada perubahan atau poligon di editor. Pilih apakah snapshot lama dipertahankan, perubahan dibuang, atau semua koordinat dipertahankan karena resolusi frame sama persis.':'Ada perubahan/poligon, tetapi resolusi frame baru berbeda. Koordinat lama tidak bisa dipertahankan.';
        const preserve=document.querySelector('[data-snapshot-choice="preserve"]');preserve.disabled=!sameResolution;
        return new Promise(resolve=>{dialog.addEventListener('close',()=>resolve(dialog.returnValue||'keep'),{once:true});dialog.querySelectorAll('[data-snapshot-choice]').forEach(button=>button.onclick=()=>dialog.close(button.dataset.snapshotChoice));dialog.showModal();});
      }
      async function loadSnapshotImage(snapshotInfo){
        return new Promise((resolve,reject)=>{const candidateImage=new Image();candidateImage.onload=()=>resolve(candidateImage);candidateImage.onerror=()=>reject(new Error('Gambar frame snapshot tidak dapat dimuat'));candidateImage.src='/api/calibration/snapshot.jpg?snapshot_id='+encodeURIComponent(snapshotInfo.snapshot_id)+'&t='+Date.now();});
      }
      async function captureSnapshot(){
        const response=await fetch('/api/calibration/snapshot',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const candidate=await response.json();
        if(!response.ok)return msg(candidate.error||'Gagal mengambil snapshot','error');
        let candidateImage;
        try{candidateImage=await loadSnapshotImage(candidate);}catch(error){deleteSnapshot(candidate.snapshot_id);return msg(error.message,'error');}
        const nextW=candidateImage.naturalWidth,nextH=candidateImage.naturalHeight,sameResolution=w===nextW&&h===nextH;
        const hasEditorState=dirty||zones.length>0||draft.length>0;
        let decision=hasEditorState?(dirty||!sameResolution?'ask':'preserve'):'replace';
        if(decision==='ask')decision=await chooseReplacement(sameResolution);
        if(decision==='keep'||(decision==='preserve'&&!sameResolution)){
          deleteSnapshot(candidate.snapshot_id);msg('Snapshot lama dan poligon editor tetap dipakai. Frame baru dibatalkan.');return;
        }
        const oldId=snapshot?.snapshot_id;
        if(decision==='discard'){zones=[];draft=[];selectedIndex=-1;dirty=false;}
        snapshot=candidate;image=candidateImage;people=candidate.people||[];w=nextW;h=nextH;canvas.width=w;canvas.height=h;
        if(oldId)deleteSnapshot(oldId);
        document.querySelector('#snapshotmeta').textContent=`Waktu ${candidate.captured_at} · Resolusi ${w}×${h} · Frame ${candidate.frame_id??'—'}`;
        document.querySelector('#calPath').textContent='';
        document.querySelector('#saveResult').hidden=true;
        render();
        msg(decision==='preserve'&&hasEditorState?'Frame diganti; seluruh poligon dan draft dipertahankan karena resolusi sama.':'Snapshot baru aktif. Gambar poligon yang diperlukan.','success');
        refreshZoneStatus();
      }
      async function refreshZoneStatus(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();document.querySelector('#zonestatus').innerHTML=zoneText(s);document.querySelector('#calPath').textContent=`File JSON: ${s.zone_calibration||'—'}`;}catch(_){document.querySelector('#zonestatus').innerHTML='<div class="notice error">Status file kalibrasi tidak dapat dibaca.</div>';}}
      async function releaseSnapshot(){
        if((dirty||zones.length||draft.length)&&!confirm('Batalkan snapshot dan buang semua perubahan/draft yang belum disimpan?'))return;
        deleteSnapshot(snapshot?.snapshot_id);snapshot=null;image=new Image();people=[];zones=[];draft=[];selectedIndex=-1;w=h=0;dirty=false;canvas.width=canvas.height=0;document.querySelector('#snapshotmeta').textContent='Snapshot belum diambil.';render();msg('Snapshot dibatalkan; frame dan poligon editor dikosongkan atas konfirmasi.');
      }
      async function loadZones(){
        if(!snapshot)return msg('Ambil frame sebelum memuat file agar resolusi dapat diperiksa.','error');
        if(dirty&&!confirm('Muat file akan mengganti perubahan poligon yang belum disimpan. Lanjutkan?'))return;
        const response=await fetch('/api/calibration/load?snapshot_id='+encodeURIComponent(snapshot.snapshot_id));const data=await response.json();
        if(!response.ok){msg(data.error||'Kalibrasi gagal dimuat.','error');refreshZoneStatus();return;}
        zones=data.zones.map(z=>({name:z.name,color:z.color,points:z.points.map(p=>[p[0],p[1]])}));draft=[];selectedIndex=-1;dirty=false;render();msg(`Kalibrasi dimuat dari ${document.querySelector('#calPath').textContent}.`,'success');refreshZoneStatus();
      }
      async function saveZones(){
        if(!snapshot)return msg('Ambil frame sebelum menyimpan zona.','error');
        const c=counts(),complete=Object.values(c).every(n=>n>0);
        if(!zones.length)return msg('Tambahkan dan selesaikan minimal satu poligon sebelum menyimpan.','error');
        if(draft.length)return msg('Selesaikan atau batalkan draft poligon sebelum menyimpan.','error');
        if(!complete&&!confirm(`Kalibrasi belum lengkap. Bahaya ${c.hazard}, Waspada ${c.warning}, Aman ${c.safe}. Simpan sebagai BELUM LENGKAP?`))return;
        const payload={snapshot_id:snapshot.snapshot_id,include_detections:document.querySelector('#boxes').checked,zones:zones.map(z=>({name:z.name,color:zoneKinds[z.name].json,points:z.points.map(p=>[p[0],p[1]])}))};
        const response=await fetch('/api/calibration/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const result=await response.json();
        if(!response.ok)return msg(result.error||'Kalibrasi gagal disimpan. File sebelumnya tetap dipertahankan.','error');
        snapshot=null;dirty=false;refreshSummary();
        const countsText=Object.entries(zoneKinds).map(([name,k])=>`${k.label} ${result.zone_counts[name]||0}`).join(' · ');
        const saveResult=document.querySelector('#saveResult');saveResult.hidden=false;saveResult.className=`notice ${result.complete?'success':''}`;saveResult.innerHTML=`<strong>${result.complete?'Tersimpan · LENGKAP':'Tersimpan · BELUM LENGKAP'}</strong><br>${esc(result.zone_total)} poligon: ${esc(countsText)}<br><span class="path">JSON: ${esc(result.path)}</span><br><span class="path">Gambar referensi: ${esc(result.reference_path)}</span>${result.overlap_warning?'<br><strong>Peringatan: beberapa poligon bertumpang tindih.</strong>':''}`;
        document.querySelector('#calPath').textContent=`File JSON: ${result.path}`;msg('Kalibrasi berhasil disimpan.','success');refreshZoneStatus();
      }
      document.querySelector('#capture').onclick=()=>captureSnapshot().catch(error=>msg(`Gagal mengambil snapshot: ${error.message}`,'error'));
      document.querySelector('#release').onclick=releaseSnapshot;
      document.querySelector('#boxes').onchange=draw;
      document.querySelectorAll('[data-zone]').forEach(button=>button.onclick=()=>{selected=button.dataset.zone;document.querySelectorAll('[data-zone]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));msg(`Jenis poligon berikutnya: ${zoneKinds[selected].label}. Poligon lama tidak berubah.`);});
      document.querySelector('#finish').onclick=()=>{if(!snapshot)return msg('Ambil frame terlebih dahulu.','error');if(draft.length<3)return msg('Poligon perlu minimal tiga titik.','error');zones.push({name:selected,color:zoneKinds[selected].json,points:draft.map(p=>[p[0],p[1]])});draft=[];selectedIndex=zones.length-1;setDirty();render();msg(`${zoneKinds[selected].label} selesai ditambahkan.`,'success');};
      document.querySelector('#undo').onclick=()=>{if(draft.length){draft.pop();setDirty();render();msg('Titik terakhir dibatalkan.');}else msg('Tidak ada titik draft untuk di-undo.');};
      document.querySelector('#cancelpoly').onclick=()=>{if(!draft.length)return msg('Tidak ada draft poligon.');draft=[];setDirty();render();msg('Draft poligon dibatalkan; perubahan lain tetap ada.');};
      document.querySelector('#delete').onclick=()=>{if(selectedIndex<0)return msg('Pilih satu poligon pada daftar terlebih dahulu.','error');const removed=zones.splice(selectedIndex,1)[0];selectedIndex=-1;setDirty();render();msg(`Poligon ${zoneKinds[removed.name].label} dihapus.`);};
      document.querySelector('#reset').onclick=()=>{if(!zones.length&&!draft.length)return msg('Editor sudah kosong.');if(!confirm('Hapus semua poligon dan draft dari editor? File tersimpan tidak diubah sampai Anda menyimpan.'))return;zones=[];draft=[];selectedIndex=-1;setDirty();render();msg('Semua poligon editor dihapus; file tersimpan belum berubah.');};
      document.querySelector('#load').onclick=()=>loadZones().catch(error=>msg(`Gagal memuat kalibrasi: ${error.message}`,'error'));
      document.querySelector('#save').onclick=()=>saveZones().catch(error=>msg(`Gagal menyimpan kalibrasi: ${error.message}`,'error'));
      document.querySelector('#zonelist').onclick=event=>{const button=event.target.closest('[data-index]');if(!button)return;selectedIndex=Number(button.dataset.index);render();};
      canvas.onpointerdown=event=>{if(!snapshot)return msg('Ambil frame sebelum menggambar.','error');const point=canvasPoint(event);if(selectedIndex>=0){const zone=zones[selectedIndex],vertex=zone.points.findIndex(p=>Math.hypot(p[0]-point[0],p[1]-point[1])<Math.max(12,w/80));if(vertex>=0){drag={zone:selectedIndex,vertex};canvas.setPointerCapture(event.pointerId);return;}}draft.push(point);setDirty();render();};
      canvas.onpointermove=event=>{if(!drag)return;zones[drag.zone].points[drag.vertex]=canvasPoint(event);setDirty();draw();};
      canvas.onpointerup=()=>{drag=null;};
      canvas.onpointercancel=()=>{drag=null;};
      window.addEventListener('keydown',event=>{if(dialog.open||event.target.matches('input,textarea,select'))return;if(event.key==='Enter'&&event.target===canvas){event.preventDefault();document.querySelector('#finish').click();}else if(event.key==='Backspace'&&draft.length){event.preventDefault();document.querySelector('#undo').click();}else if(event.key==='Escape'&&draft.length){event.preventDefault();document.querySelector('#cancelpoly').click();}else if(event.key==='Delete'&&selectedIndex>=0){event.preventDefault();document.querySelector('#delete').click();}});
      render();refreshZoneStatus();aiResolutionUI();distanceCalibrationUI();
    }
    if(location.pathname==='/calibration')calibration();else monitoring();
  </script>
</main>
</body>
</html>'''


def create_app(pipeline: CameraPipeline, zone_path: Path, host: str = "127.0.0.1") -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "Request body exceeds the 256 KiB limit"}), 413

    @app.get("/")
    @app.get("/calibration")
    def home():
        response = make_response(render_template_string(PAGE))
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/api/status")
    def api_status():
        jpeg, status = pipeline.snapshot()
        current_zones = (pipeline.zones or {}).get("zones", [])
        counts = {name: sum(zone.get("name") == name for zone in current_zones)
                  for name in ("hazard", "warning", "safe")}
        status["frame_available"] = jpeg is not None
        status["zone_calibration"] = str(zone_path.resolve())
        status["distance_calibration_file"] = str(pipeline.distance_calibration_path.resolve())
        status["distance_calibration_status"] = pipeline.distance_calibration_status
        status["distance_calibration_message"] = pipeline.distance_calibration_message
        status["distance_calibration_scale"] = pipeline.distance_calibration["fit"]["scale"] if pipeline.distance_calibration else None
        status["distance_calibration_samples"] = len(pipeline.distance_calibration["samples"]) if pipeline.distance_calibration else 0
        status["zone_counts"] = counts
        status["zone_total"] = len(current_zones)
        status["zone_complete"] = all(counts.values())
        status["lan_warning"] = host not in ("127.0.0.1", "localhost", "::1")
        return jsonify(status)

    @app.get("/api/frame.jpg")
    def api_frame():
        view = request.args.get("view", "ai")
        if view not in {"camera", "ai", "depth"}:
            return jsonify({"error": "view must be one of: camera, ai, depth"}), 400
        requested_frame = request.args.get("frame_id")
        try:
            frame_id = int(requested_frame) if requested_frame is not None else None
        except ValueError:
            return jsonify({"error": "frame_id must be an integer"}), 400
        jpeg, status = pipeline.view_snapshot(view, frame_id)
        if jpeg is None and frame_id is not None and "requested_frame_id" in status:
            return jsonify({"error": f"Frame {frame_id} is no longer in the synchronization buffer", "latest_frame_id": status.get("frame_id")}), 409
        if jpeg is None:
            return jsonify({"error": status.get("message") or status.get("error") or f"{view} view is not available"}), 503
        selected_frame_id = frame_id if frame_id is not None else status.get("frame_id")
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store", "X-Frame-ID": str(selected_frame_id)})

    @app.get("/api/settings/ai-resolution")
    def api_ai_resolution_settings():
        return jsonify(pipeline.ai_resolution_info())

    @app.post("/api/settings/ai-resolution")
    def api_set_ai_resolution():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("mode"), str):
            return jsonify({"error": "mode is required"}), 400
        try:
            return jsonify(pipeline.set_ai_resolution(payload["mode"]))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": f"Could not save AI resolution setting: {exc}"}), 500

    @app.route("/api/distance-calibration/snapshot", methods=["POST", "GET", "DELETE"])
    def api_distance_snapshot():
        if request.method == "POST":
            try:
                snapshot = pipeline.capture_distance_calibration_snapshot()
            except SnapshotCapacityError as exc:
                return jsonify({"error": str(exc)}), 429
            if snapshot is None:
                depth_status = pipeline.status.get("depth_model_status")
                if depth_status != "ready":
                    detail = pipeline.status.get("model_message") or "Depth model is unavailable"
                else:
                    detail = "Waiting for a fresh Depth Anything inference frame"
                return jsonify({"error": detail}), 503
            return jsonify(snapshot)
        snapshot_id = request.args.get("snapshot_id")
        if not snapshot_id:
            return jsonify({"error": "snapshot_id is required"}), 400
        if request.method == "DELETE":
            with pipeline.lock:
                pipeline.distance_calibration_snapshots.pop(snapshot_id, None)
            return jsonify({"ok": True})
        snapshot = pipeline.get_distance_calibration_snapshot(snapshot_id)
        if snapshot is None:
            return jsonify({"error": "Distance calibration snapshot expired; capture a new depth frame"}), 410
        return jsonify({key: value for key, value in snapshot.items()
                        if key not in {"frame", "depth_map", "depth_map_zlib", "depth_dtype", "jpeg", "depth_jpeg", "_created_monotonic"}})

    @app.get("/api/distance-calibration/snapshot.jpg")
    def api_distance_snapshot_frame():
        snapshot = pipeline.get_distance_calibration_snapshot(request.args.get("snapshot_id"))
        if snapshot is None:
            return jsonify({"error": "Distance calibration snapshot expired or unavailable"}), 410
        return Response(snapshot["jpeg"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/distance-calibration/depth.jpg")
    def api_distance_snapshot_depth():
        snapshot = pipeline.get_distance_calibration_snapshot(request.args.get("snapshot_id"))
        if snapshot is None:
            return jsonify({"error": "Distance calibration snapshot expired or unavailable"}), 410
        return Response(snapshot["depth_jpeg"], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/distance-calibration/sample")
    def api_distance_sample():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object"}), 400
        snapshot_id = payload.get("snapshot_id")
        pixel = payload.get("pixel")
        if not isinstance(snapshot_id, str) or not isinstance(pixel, list) or len(pixel) != 2:
            return jsonify({"error": "snapshot_id and pixel [x, y] are required"}), 400
        try:
            actual = float(payload.get("actual_distance_m"))
            sample = pipeline.sample_distance_calibration_point(snapshot_id, pixel, actual)
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 410
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(sample)

    @app.post("/api/distance-calibration/save")
    def api_save_distance_calibration():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            return jsonify({"error": "Expected samples as a JSON array"}), 400
        try:
            calibration = pipeline.save_distance_calibration(payload["samples"])
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 410
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400
        except OSError as exc:
            return jsonify({"error": f"Could not save distance calibration: {exc}"}), 500
        return jsonify({"ok": True, "path": str(pipeline.distance_calibration_path.resolve()),
                        "fit": calibration["fit"], "sample_count": len(calibration["samples"]),
                        "frame_size": calibration["frame_size"], "model_id": calibration["model_id"]})

    @app.get("/api/calibration/frame")
    def api_calibration_frame():
        return jsonify({"error": "Use Ambil frame untuk kalibrasi to freeze a snapshot"}), 409

    @app.route("/api/calibration/snapshot", methods=["POST", "GET", "DELETE"])
    def api_calibration_snapshot():
        if request.method == "POST":
            previous = request.get_json(silent=True) or {}
            try:
                snapshot = pipeline.capture_calibration_snapshot()
            except SnapshotCapacityError as exc:
                return jsonify({"error": str(exc)}), 429
            if snapshot is None:
                return jsonify({"error": "Frame kamera terbaru tidak tersedia; coba lagi setelah frame baru masuk"}), 503
            if isinstance(previous, dict):
                pipeline.cancel_calibration_snapshot(previous.get("snapshot_id"))
            return jsonify(snapshot)
        if request.method == "DELETE":
            pipeline.cancel_calibration_snapshot(request.args.get("snapshot_id"))
            return jsonify({"ok": True})
        snapshot_id = request.args.get("snapshot_id")
        if not snapshot_id:
            return jsonify({"error": "snapshot_id is required"}), 400
        _, snapshot = pipeline.get_calibration_snapshot(snapshot_id)
        if snapshot is None:
            return jsonify({"error": "Snapshot expired or unavailable; capture a new calibration frame"}), 410
        return jsonify(snapshot)

    @app.get("/api/calibration/snapshot.jpg")
    def api_calibration_snapshot_jpg():
        snapshot_id = request.args.get("snapshot_id")
        if not snapshot_id:
            return jsonify({"error": "snapshot_id is required"}), 400
        jpeg, snapshot = pipeline.get_calibration_snapshot(snapshot_id)
        if jpeg is None or snapshot is None:
            return jsonify({"error": "Snapshot expired or unavailable; capture a new calibration frame"}), 410
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/calibration/load")
    def api_load_zones():
        requested_id = request.args.get("snapshot_id")
        if not requested_id:
            return jsonify({"error": "snapshot_id is required to verify calibration resolution"}), 400
        try:
            with pipeline.lock:
                requested_snapshot = pipeline._get_calibration_snapshot_locked(requested_id)
                if requested_snapshot is None:
                    return jsonify({"error": "Snapshot expired or unavailable; capture a new calibration frame", "zone_status": "not_available"}), 410
                frame_size = requested_snapshot["frame_size"]
            if not isinstance(frame_size, dict):
                return jsonify({"error": "No frame resolution available to check calibration"}), 503
            width, height = int(frame_size["width"]), int(frame_size["height"])
            data = load_zone_json(zone_path, (width, height))
            with pipeline.lock:
                pipeline.zones = data
                pipeline.status.update(zone_status="loaded", zone_message="Kalibrasi berhasil dimuat", zone_types=sorted({z["name"] for z in data["zones"]}))
            return jsonify(data)
        except ResolutionMismatchError as exc:
            with pipeline.lock:
                pipeline.zones = None
                pipeline.status.update(zone_status="resolution_mismatch", zone_message=str(exc), zone_types=[])
            return jsonify({"error": str(exc), "zone_status": "resolution_mismatch"}), 400
        except (CalibrationError, OSError) as exc:
            with pipeline.lock:
                pipeline.zones = None
                pipeline.status.update(zone_status="invalid", zone_message=str(exc), zone_types=[])
            return jsonify({"error": str(exc), "zone_status": "invalid"}), 400

    @app.post("/api/calibration/save")
    def api_save_zones():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Expected a JSON body"}), 400
        include_detections = bool(payload.get("include_detections", False)) if isinstance(payload, dict) else False
        snapshot_id = payload.get("snapshot_id") if isinstance(payload, dict) else None
        if not snapshot_id:
            return jsonify({"error": "snapshot_id is required to save calibration"}), 400
        with pipeline.lock:
            snapshot = pipeline._get_calibration_snapshot_locked(snapshot_id)
            if snapshot is None:
                return jsonify({"error": "Snapshot expired or unavailable; capture a new calibration frame"}), 410
            snapshot_jpeg = snapshot["jpeg"]
            frame_size = dict(snapshot["frame_size"])
            source_text = snapshot["source"]
            detections = [dict(p) for p in snapshot["people"]]
        frame = cv2.imdecode(np.frombuffer(snapshot_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[1] != frame_size["width"] or frame.shape[0] != frame_size["height"]:
            return jsonify({"error": "Could not decode the frozen calibration snapshot"}), 500
        height, width = frame.shape[:2]
        try:
            calibration = validate_zones_payload(payload, width, height, source_text)
            overlaps = find_overlaps(calibration["zones"], width, height)
            image = render_clean_reference(frame, calibration["zones"], detections if include_detections else None)
            ref = staged_pair_write(zone_path, calibration, image)
            zone_counts = {name: sum(zone["name"] == name for zone in calibration["zones"])
                           for name in ("hazard", "warning", "safe")}
            available = [name for name, count in zone_counts.items() if count]
            with pipeline.lock:
                pipeline.zones = calibration
                pipeline.calibration_snapshots.pop(snapshot_id, None)
                pipeline.status.update(zone_status="loaded", zone_message="Kalibrasi berhasil dimuat", zone_types=available)
            return jsonify({"ok": True, "path": str(zone_path.resolve()), "reference_path": str(ref.resolve()),
                            "zone_types": available, "zone_counts": zone_counts, "zone_total": len(calibration["zones"]),
                            "complete": all(zone_counts.values()), "overlap_warning": bool(overlaps),
                            "overlapping_pairs": overlaps}), 200
        except (CalibrationError, OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 500 if isinstance(exc, OSError) else 400

    if host not in ("127.0.0.1", "localhost", "::1"):
        LOG.warning("LAN bind enabled without login: any reachable LAN user can view camera video and change zone calibration")
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Forklift Vision AI monitoring and pixel-zone calibration web app")
    parser.add_argument("--source", help="camera index, video file, or RTSP URL; defaults to config")
    parser.add_argument("--config", help="settings JSON; defaults to config.jetson.json on Jetson, otherwise config.json")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow LAN access")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", help="existing detector weights path; no automatic download")
    parser.add_argument("--calibration", default="calibration/zone_calibration.json")
    parser.add_argument("--distance-calibration", default="calibration/distance_calibration.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline: CameraPipeline | None = None
    exit_code = 0
    try:
        settings = load_settings(Path(args.config) if args.config else None, source=args.source, model=args.model)
        zone_path = Path(args.calibration).expanduser()
        if not zone_path.is_absolute():
            zone_path = ROOT / zone_path
        distance_calibration_path = Path(args.distance_calibration).expanduser()
        if not distance_calibration_path.is_absolute():
            distance_calibration_path = ROOT / distance_calibration_path
        settings["distance_calibration_file"] = distance_calibration_path
        pipeline = CameraPipeline(settings, zone_path, distance_calibration_path)
        pipeline.start()
        app = create_app(pipeline, zone_path, args.host)
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        LOG.info("Web server stopped by Ctrl+C")
    except Exception as exc:
        LOG.exception("Web server or app startup failed")
        print(f"ERROR: Web server/app failed: {exc}")
        exit_code = 2
    finally:
        if pipeline is not None:
            try:
                shutdown_ok = pipeline.shutdown()
            except Exception as exc:
                LOG.exception("Pipeline shutdown raised an exception")
                print(f"ERROR: Pipeline shutdown raised: {exc}")
                shutdown_ok = False
            if not shutdown_ok:
                message = "Pipeline shutdown was not clean; worker or camera reader may still be active"
                LOG.error(message)
                print(f"ERROR: {message}")
                if exit_code == 0:
                    exit_code = 3
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
