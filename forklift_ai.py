from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from forklift_vision.logic import (
    DepthFilter, MotionHistory, WarningMachine, blend_ground_zone_overlay,
    build_ground_zone_overlay, data_is_stale, estimate_roi_depth, project_ground_point,
)
from forklift_vision.runtime import (
    DepthEstimator, VideoSource, environment_report, load_calibration,
    load_calibration_image_points, resolve_device, run_calibration,
)

ROOT = Path(__file__).resolve().parent
PIXEL_ZONE_BGR = {"hazard": (0, 0, 255), "warning": (255, 0, 0), "safe": (0, 255, 0)}
PIXEL_ZONE_LABELS = {"hazard": "BAHAYA", "warning": "WASPADA", "safe": "AMAN"}
PRESETS = {
    "laptop-cpu": {"device": "cpu", "yolo_size": 480, "depth_size": 392, "depth_every": 5},
    "laptop-gpu": {"device": "auto", "yolo_size": 640, "depth_size": 518, "depth_every": 3},
    "jetson": {"device": "auto", "yolo_size": 480, "depth_size": 392, "depth_every": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forklift Vision AI prototype: person tracking and metric camera-depth estimation")
    parser.add_argument("--config", help="settings JSON; defaults to config.jetson.json on Jetson and config.json elsewhere")
    parser.add_argument("--calibration-file")
    parser.add_argument("--zone-calibration", help="pixel-zone JSON (default: calibration/zone_calibration.json)")
    parser.add_argument("--source")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--yolo-model")
    parser.add_argument("--depth-model", choices=("indoor", "outdoor"))
    parser.add_argument("--depth-checkpoint")
    parser.add_argument("--depth-source")
    parser.add_argument("--depth-size", type=int)
    parser.add_argument("--yolo-size", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--depth-every", type=int)
    parser.add_argument("--show-depth", action="store_true", default=None)
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--save-video", nargs="?", const="auto")
    parser.add_argument("--log-file")
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument("--gstreamer", action="store_true", default=None)
    parser.add_argument("--calibrate-floor", action="store_true")
    parser.add_argument("--warning-distance", type=float)
    parser.add_argument("--critical-distance", type=float)
    parser.add_argument("--closing-speed-threshold", type=float)
    return parser.parse_args()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        is_jetson = platform.system() == "Linux" and Path("/etc/nv_tegra_release").is_file()
        config_path = ROOT / ("config.jetson.json" if is_jetson else "config.json")
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        settings = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read config {config_path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError("Configuration root must be a JSON object")
    if settings.get("preset") in PRESETS:
        settings.update(PRESETS[settings["preset"]])
    if args.preset:
        settings.update(PRESETS[args.preset])
    overrides = {
        "source": args.source, "device": args.device, "yolo_model": args.yolo_model,
        "depth_model": args.depth_model, "depth_checkpoint": args.depth_checkpoint,
        "depth_source": args.depth_source, "depth_size": args.depth_size,
        "yolo_size": args.yolo_size, "confidence": args.conf,
        "depth_every": args.depth_every, "show_depth": args.show_depth,
        "headless": args.headless, "save_video": args.save_video,
        "log_file": args.log_file, "preset": args.preset,
        "calibration_file": args.calibration_file,
        "zone_calibration_file": args.zone_calibration,
        "gstreamer": args.gstreamer,
        "warning_distance_m": args.warning_distance,
        "critical_distance_m": args.critical_distance,
        "closing_speed_threshold_mps": args.closing_speed_threshold,
    }
    settings.update({key: value for key, value in overrides.items() if value is not None})
    required = ("source", "device", "yolo_model", "depth_model", "depth_checkpoint", "depth_source", "depth_size", "yolo_size", "confidence", "depth_every", "log_file")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    if settings["depth_size"] <= 0 or settings["yolo_size"] <= 0 or settings["depth_every"] <= 0:
        raise ValueError("depth-size, yolo-size, and depth-every must be positive")
    if not 0 < float(settings["confidence"]) <= 1:
        raise ValueError("confidence must be in (0, 1]")
    if settings["critical_distance_m"] >= settings["warning_distance_m"]:
        raise ValueError("critical distance must be less than warning distance")
    for key in ("yolo_model", "depth_checkpoint", "depth_source", "log_file", "calibration_file", "zone_calibration_file", "save_video"):
        value = settings.get(key)
        if value and value != "auto":
            path = Path(value).expanduser()
            settings[key] = path if path.is_absolute() else ROOT / path
    return settings


def load_pixel_zone_calibration(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    """Load pixel zones only for their original frame resolution."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load pixel-zone calibration {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "forklift-vision-pixel-zones" or payload.get("version") != 1:
        raise ValueError("Pixel-zone JSON has an unsupported format or version")
    frame_size = payload.get("frame_size")
    if not isinstance(frame_size, dict):
        raise ValueError("Pixel-zone JSON is missing frame_size")
    if (frame_size.get("width"), frame_size.get("height")) != (width, height):
        raise ValueError(
            f"Pixel-zone calibration is {frame_size.get('width')}x{frame_size.get('height')}, "
            f"but camera frame is {width}x{height}; zones were not applied"
        )
    zones = payload.get("zones")
    if not isinstance(zones, list) or not zones:
        raise ValueError("Pixel-zone JSON has no zones")
    checked = []
    for zone in zones:
        if not isinstance(zone, dict) or zone.get("name") not in PIXEL_ZONE_BGR:
            raise ValueError("Pixel-zone JSON contains an unknown zone name")
        points = np.asarray(zone.get("points"), dtype=np.float32)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all()
                or np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
            raise ValueError(f"Zone {zone['name']} has invalid points for this frame")
        checked.append({"name": zone["name"], "points": np.rint(points).astype(np.int32)})
    return checked


def pixel_zone_for_point(zones: list[dict[str, Any]] | None, point: tuple[float, float]) -> str | None:
    if not zones:
        return None
    priority = {"hazard": 0, "warning": 1, "safe": 2}
    hits = []
    for zone in zones:
        polygon = zone["points"].reshape((-1, 1, 2))
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            hits.append(zone["name"])
    return min(hits, key=lambda name: priority[name]) if hits else None


def overlay_pixel_zones(image: np.ndarray, zones: list[dict[str, Any]]) -> np.ndarray:
    """Tint calibrated pixel polygons and label each area on a copy of the image."""
    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    ordered_zones = sorted(zones, key=lambda zone: {"safe": 0, "warning": 1, "hazard": 2}[zone["name"]])
    for zone in ordered_zones:
        polygon = zone["points"]
        cv2.fillPoly(overlay, [polygon], PIXEL_ZONE_BGR[zone["name"]])
        cv2.fillPoly(mask, [polygon], 255)
    blended = cv2.addWeighted(image, 0.58, overlay, 0.42, 0)
    result = image.copy()
    result[mask > 0] = blended[mask > 0]
    for zone in ordered_zones:
        polygon = zone["points"].reshape((-1, 1, 2))
        color = PIXEL_ZONE_BGR[zone["name"]]
        cv2.polylines(result, [polygon], True, color, 3, cv2.LINE_AA)
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) > 1e-6:
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        else:
            center = tuple(np.mean(zone["points"], axis=0).astype(int).tolist())
        cv2.putText(result, PIXEL_ZONE_LABELS[zone["name"]], center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return result


def setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Cannot write log file {log_path}: {exc}") from exc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings(args)
        if args.calibrate_floor:
            source = str(settings["source"])
            run_calibration(source, settings["calibration_file"])
            return 0
        setup_logging(settings["log_file"])
        logger = logging.getLogger("forklift_ai")
        import torch
        device = resolve_device(settings["device"], torch)
        logger.info("Environment: %s", environment_report(torch, device))
        if device == "cuda":
            try:
                torch.cuda.init()
            except (RuntimeError, AssertionError) as exc:
                if settings["device"] == "cuda":
                    raise RuntimeError(f"Requested CUDA initialization failed: {exc}") from exc
                logger.warning("CUDA initialization failed; falling back to CPU: %s", exc)
                device = "cpu"
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is unavailable. Install project requirements in the active Python environment.") from exc
        if not Path(settings["yolo_model"]).is_file() and not str(settings["yolo_model"]).startswith("yolo"):
            raise FileNotFoundError(f"YOLO weights not found: {settings['yolo_model']}; download yolo26n.pt or provide --yolo-model")
        yolo = YOLO(str(settings["yolo_model"]))
        depth = DepthEstimator(settings["depth_model"], settings["depth_checkpoint"], settings["depth_source"], device, settings["depth_size"])
        source = VideoSource(str(settings["source"]), gstreamer=bool(settings.get("gstreamer")), retries=int(settings.get("reconnect_retries", 5)))
        calibration = load_calibration(settings["calibration_file"])
        calibration_image_points = load_calibration_image_points(settings["calibration_file"])
        zones_calibrated = calibration is not None and calibration_image_points is not None
        logger.info("Ground-plane zones: %s", "calibrated" if zones_calibrated else "ZONE UNCALIBRATED")
        headless = bool(settings.get("headless", False))
        if headless:
            os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
        writer = None
        output_video = settings.get("save_video")
        if output_video:
            if output_video == "auto":
                output_video = ROOT / "outputs" / "recordings" / f"annotated-{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
            Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        motion = MotionHistory(timeout_s=float(settings.get("tracking_timeout_s", 3)), speed_threshold_mps=float(settings.get("closing_speed_threshold_mps", 0.3)))
        filters: dict[int, DepthFilter] = {}
        warning = WarningMachine(
            float(settings["warning_distance_m"]), float(settings["critical_distance_m"]), hysteresis_m=0.0
        )
        latest_depth = None
        latest_depth_at = None
        last_media_time = None
        frame_id = 0
        process_times: list[float] = []
        capture_times: list[float] = []
        last_capture = time.monotonic()
        zone_overlay = None
        zone_overlay_mask = None
        zone_overlay_shape = None
        pixel_zones: list[dict[str, Any]] | None = None
        pixel_zone_frame_size = None
        zone_calibration_path: Path = settings.get(
            "zone_calibration_file", ROOT / "calibration" / "zone_calibration.json"
        )
        pixel_zone_status = "PIXEL ZONES NOT LOADED"
        logging_path: Path = settings["log_file"]
        logging_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            while True:
                frame, media_time, observed_time = source.read()
                now = time.monotonic()
                if frame is None:
                    if source.is_live:
                        if source.reconnected:
                            source.reconnected = False
                            motion.clear()
                            filters.clear()
                            latest_depth = None
                            latest_depth_at = None
                            yolo = YOLO(str(settings["yolo_model"]))
                            logger.warning("Reset tracker and depth history after live-source reconnect")
                        continue
                    break
                frame_id += 1
                frame_height, frame_width = frame.shape[:2]
                frame_size = (frame_width, frame_height)
                if pixel_zone_frame_size != frame_size:
                    pixel_zone_frame_size = frame_size
                    try:
                        pixel_zones = load_pixel_zone_calibration(zone_calibration_path, frame_width, frame_height)
                        pixel_zone_status = "PIXEL ZONES: RED HAZARD | BLUE WARNING | GREEN SAFE"
                        logger.info("Loaded %d pixel zones from %s for %dx%d", len(pixel_zones), zone_calibration_path, frame_width, frame_height)
                    except (OSError, ValueError) as exc:
                        pixel_zones = None
                        pixel_zone_status = str(exc)
                        logger.warning("Pixel-zone overlay disabled: %s", exc)
                capture_times.append(now - last_capture)
                last_capture = now
                capture_times = capture_times[-30:]
                video_time = media_time if media_time is not None else observed_time
                if last_media_time is not None and video_time is not None and video_time <= last_media_time:
                    motion.clear()
                    filters.clear()
                last_media_time = video_time
                motion.cleanup(float(video_time if video_time is not None else observed_time))
                filters = {ident: filt for ident, filt in filters.items() if ident in motion.samples}
                start = time.perf_counter()
                depth_inference_ms = 0.0
                if frame_id == 1 or (frame_id - 1) % int(settings["depth_every"]) == 0:
                    try:
                        depth_start = time.perf_counter()
                        latest_depth = depth.infer(frame)
                        depth_inference_ms = (time.perf_counter() - depth_start) * 1000
                        latest_depth_at = observed_time
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower() and device == "cuda":
                            logger.error("CUDA out of memory during depth inference; stop and select --device cpu or a smaller preset: %s", exc)
                            latest_depth = None
                            latest_depth_at = None
                        else:
                            raise
                results = yolo.track(frame, persist=True, tracker="botsort.yaml", classes=[0], conf=float(settings["confidence"]), imgsz=int(settings["yolo_size"]), device=device, verbose=False)
                result = results[0]
                annotated = frame.copy()
                if zones_calibrated:
                    if zone_overlay_shape != frame.shape[:2]:
                        zone_overlay, zone_overlay_mask = build_ground_zone_overlay(
                            frame.shape, calibration, calibration_image_points,
                            float(settings["warning_distance_m"]), float(settings["critical_distance_m"]),
                        )
                        zone_overlay_shape = frame.shape[:2]
                    annotated = blend_ground_zone_overlay(annotated, zone_overlay, zone_overlay_mask)
                    zone_legend = (
                        f"FLOOR ZONES: RED <{settings['critical_distance_m']:.1f}m | "
                        f"YELLOW {settings['critical_distance_m']:.1f}-{settings['warning_distance_m']:.1f}m | "
                        f"GREEN >{settings['warning_distance_m']:.1f}m | origin (0,0)"
                    )
                    cv2.putText(annotated, zone_legend, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(annotated, "FLOOR ZONES UNCALIBRATED - run --calibrate-floor", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
                boxes = result.boxes
                person_records = []
                any_unknown = data_is_stale(latest_depth_at, observed_time, float(settings["max_depth_age_s"]))
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    ids_tensor = boxes.id
                    ids = ids_tensor.int().cpu().tolist() if ids_tensor is not None else [None] * len(xyxy)
                    for bbox, confidence, ident in zip(xyxy, confs, ids):
                        x1, y1, x2, y2 = bbox.tolist()
                        x1, x2 = max(0, min(frame.shape[1] - 1, int(x1))), max(0, min(frame.shape[1], int(x2)))
                        y1, y2 = max(0, min(frame.shape[0] - 1, int(y1))), max(0, min(frame.shape[0], int(y2)))
                        depth_value = None
                        raw = None
                        valid_fraction = 0.0
                        dispersion = None
                        if ident is not None and latest_depth is not None and not any_unknown:
                            measurement = estimate_roi_depth(latest_depth, [x1, y1, x2, y2], min_depth_m=float(settings["min_depth_m"]), max_depth_m=float(settings["max_depth_m"]))
                            if measurement:
                                raw, valid_fraction, dispersion = measurement
                                filt = filters.setdefault(int(ident), DepthFilter())
                                depth_value = filt.update(raw)
                        motion_estimate = motion.update(int(ident), float(video_time if video_time is not None else observed_time), depth_value) if ident is not None and depth_value is not None else None
                        if depth_value is None:
                            any_unknown = True
                        if ident is None:
                            motion_label, closing_speed, ttc = "UNKNOWN", None, None
                        elif motion_estimate:
                            motion_label, closing_speed, ttc = motion_estimate.status, motion_estimate.closing_speed_mps, motion_estimate.experimental_ttc_s
                        else:
                            motion_label, closing_speed, ttc = "UNKNOWN", None, None
                        ground = project_ground_point(calibration, ((x1 + x2) / 2, y2)) if zones_calibrated else None
                        floor_zone = None
                        floor_distance = None
                        if ground is not None and cv2.pointPolygonTest(
                            calibration_image_points.reshape((-1, 1, 2)), ((x1 + x2) / 2, y2), False
                        ) >= 0:
                            floor_distance = math.hypot(ground[0], ground[1])
                            if floor_distance < float(settings["critical_distance_m"]):
                                floor_zone = "DANGER"
                            elif floor_distance <= float(settings["warning_distance_m"]):
                                floor_zone = "WARNING"
                            else:
                                floor_zone = "SAFE"
                        foot_point = ((x1 + x2) / 2, min(frame.shape[0] - 1, y2))
                        pixel_zone = pixel_zone_for_point(pixel_zones, foot_point)
                        warning_state = warning.update(float(observed_time), depth_value, valid=depth_value is not None, closing=motion_label == "CLOSING", degraded=False)
                        color = {"NORMAL": (0, 200, 0), "WARNING": (0, 220, 255), "CRITICAL": (0, 0, 255), "UNKNOWN": (128, 128, 128)}[warning_state]
                        if floor_zone is not None:
                            color = {"SAFE": (0, 200, 0), "WARNING": (0, 220, 255), "DANGER": (0, 0, 255)}[floor_zone]
                        if pixel_zone is not None:
                            color = PIXEL_ZONE_BGR[pixel_zone]
                        label = f"ID {ident if ident is not None else '?'} | {confidence:.2f} | {depth_value:.2f} m" if depth_value is not None else f"ID {ident if ident is not None else '?'} | {confidence:.2f} | DISTANCE UNKNOWN"
                        if floor_zone is not None:
                            label += f" | FLOOR {floor_zone} {floor_distance:.1f}m"
                        if pixel_zone is not None:
                            label += f" | {PIXEL_ZONE_LABELS[pixel_zone]}"
                        if not bool(settings.get("hide_boxes", False)):
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            cv2.putText(annotated, f"{motion_label} | {warning_state}" + (f" | TTC* {ttc:.1f}s" if ttc is not None else ""), (x1, min(frame.shape[0] - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
                        person_records.append({
                            "timestamp": datetime.now().astimezone().isoformat(), "frame_id": frame_id,
                            "video_time_s": video_time, "person_id": int(ident) if ident is not None else None,
                            "class": "person", "confidence": float(confidence), "bbox": [x1, y1, x2, y2],
                            "raw_depth_m": raw if depth_value is not None else None,
                            "estimated_camera_depth_m": depth_value, "depth_valid": depth_value is not None,
                            "depth_valid_fraction": valid_fraction, "depth_dispersion_m": dispersion,
                            "depth_age_s": max(0.0, observed_time - latest_depth_at) if latest_depth_at is not None else None,
                            "closing_speed_mps": closing_speed, "motion_status": motion_label,
                            "experimental_ttc_s": ttc, "warning_status": warning_state,
                            "ground_position_m": {"x": ground[0], "y": ground[1]} if ground else None,
                            "floor_zone": floor_zone, "floor_distance_from_origin_m": floor_distance,
                            "pixel_zone": pixel_zone,
                        })
                if any_unknown:
                    overall = "UNKNOWN / DEGRADED"
                elif any(record["warning_status"] == "CRITICAL" for record in person_records):
                    overall = "CRITICAL"
                elif any(record["warning_status"] == "WARNING" for record in person_records):
                    overall = "WARNING"
                else:
                    overall = "NORMAL"
                elapsed_ms = (time.perf_counter() - start) * 1000
                process_times.append(elapsed_ms)
                process_times = process_times[-30:]
                proc_fps = 1000 / float(np.mean(process_times)) if process_times and np.mean(process_times) > 0 else 0
                cap_fps = 1 / float(np.mean(capture_times)) if capture_times and np.mean(capture_times) > 0 else 0
                if elapsed_ms > float(settings.get("latency_limit_ms", 1500)):
                    overall = "UNKNOWN / DEGRADED"
                    for record in person_records:
                        record["warning_status"] = "UNKNOWN"
                    warning.update(float(observed_time), None, valid=False, degraded=True)
                cv2.putText(annotated, f"{overall} | CPU/CUDA {device} | capture {cap_fps:.1f} FPS | process {proc_fps:.1f} FPS | total {elapsed_ms:.0f} ms", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(annotated, f"YOLO {getattr(result, 'speed', {}).get('inference', 0):.1f} ms | depth {depth_inference_ms:.1f} ms | camera-axis estimate only", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                if person_records:
                    with logging_path.open("a", encoding="utf-8") as log_file:
                        for record in person_records:
                            log_file.write(json.dumps(record, allow_nan=False) + "\n")
                if output_video:
                    if writer is None:
                        height, width = annotated.shape[:2]
                        fps = source.fps if source.fps > 0 else 20.0
                        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot write annotated video to {output_video}")
                    writer.write(annotated)
                if not headless:
                    cv2.imshow("Forklift Vision AI - Detection", annotated)

                    depth_view = np.zeros_like(frame)
                    min_depth = float(settings["min_depth_m"])
                    max_depth = float(settings["max_depth_m"])
                    depth_is_stale = data_is_stale(
                        latest_depth_at, observed_time, float(settings["max_depth_age_s"])
                    )
                    if latest_depth is not None:
                        valid_depth = np.isfinite(latest_depth) & (latest_depth >= min_depth) & (latest_depth <= max_depth)
                        if valid_depth.any():
                            normalized = np.zeros(latest_depth.shape, dtype=np.uint8)
                            scaled = np.clip((latest_depth[valid_depth] - min_depth) / (max_depth - min_depth), 0, 1)
                            normalized[valid_depth] = (scaled * 255).astype(np.uint8)
                            depth_view = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
                            depth_view[~valid_depth] = (0, 0, 0)
                    if pixel_zones:
                        depth_view = overlay_pixel_zones(depth_view, pixel_zones)
                    for record in person_records:
                        x1, y1, x2, y2 = record["bbox"]
                        depth_label = record["estimated_camera_depth_m"]
                        label = f"ID {record['person_id'] if record['person_id'] is not None else '?'}: {depth_label:.2f} m" if depth_label is not None else "DEPTH UNKNOWN"
                        pixel_zone = record["pixel_zone"]
                        box_color = PIXEL_ZONE_BGR[pixel_zone] if pixel_zone is not None else (230, 230, 230)
                        if pixel_zone is not None:
                            label += f" | {PIXEL_ZONE_LABELS[pixel_zone]}"
                        cv2.rectangle(depth_view, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(depth_view, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                    depth_age = max(0.0, observed_time - latest_depth_at) if latest_depth_at is not None else None
                    depth_status = "DEPTH MAP STALE / UNKNOWN" if depth_is_stale else f"DEPTH MAP | age {depth_age:.1f}s | {min_depth:.1f}-{max_depth:.1f} m"
                    cv2.putText(depth_view, depth_status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(depth_view, "Camera-axis estimate; color scale is relative to configured range", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.putText(depth_view, pixel_zone_status[:110], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.imshow("Forklift Vision AI - Depth", depth_view)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("d"):
                        settings["show_depth"] = not settings.get("show_depth", False)
                    elif key == ord("b"):
                        settings["hide_boxes"] = not settings.get("hide_boxes", False)
                    elif key == ord("r"):
                        motion.clear()
                        filters.clear()
                    elif key == ord("s"):
                        shot = ROOT / "outputs" / "recordings" / f"screenshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
                        shot.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(shot), annotated):
                            logger.error("Failed to save screenshot: %s", shot)
                        else:
                            logger.info("Saved screenshot: %s", shot)
                if elapsed_ms > float(settings.get("latency_limit_ms", 1500)):
                    logger.warning("Processing latency %.0f ms exceeds configured limit", elapsed_ms)
        finally:
            source.close()
            if writer is not None:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()
        return 0
    except (RuntimeError, ValueError, FileNotFoundError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
