"""Standalone pixel-zone calibration tool for Forklift Vision AI.

Run from the project directory with:
    python calibrate_zones.py
    python calibrate_zones.py --source 0
    python calibrate_zones.py --source "rtsp://camera/stream" --model models/yolo26n.pt
    python calibrate_zones.py --source video.mp4 --no-detector --output outputs/zones.json
    python calibrate_zones.py --source 0 --load outputs/zones.json
    python -m unittest calibrate_zones -v

Mouse: left click adds a vertex or activates toolbar buttons for zone type, finish,
undo, cancel, delete, open, save, and quit. Keyboard: 1/2/3 selects a zone,
Enter=finish polygon, Backspace=undo vertex, C=cancel current polygon,
D=delete most recent completed zone, S=save JSON/reference image, Q/Esc=quit.
Default save and Open dialog folder: calibration/; existing files are not overwritten
silently (Save creates a timestamped copy unless --overwrite is supplied).

The JSON stores image-pixel polygons only. It does not represent metric distance,
world coordinates, homography, or a safety-certified boundary. A later application
must load the JSON and require an exact frame resolution match before using points.

JSON shape:
    {"format":"forklift-vision-pixel-zones", "version":1,
     "frame_size":{"width":1280,"height":720}, "source":"rtsp://user:***@host/path",
     "created_at":"UTC ISO-8601", "zones":[
       {"name":"hazard", "color":"#FF0000", "points":[[x,y],[x,y],[x,y]]}]}

Future integration: import load_calibration(Path("zone_calibration.json"),
(frame_width, frame_height)); draw each returned zone polygon on the matching-size
frame. A resolution mismatch raises ResolutionMismatchError; no scaling is implicit.
The source metadata redacts RTSP passwords before writing the JSON.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence
import unittest
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np


ZONE_COLORS = {
    "hazard": "#FF0000",
    "warning": "#FFFF00",
    "safe": "#00FF00",
}
ZONE_LABELS = {
    "hazard": "BAHAYA",
    "warning": "WASPADA",
    "safe": "AMAN",
}
ZONE_BGR = {
    name: tuple(int(color[index:index + 2], 16) for index in (5, 3, 1))
    for name, color in ZONE_COLORS.items()
}
OVERLAY_ALPHA = 0.30
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
DEFAULT_OUTPUT = CALIBRATION_DIR / "zone_calibration.json"


class CalibrationError(ValueError):
    """Invalid or incompatible zone calibration data."""


class ResolutionMismatchError(CalibrationError):
    """Calibration pixel coordinates do not match the current frame size."""


def _cross(a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Sequence[int], b: Sequence[int], p: Sequence[int]) -> bool:
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _segments_intersect(a: Sequence[int], b: Sequence[int], c: Sequence[int], d: Sequence[int]) -> bool:
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > 0 > ab_d or ab_d > 0 > ab_c)
            and (cd_a > 0 > cd_b or cd_b > 0 > cd_a)):
        return True
    return ((ab_c == 0 and _on_segment(a, b, c))
            or (ab_d == 0 and _on_segment(a, b, d))
            or (cd_a == 0 and _on_segment(c, d, a))
            or (cd_b == 0 and _on_segment(c, d, b)))


def validate_polygon(points: Iterable[Sequence[Any]], width: int, height: int) -> list[list[int]]:
    """Validate a simple, non-degenerate polygon and return integer pixel points."""
    if width <= 0 or height <= 0:
        raise CalibrationError("Frame width and height must be positive")
    try:
        raw_points = list(points)
    except TypeError as exc:
        raise CalibrationError("Polygon points must be a list of (x, y) pairs") from exc
    if len(raw_points) < 3:
        raise CalibrationError("A zone polygon needs at least three points")

    normalized: list[list[int]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CalibrationError("Each polygon point must contain x and y")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise CalibrationError("Polygon coordinates must be numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise CalibrationError("Polygon coordinates must be finite")
        if not x.is_integer() or not y.is_integer():
            raise CalibrationError("Polygon coordinates must be integer pixel positions")
        ix, iy = int(x), int(y)
        if not (0 <= ix < width and 0 <= iy < height):
            raise CalibrationError(f"Point ({ix}, {iy}) is outside the {width}x{height} frame")
        normalized.append([ix, iy])

    if len({(p[0], p[1]) for p in normalized}) < 3:
        raise CalibrationError("A zone polygon needs at least three distinct points")
    if any(normalized[i] == normalized[(i + 1) % len(normalized)] for i in range(len(normalized))):
        raise CalibrationError("Adjacent polygon points must be different")

    twice_area = sum(
        normalized[i][0] * normalized[(i + 1) % len(normalized)][1]
        - normalized[(i + 1) % len(normalized)][0] * normalized[i][1]
        for i in range(len(normalized))
    )
    if abs(twice_area) < 2:
        raise CalibrationError("Polygon is degenerate or has zero area")

    count = len(normalized)
    for i in range(count):
        a, b = normalized[i], normalized[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i or j == (i + 1) % count or (j + 1) % count == i:
                continue
            c, d = normalized[j], normalized[(j + 1) % count]
            if _segments_intersect(a, b, c, d):
                raise CalibrationError("Polygon edges cross or touch themselves; draw a simple outline")
    return normalized


def make_zone(name: str, points: Iterable[Sequence[Any]], width: int, height: int) -> dict[str, Any]:
    if name not in ZONE_COLORS:
        raise CalibrationError(f"Unknown zone type: {name}")
    return {"name": name, "color": ZONE_COLORS[name], "points": validate_polygon(points, width, height)}


def safe_source_metadata(source: str) -> str:
    """Keep RTSP source metadata useful without persisting its password."""
    parts = urlsplit(source)
    if parts.scheme.lower() not in {"rtsp", "rtsps"} or "@" not in parts.netloc:
        return source
    user_info, host = parts.netloc.rsplit("@", 1)
    username = user_info.split(":", 1)[0]
    netloc = f"{username}:***@{host}" if username else f"***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def create_calibration(source: str, width: int, height: int,
                       zones: Iterable[dict[str, Any]], *, created_at: str | None = None) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        raise CalibrationError("Frame width and height must be positive")
    checked_zones = [make_zone(zone["name"], zone["points"], width, height) for zone in zones]
    if not checked_zones:
        raise CalibrationError("Add at least one completed zone before saving")
    return {
        "format": "forklift-vision-pixel-zones",
        "version": 1,
        "frame_size": {"width": int(width), "height": int(height)},
        "source": safe_source_metadata(source),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "zones": checked_zones,
    }


def save_calibration(path: Path, source: str, width: int, height: int,
                     zones: Iterable[dict[str, Any]], *, overwrite: bool = False) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; choose another --output or pass --overwrite")
    payload = create_calibration(source, width, height, zones)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_calibration(path: Path, frame_size: tuple[int, int]) -> dict[str, Any]:
    """Load calibration for (width, height), rejecting any resolution mismatch."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Cannot read calibration {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "forklift-vision-pixel-zones" or payload.get("version") != 1:
        raise CalibrationError("Unsupported zone calibration JSON format/version")
    stored = payload.get("frame_size")
    if not isinstance(stored, dict):
        raise CalibrationError("Calibration JSON is missing frame_size")
    expected_width, expected_height = int(frame_size[0]), int(frame_size[1])
    stored_width, stored_height = stored.get("width"), stored.get("height")
    if (stored_width, stored_height) != (expected_width, expected_height):
        raise ResolutionMismatchError(
            f"Calibration is for {stored_width}x{stored_height}, current frame is {expected_width}x{expected_height}; "
            "pixel zones were not resized or applied"
        )
    if not isinstance(payload.get("source"), str) or not isinstance(payload.get("created_at"), str):
        raise CalibrationError("Calibration JSON is missing source or created_at")
    zones = payload.get("zones")
    if not isinstance(zones, list) or not zones:
        raise CalibrationError("Calibration JSON must contain at least one zone")
    checked = []
    for zone in zones:
        if not isinstance(zone, dict) or zone.get("color") != ZONE_COLORS.get(zone.get("name")):
            raise CalibrationError("Zone name/color is invalid")
        checked.append(make_zone(zone["name"], zone.get("points", []), expected_width, expected_height))
    payload["zones"] = checked
    return payload


def find_overlaps(zones: Sequence[dict[str, Any]], width: int, height: int) -> list[tuple[int, int]]:
    """Return pairs of zone indices whose filled pixel areas overlap."""
    overlaps: list[tuple[int, int]] = []
    for i in range(len(zones)):
        first = np.asarray(zones[i]["points"], dtype=np.int32)
        for j in range(i + 1, len(zones)):
            second = np.asarray(zones[j]["points"], dtype=np.int32)
            all_points = np.vstack((first, second))
            x0 = max(0, int(all_points[:, 0].min()))
            y0 = max(0, int(all_points[:, 1].min()))
            x1 = min(width, int(all_points[:, 0].max()) + 1)
            y1 = min(height, int(all_points[:, 1].max()) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            shape = (y1 - y0, x1 - x0)
            mask_a = np.zeros(shape, dtype=np.uint8)
            mask_b = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(mask_a, [first - (x0, y0)], 255)
            cv2.fillPoly(mask_b, [second - (x0, y0)], 255)
            if cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b)) > 0:
                overlaps.append((i, j))
    return overlaps


def _source_value(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def capture_one_frame(source: str) -> np.ndarray:
    capture = cv2.VideoCapture(_source_value(source))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source {source!r}; check camera, video path, RTSP URL, and permissions")
        ok, frame = capture.read()  # Exactly one frame is requested.
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"Source {source!r} opened but returned an empty/unreadable frame")
        return frame
    finally:
        capture.release()


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_source(source: str | None, config_path: str | None = None) -> str:
    """Use --source when supplied; otherwise read source from project config."""
    if source is not None:
        resolved = source.strip()
        if resolved:
            return resolved
        raise CalibrationError("--source cannot be empty")
    root = _project_root()
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_absolute():
            path = root / path
    else:
        is_jetson = Path("/etc/nv_tegra_release").is_file()
        path = root / ("config.jetson.json" if is_jetson else "config.json")
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Cannot read source from config {path}: {exc}") from exc
    configured_source = settings.get("source") if isinstance(settings, dict) else None
    if configured_source is None or not str(configured_source).strip():
        raise CalibrationError(f"Configuration {path} has no usable 'source'; pass --source to override")
    return str(configured_source).strip()


def _default_model() -> Path | None:
    root = _project_root()
    for config_name in ("config.json", "config.jetson.json"):
        try:
            settings = json.loads((root / config_name).read_text(encoding="utf-8"))
            model = settings.get("yolo_model")
            if model:
                path = Path(model).expanduser()
                if not path.is_absolute():
                    path = root / path
                if path.is_file():
                    return path
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
    return None


def detect_people(frame: np.ndarray, model_path: Path | None, confidence: float,
                  disabled: bool = False) -> tuple[list[tuple[int, int, int, int, float]], str]:
    if disabled:
        return [], "Detector disabled; manual calibration mode"
    if model_path is None:
        return [], "No existing detector weights found; manual calibration mode"
    if not model_path.is_file():
        return [], f"Detector model not found: {model_path}; manual calibration mode"
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        results = model.predict(source=frame, conf=confidence, classes=[0], verbose=False)
        result = results[0]
        boxes = result.boxes
        detections: list[tuple[int, int, int, int, float]] = []
        if boxes is not None and len(boxes):
            coords = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            height, width = frame.shape[:2]
            for box, score in zip(coords, confidences):
                x1, y1, x2, y2 = (int(round(float(v))) for v in box)
                detections.append((max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2), float(score)))
        return detections, f"Person detections: {len(detections)} | model: {model_path.name}"
    except ImportError:
        return [], "Ultralytics is unavailable; manual calibration mode"
    except Exception as exc:  # Keep the calibration UI usable if a local model cannot load.
        return [], f"Detector failed ({exc}); manual calibration mode"


def _zone_centroid(points: Sequence[Sequence[int]]) -> tuple[int, int]:
    contour = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return tuple(int(round(v)) for v in np.mean(np.asarray(points), axis=0))
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def render_frame(frame: np.ndarray, zones: Sequence[dict[str, Any]],
                 draft: Sequence[Sequence[int]], draft_name: str,
                 detections: Sequence[tuple[int, int, int, int, float]],
                 message: str, button_rects: list[tuple[str, tuple[int, int, int, int]]]) -> np.ndarray:
    rendered = frame.copy()
    overlay = rendered.copy()
    for zone in zones:
        polygon = np.asarray(zone["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], ZONE_BGR[zone["name"]])
    rendered = cv2.addWeighted(overlay, OVERLAY_ALPHA, rendered, 1.0 - OVERLAY_ALPHA, 0)
    for zone in zones:
        polygon = np.asarray(zone["points"], dtype=np.int32)
        color = ZONE_BGR[zone["name"]]
        cv2.polylines(rendered, [polygon], True, color, 2, cv2.LINE_AA)
        cx, cy = _zone_centroid(zone["points"])
        cv2.putText(rendered, ZONE_LABELS[zone["name"]], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    for index, point in enumerate(draft):
        cv2.circle(rendered, tuple(point), 5, ZONE_BGR[draft_name], -1)
        if index:
            cv2.line(rendered, tuple(draft[index - 1]), tuple(point), ZONE_BGR[draft_name], 2, cv2.LINE_AA)
    if len(draft) > 1:
        cv2.line(rendered, tuple(draft[-1]), tuple(draft[0]), ZONE_BGR[draft_name], 1, cv2.LINE_AA)
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(rendered, f"PERSON {score:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    banner_height = min(98, rendered.shape[0])
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], banner_height), (0, 0, 0), -1)
    cv2.putText(rendered, "1 DANGER/BAHAYA | 2 WARNING/WASPADA | 3 SAFE/AMAN", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, message[:140], (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, f"Current zone: {ZONE_LABELS[draft_name]} | completed polygons: {len(zones)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    specs = [
        ("BAHAYA", "zone:hazard", ZONE_BGR["hazard"]),
        ("WASPADA", "zone:warning", ZONE_BGR["warning"]),
        ("AMAN", "zone:safe", ZONE_BGR["safe"]),
        ("SELESAI", "finish", (110, 90, 35)),
        ("UNDO", "undo", (90, 90, 90)),
        ("BATAL", "cancel", (90, 90, 90)),
        ("HAPUS ZONA", "delete", (90, 90, 90)),
        ("BUKA", "open", (150, 80, 30)),
        ("SIMPAN", "save", (40, 120, 20)),
        ("KELUAR", "quit", (60, 60, 60)),
    ]
    button_rects.clear()
    x, y, button_height, gap = 8, 80, 30, 5
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    for label, action, color in specs:
        text_size = cv2.getTextSize(label, font, scale, thickness)[0]
        button_width = max(62, text_size[0] + 16)
        if x + button_width > rendered.shape[1] - 8:
            x, y = 8, y + button_height + gap
        x2, y2 = min(rendered.shape[1] - 1, x + button_width), min(rendered.shape[0] - 1, y + button_height)
        if x2 > x and y2 > y:
            cv2.rectangle(rendered, (x, y), (x2, y2), color, -1)
            cv2.rectangle(rendered, (x, y), (x2, y2), (230, 230, 230), 1)
            cv2.putText(rendered, label, (x + 8, y + 21), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            button_rects.append((action, (x, y, x2, y2)))
        x = x2 + gap
    return rendered


def _reference_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_reference.jpg")


def _non_overwriting_output(output: Path, overwrite: bool) -> Path:
    if overwrite or not (output.exists() or _reference_path(output).exists()):
        return output
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output.with_name(f"{output.stem}_{stamp}{output.suffix}")
    counter = 1
    while candidate.exists() or _reference_path(candidate).exists():
        candidate = output.with_name(f"{output.stem}_{stamp}_{counter}{output.suffix}")
        counter += 1
    return candidate


def _choose_calibration_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("Open file picker is unavailable (Tkinter is not installed); use --load PATH") from exc
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        return filedialog.askopenfilename(
            title="Open zone calibration",
            initialdir=str(CALIBRATION_DIR),
            filetypes=(("Zone calibration JSON", "*.json"), ("All files", "*.*")),
        ) or None
    finally:
        root.destroy()


def _save_editor_state(source: str, width: int, height: int, zones: Sequence[dict[str, Any]],
                       output: Path, image: np.ndarray, overwrite: bool) -> tuple[Path, Path]:
    payload = create_calibration(source, width, height, zones)
    overlaps = find_overlaps(zones, width, height)
    if overlaps:
        print("Warning: calibration contains overlapping zones; review the saved reference image.")
    target = _non_overwriting_output(output, overwrite)
    reference = _reference_path(target)
    _write_outputs(target, reference, payload, image)
    return target, reference


def _write_outputs(output: Path, reference: Path, payload: dict[str, Any], image: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_reference: Path | None = None
    temporary_json: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=reference.suffix, dir=reference.parent, delete=False) as temp_image:
            temporary_reference = Path(temp_image.name)
        if not cv2.imwrite(str(temporary_reference), image):
            raise OSError(f"Could not write reference frame: {reference}")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=output.suffix,
                                         dir=output.parent, delete=False) as temp_file:
            temporary_json = Path(temp_file.name)
            temp_file.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary_json, output)
        temporary_json = None
        os.replace(temporary_reference, reference)
        temporary_reference = None
    finally:
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)
        if temporary_reference is not None:
            temporary_reference.unlink(missing_ok=True)


def run_editor(frame: np.ndarray, source: str, output: Path,
               detections: Sequence[tuple[int, int, int, int, float]],
               initial_zones: list[dict[str, Any]], overwrite: bool) -> int:
    height, width = frame.shape[:2]
    zones = list(initial_zones)
    draft: list[tuple[int, int]] = []
    state: dict[str, Any] = {
        "selected": "hazard", "message": "Click a zone button, draw its polygon, then click Finish or press Enter.",
        "action": None, "button_rects": [], "dirty": False,
    }

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, (x1, y1, x2, y2) in state["button_rects"]:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    state["action"] = action
                    return
            if 0 <= x < width and 0 <= y < height:
                draft.append((x, y))
                state["dirty"] = True
                state["message"] = f"Point added ({x}, {y}). Press Enter to finish this polygon."

    window_name = "Forklift Vision AI - Zone Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Calibration UI opened. Use the on-screen buttons or keyboard shortcuts.")
    print(f"Calibration folder: {CALIBRATION_DIR}\nDefault JSON output: {output}")
    initial_overlaps = find_overlaps(zones, width, height)
    if initial_overlaps:
        state["message"] = "Warning: loaded calibration contains overlapping zones."
        print(state["message"])
    try:
        while True:
            display = render_frame(
                frame, zones, draft, state["selected"], detections,
                state["message"], state["button_rects"],
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            action = state.pop("action", None)
            if action is not None:
                if action.startswith("zone:"):
                    if draft:
                        state["message"] = "Finish or cancel the current polygon before changing zone type."
                    else:
                        state["selected"] = action.split(":", 1)[1]
                        state["message"] = f"Selected {ZONE_LABELS[state['selected']]}. Draw its polygon."
                elif action == "undo":
                    if draft:
                        draft.pop()
                        state["dirty"] = True
                        state["message"] = "Removed the last polygon point."
                    else:
                        state["message"] = "No unfinished point to undo."
                elif action == "finish":
                    try:
                        zone = make_zone(state["selected"], draft, width, height)
                        zones.append(zone)
                        draft.clear()
                        state["dirty"] = True
                        overlaps = find_overlaps(zones, width, height)
                        if overlaps:
                            descriptions = [f"{ZONE_LABELS[zones[i]['name']]} / {ZONE_LABELS[zones[j]['name']]}" for i, j in overlaps]
                            state["message"] = "Warning: overlapping zones: " + ", ".join(descriptions)
                            print(state["message"])
                        else:
                            state["message"] = f"Added {ZONE_LABELS[zone['name']]} polygon."
                    except CalibrationError as exc:
                        state["message"] = str(exc)
                elif action == "cancel":
                    draft.clear()
                    state["message"] = "Current unfinished polygon cancelled."
                elif action == "delete":
                    if zones:
                        removed = zones.pop()
                        state["dirty"] = True
                        state["message"] = f"Deleted most recent {ZONE_LABELS[removed['name']]} zone."
                    else:
                        state["message"] = "There are no completed zones to delete."
                elif action == "open":
                    try:
                        selected_file = _choose_calibration_file()
                        if selected_file:
                            loaded = load_calibration(Path(selected_file), (width, height))
                            zones[:] = loaded["zones"]
                            draft.clear()
                            state["dirty"] = False
                            state["message"] = f"Opened {Path(selected_file).name}; resolution matches {width}x{height}."
                        else:
                            state["message"] = "Open cancelled; current zones unchanged."
                    except Exception as exc:
                        state["message"] = f"Open failed: {exc}"
                elif action == "save":
                    if draft:
                        state["message"] = "Finish the current polygon or cancel it before saving."
                    else:
                        try:
                            target, reference = _save_editor_state(
                                source, width, height, zones, output, display, overwrite
                            )
                            output = target
                            state["dirty"] = False
                            state["message"] = f"Saved {target.name} and {reference.name} in calibration/."
                            print(f"Saved calibration: {target}\nSaved reference frame: {reference}")
                        except (CalibrationError, OSError, cv2.error) as exc:
                            state["message"] = f"Save failed: {exc}"
                            print(state["message"])
                elif action == "quit":
                    if state["dirty"]:
                        print("Exiting; unsaved edits were discarded.")
                    return 0
                continue

            if key in (ord("1"), ord("2"), ord("3")):
                if draft:
                    state["message"] = "Finish (Enter) or cancel (C) the current polygon before changing zone type."
                else:
                    state["selected"] = {ord("1"): "hazard", ord("2"): "warning", ord("3"): "safe"}[key]
                    state["message"] = f"Selected {ZONE_LABELS[state['selected']]} zone. Click to draw its polygon."
            elif key in (10, 13):
                try:
                    zone = make_zone(state["selected"], draft, width, height)
                    zones.append(zone)
                    draft.clear()
                    state["dirty"] = True
                    overlaps = find_overlaps(zones, width, height)
                    if overlaps:
                        descriptions = [f"{ZONE_LABELS[zones[i]['name']]} / {ZONE_LABELS[zones[j]['name']]}" for i, j in overlaps]
                        state["message"] = "Warning: overlapping zones: " + ", ".join(descriptions)
                        print(state["message"])
                    else:
                        state["message"] = f"Added {ZONE_LABELS[zone['name']]} polygon. Draw another zone or press S to save."
                except CalibrationError as exc:
                    state["message"] = str(exc)
            elif key in (8, 127):
                if draft:
                    point = draft.pop()
                    state["message"] = f"Removed last point {point}."
                else:
                    state["message"] = "No unfinished point to undo."
            elif key in (ord("c"), ord("C")):
                draft.clear()
                state["message"] = "Current unfinished polygon cancelled. Completed zones remain."
            elif key in (ord("d"), ord("D")):
                if zones:
                    removed = zones.pop()
                    state["dirty"] = True
                    state["message"] = f"Deleted most recent {ZONE_LABELS[removed['name']]} zone."
                else:
                    state["message"] = "There are no completed zones to delete."
            elif key in (ord("s"), ord("S")):
                if draft:
                    state["message"] = "Finish the current polygon with Enter or cancel it with C before saving."
                    continue
                try:
                    target, reference = _save_editor_state(
                        source, width, height, zones, output, display, overwrite
                    )
                    output = target
                    state["dirty"] = False
                    state["message"] = f"Saved {target.name} and {reference.name} in calibration/."
                    print(f"Saved calibration: {target}\nSaved reference frame: {reference}")
                except (CalibrationError, OSError, cv2.error) as exc:
                    state["message"] = f"Save failed: {exc}"
                    print(state["message"])
            elif key in (ord("q"), 27):
                if state["dirty"]:
                    print("Calibration cancelled; unsaved edits were discarded.")
                return 0
    finally:
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw pixel-based hazard, warning, and safe zones on one camera/video frame")
    parser.add_argument("--source", help="camera index, video path, or RTSP URL; overrides the project config")
    parser.add_argument("--config", help="settings JSON to read source from; defaults to config.json or config.jetson.json on Jetson")
    detector = parser.add_mutually_exclusive_group()
    detector.add_argument("--model", help="existing local Ultralytics detector weights; no download is attempted")
    detector.add_argument("--no-detector", action="store_true", help="skip person detection and calibrate manually")
    parser.add_argument("--conf", type=float, default=0.35, help="person detection confidence threshold (default: 0.35)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output JSON path (default: calibration/zone_calibration.json)")
    parser.add_argument("--load", help="load existing JSON zones, requiring an exact frame resolution match")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing output JSON/reference files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.conf <= 1.0:
        print("ERROR: --conf must be in (0, 1].")
        return 2
    output = Path(args.output).expanduser()
    try:
        source = resolve_source(args.source, args.config)
    except CalibrationError as exc:
        print(f"ERROR: {exc}")
        return 2
    try:
        frame = capture_one_frame(source)
        height, width = frame.shape[:2]
        if args.load:
            initial = load_calibration(Path(args.load).expanduser(), (width, height))["zones"]
            print(f"Loaded {len(initial)} zones from {args.load} for {width}x{height} frame.")
        else:
            initial = []
        model_path = Path(args.model).expanduser() if args.model else (None if args.no_detector else _default_model())
        if model_path is not None and not model_path.is_absolute():
            model_path = _project_root() / model_path
        detections, detector_status = detect_people(frame, model_path, args.conf, args.no_detector)
        print(detector_status)
        return run_editor(frame, source, output, detections, initial, args.overwrite)
    except ResolutionMismatchError as exc:
        print(f"ERROR: {exc}")
        return 2
    except CalibrationError as exc:
        print(f"ERROR: {exc}")
        return 2
    except (RuntimeError, OSError, cv2.error) as exc:
        print(f"ERROR: {exc}")
        return 2
    except KeyboardInterrupt:
        print("Calibration cancelled by user.")
        cv2.destroyAllWindows()
        return 130


class PolygonValidationTests(unittest.TestCase):
    def test_source_uses_config_and_strips_whitespace_or_accepts_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"source": "  rtsp://camera/stream  "}), encoding="utf-8")
            self.assertEqual(resolve_source(None, str(path)), "rtsp://camera/stream")
            self.assertEqual(resolve_source("  0  ", str(path)), "0")

    def test_valid_triangle(self) -> None:
        self.assertEqual(validate_polygon([(1, 1), (8, 1), (4, 8)], 10, 10), [[1, 1], [8, 1], [4, 8]])

    def test_less_than_three_points_is_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            validate_polygon([(1, 1), (5, 5)], 10, 10)

    def test_degenerate_and_self_crossing_polygons_are_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            validate_polygon([(1, 1), (5, 1), (9, 1)], 10, 10)
        with self.assertRaises(CalibrationError):
            validate_polygon([(1, 1), (8, 8), (1, 8), (8, 1)], 10, 10)

    def test_coordinates_outside_frame_are_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            validate_polygon([(-1, 1), (8, 1), (4, 8)], 10, 10)
        with self.assertRaises(CalibrationError):
            validate_polygon([(1, 1), (10, 1), (4, 8)], 10, 10)

    def test_json_serialization_round_trip(self) -> None:
        zone = make_zone("warning", [(1, 1), (8, 1), (8, 8), (1, 8)], 10, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            save_calibration(path, "video.mp4", 10, 10, [zone])
            loaded = load_calibration(path, (10, 10))
        self.assertEqual(loaded["source"], "video.mp4")
        self.assertEqual(loaded["zones"][0], zone)

    def test_resolution_mismatch_is_rejected(self) -> None:
        zone = make_zone("safe", [(1, 1), (8, 1), (8, 8)], 10, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            save_calibration(path, "camera", 10, 10, [zone])
            with self.assertRaises(ResolutionMismatchError):
                load_calibration(path, (20, 10))


if __name__ == "__main__":
    raise SystemExit(main())
