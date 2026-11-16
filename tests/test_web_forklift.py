import json
import io
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from forklift_vision.runtime import VideoSource

try:
    import web_forklift
    from web_forklift import (
        CameraPipeline, SnapshotCapacityError, ai_resolution_choices, classify_pixel_zone,
        create_app, render_clean_reference, render_depth_view, scaled_ai_resolution,
        validate_zones_payload,
    )
except ModuleNotFoundError as exc:
    if exc.name == "flask":
        raise unittest.SkipTest("Flask is declared in requirements.txt but is not installed")
    raise


class FakePipeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        self.latest_annotated = self.latest_frame.copy()
        self.latest_detections = [{"id": 9, "confidence": .9, "bbox": [1, 1, 5, 5]}]
        self.settings = {"source": "camera"}
        self.people = list(self.latest_detections)
        self.status = {"state": "running", "frame_id": 17, "zone_status": "not_available", "zone_message": "Kalibrasi belum tersedia", "zone_types": []}
        self.updated_at = time.time()
        self.zones = None
        self.calibration_snapshots = {}
        self.views = {"camera": b"camera-jpeg", "ai": b"ai-jpeg", "depth": b"depth-jpeg"}
        self.distance_calibration_path = Path("distance-calibration.json")
        self.distance_calibration = None
        self.distance_calibration_status = "not_available"
        self.distance_calibration_message = "Distance calibration not created"

    def snapshot(self):
        return b"jpeg", dict(self.status)

    def view_snapshot(self, view, frame_id=None):
        status = dict(self.status)
        if frame_id is not None and frame_id != status.get("frame_id"):
            status["requested_frame_id"] = frame_id
            return None, status
        return self.views.get(view), status

    def capture_calibration_snapshot(self):
        with self.lock:
            raw = self.latest_frame.copy()
            ok, encoded = cv2.imencode(".jpg", raw)
            if not ok:
                return None
            h, w = raw.shape[:2]
            snapshot_id = "test-snapshot"
            self.calibration_snapshots[snapshot_id] = {"frame": raw, "jpeg": encoded.tobytes(), "people": [dict(p) for p in self.latest_detections],
                                                       "captured_at": "2026-01-01T00:00:00+00:00", "frame_id": 3,
                                                       "frame_size": {"width": w, "height": h}, "source": "camera", "snapshot_id": snapshot_id,
                                                       "_created_monotonic": time.monotonic()}
            return {k: v for k, v in self.calibration_snapshots[snapshot_id].items() if k not in {"frame", "jpeg", "_created_monotonic"}}

    def _get_calibration_snapshot_locked(self, snapshot_id):
        return self.calibration_snapshots.get(snapshot_id or "")

    def get_calibration_snapshot(self, snapshot_id):
        with self.lock:
            snapshot = self.calibration_snapshots.get(snapshot_id or "")
            if snapshot is None:
                return None, None
            return snapshot["jpeg"], {k: v for k, v in snapshot.items() if k not in {"frame", "jpeg", "_created_monotonic"}}

    def cancel_calibration_snapshot(self, snapshot_id):
        self.calibration_snapshots.pop(snapshot_id or "", None)


class WebZoneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.zone_path = Path(self.temp.name) / "zones.json"
        self.pipeline = FakePipeline()
        self.client = create_app(self.pipeline, self.zone_path).test_client()

    def payload(self, zones=None, **extra):
        return {"zones": zones or [{"name": "hazard", "color": "#FF0000", "points": [[1, 1], [8, 1], [8, 8]]}], **extra}

    def test_monitoring_can_select_camera_ai_and_depth_views(self):
        for view, expected in (("camera", b"camera-jpeg"), ("ai", b"ai-jpeg"), ("depth", b"depth-jpeg")):
            with self.subTest(view=view):
                response = self.client.get(f"/api/frame.jpg?view={view}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, expected)
        self.assertEqual(self.client.get("/api/frame.jpg?view=unknown").status_code, 400)
        page_response = self.client.get("/")
        page = page_response.get_data(as_text=True)
        self.assertIn("no-store", page_response.headers["Cache-Control"])
        self.assertIn('id="frameView"', page)
        self.assertIn("Peta kedalaman AI", page)
        self.assertIn("Estimasi jarak:", page)
        self.assertIn("frame_id=${encodeURIComponent(frameId)}", page)
        self.assertIn("X-Frame-ID", page)
        self.assertIn('id="personFrameStamp"', page)
        self.assertIn("zone-chip", page)
        self.assertIn("Estimasi jarak:", page)
        self.assertNotIn("Depth ${esc(p.depth_status", page)
        self.assertNotIn("Gerak: ${esc(p.motion", page)
        self.assertIn("await nextImage.decode()", page)
        self.assertIn("camera.replaceWith(nextImage)", page)

    def test_frame_endpoint_returns_exact_requested_frame_and_rejects_missing_id(self):
        exact = self.client.get("/api/frame.jpg?view=ai&frame_id=17")
        self.assertEqual(exact.status_code, 200)
        self.assertEqual(exact.headers["X-Frame-ID"], "17")
        self.assertEqual(exact.data, b"ai-jpeg")
        missing = self.client.get("/api/frame.jpg?view=ai&frame_id=16")
        self.assertEqual(missing.status_code, 409)
        self.assertEqual(missing.json["latest_frame_id"], 17)
        invalid = self.client.get("/api/frame.jpg?view=ai&frame_id=bad")
        self.assertEqual(invalid.status_code, 400)

    def test_foot_point_is_classified_against_calibrated_pixel_zones(self):
        calibration = {"zones": [
            {"name": "safe", "points": [[20, 20], [80, 20], [80, 80], [20, 80]]},
            {"name": "warning", "points": [[40, 40], [60, 40], [60, 60], [40, 60]]},
            {"name": "hazard", "points": [[45, 45], [55, 45], [55, 55], [45, 55]]},
        ]}
        self.assertEqual(classify_pixel_zone((30, 30), calibration), "SAFE")
        self.assertEqual(classify_pixel_zone((42, 42), calibration), "WARNING")
        self.assertEqual(classify_pixel_zone((50, 50), calibration), "HAZARD")
        self.assertEqual(classify_pixel_zone((5, 5), calibration), "OUTSIDE_ZONES")
        self.assertEqual(classify_pixel_zone((30, 30), None), "UNCALIBRATED")
        self.assertEqual(classify_pixel_zone(None, calibration), "UNKNOWN")

    def test_depth_view_renders_depth_map_and_foot_reference_marker(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.full((100, 100), 3.0, dtype=np.float32)
        person = {"id": 2, "bbox": [20, 10, 80, 90], "foot_point_px": [50, 89], "depth_m": 3.0}
        view = render_depth_view(frame, depth, [person], 0.1, 20.0)
        unavailable = render_depth_view(frame, None, [person], 0.1, 20.0)
        self.assertEqual(view.shape, frame.shape)
        self.assertEqual(unavailable.shape, frame.shape)
        self.assertFalse(np.array_equal(view, unavailable))

    def test_server_side_polygon_and_schema_validation(self):
        result = validate_zones_payload(self.payload(), 10, 10, "camera")
        self.assertEqual(result["format"], "forklift-vision-pixel-zones")
        self.assertEqual(result["frame_size"], {"width": 10, "height": 10})
        bad = self.payload(zones=[{"name": "hazard", "points": [[-1, 0], [3, 0], [3, 3]]}])
        with self.assertRaises(ValueError):
            validate_zones_payload(bad, 10, 10, "camera")

    def test_snapshot_stays_frozen_during_edit(self):
        self.client.post("/api/calibration/snapshot")
        first, meta = self.pipeline.get_calibration_snapshot("test-snapshot")
        self.pipeline.latest_frame[:] = 255
        self.pipeline.latest_detections = []
        second, same_meta = self.pipeline.get_calibration_snapshot("test-snapshot")
        self.assertEqual(first, second)
        self.assertEqual(meta, same_meta)
        self.assertEqual(meta["people"][0]["id"], 9)

    def test_pipeline_keeps_separate_frozen_snapshots_per_browser(self):
        pipeline = CameraPipeline({"source": "camera"}, self.zone_path)
        pipeline.latest_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        pipeline.updated_at = time.time()
        pipeline.settings["source"] = "camera"
        first = pipeline.capture_calibration_snapshot()
        pipeline.latest_frame[:] = 200
        pipeline.people = [{"id": 2}]
        pipeline.latest_detections = list(pipeline.people)
        second = pipeline.capture_calibration_snapshot()
        first_jpeg, first_meta = pipeline.get_calibration_snapshot(first["snapshot_id"])
        second_jpeg, second_meta = pipeline.get_calibration_snapshot(second["snapshot_id"])
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertNotEqual(first_jpeg, second_jpeg)
        self.assertEqual(first_meta["people"], [])
        self.assertEqual(second_meta["people"], [{"id": 2}])

    def test_snapshot_metadata_includes_capture_time_and_resolution(self):
        response = self.client.post("/api/calibration/snapshot")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["frame_size"], {"width": 10, "height": 10})
        self.assertTrue(response.json["captured_at"])
        sid = response.json["snapshot_id"]
        self.assertEqual(self.client.get(f"/api/calibration/snapshot.jpg?snapshot_id={sid}").status_code, 200)

    def test_css_resize_does_not_change_saved_pixel_coordinates(self):
        self.client.post("/api/calibration/snapshot")
        zones = [{"name": "hazard", "color": "#FF0000", "points": [[1, 2], [8, 2], [8, 7]]}]
        response = self.client.post("/api/calibration/save", json={"zones": zones, "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 200, response.json)
        saved = json.loads(self.zone_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["zones"][0]["points"], [[1, 2], [8, 2], [8, 7]])
        page = self.client.get("/calibration").get_data(as_text=True)
        self.assertIn("(event.clientX-rect.left)*w/rect.width", page)

    def test_reference_is_clean_and_detection_overlay_is_optional(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        zones = [{"name": "hazard", "points": [[7, 7], [9, 7], [9, 9]]}]
        with_boxes = render_clean_reference(frame, zones, [{"bbox": [1, 1, 5, 5]}])
        without_boxes = render_clean_reference(frame, zones, None)
        self.assertFalse(np.array_equal(with_boxes[1, 1], without_boxes[1, 1]))
        self.assertTrue(np.array_equal(with_boxes[0, 9], without_boxes[0, 9]))

    def test_saved_reference_can_omit_or_include_detection_boxes(self):
        self.client.post("/api/calibration/snapshot")
        zones = [{"name": "hazard", "color": "#FF0000", "points": [[7, 7], [9, 7], [9, 9]]}]
        self.client.post("/api/calibration/save", json={"zones": zones, "snapshot_id": "test-snapshot", "include_detections": False})
        ref = self.zone_path.with_name("zones_reference.jpg")
        clean = cv2.imread(str(ref))
        self.assertLess(int(clean[1, 1].max()), 50)
        self.pipeline.capture_calibration_snapshot()
        self.client.post("/api/calibration/save", json={"zones": zones, "snapshot_id": "test-snapshot", "include_detections": True})
        boxed = cv2.imread(str(ref))
        self.assertGreater(int(boxed[1, 1].min()), 180)

    def test_no_snapshot_blocks_save_and_empty_zones_not_safe(self):
        response = self.client.post("/api/calibration/save", json={"zones": [], "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 410)
        self.client.post("/api/calibration/snapshot")
        response = self.client.post("/api/calibration/save", json={"zones": [], "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 400)

    def test_invalid_and_resolution_mismatch_messages_are_visible(self):
        self.zone_path.write_text("not json", encoding="utf-8")
        self.client.post("/api/calibration/snapshot")
        invalid = self.client.get("/api/calibration/load?snapshot_id=test-snapshot")
        self.assertEqual(invalid.json["zone_status"], "invalid")
        status = self.client.get("/api/status").get_json()
        self.assertEqual(status["zone_status"], "invalid")
        page = self.client.get("/").get_data(as_text=True)
        self.assertIn("File kalibrasi invalid", page)

        data = {"format": "forklift-vision-pixel-zones", "version": 1, "frame_size": {"width": 11, "height": 10},
                "source": "camera", "created_at": "now", "zones": [{"name": "hazard", "color": "#FF0000", "points": [[1, 1], [8, 1], [8, 8]]}]}
        self.zone_path.write_text(json.dumps(data), encoding="utf-8")
        mismatch = self.client.get("/api/calibration/load?snapshot_id=test-snapshot")
        self.assertEqual(mismatch.json["zone_status"], "resolution_mismatch")
        self.assertEqual(self.client.get("/api/status").json["zone_status"], "resolution_mismatch")

    def test_pair_write_failure_preserves_previous_files(self):
        self.client.post("/api/calibration/snapshot")
        self.zone_path.write_bytes(b"old-json")
        ref = self.zone_path.with_name("zones_reference.jpg")
        ref.write_bytes(b"old-image")
        real_replace = web_forklift.os.replace
        calls = 0

        def fail_second_replace(src, dst):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated reference commit failure")
            return real_replace(src, dst)

        with mock.patch.object(web_forklift.os, "replace", side_effect=fail_second_replace):
            response = self.client.post("/api/calibration/save", json={**self.payload(), "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.zone_path.read_bytes(), b"old-json")
        self.assertEqual(ref.read_bytes(), b"old-image")

    def test_reference_encoding_failure_keeps_old_pair(self):
        self.client.post("/api/calibration/snapshot")
        self.zone_path.write_bytes(b"old-json")
        ref = self.zone_path.with_name("zones_reference.jpg")
        ref.write_bytes(b"old-image")
        with mock.patch.object(web_forklift.cv2, "imencode", return_value=(False, None)):
            response = self.client.post("/api/calibration/save", json={**self.payload(), "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(self.zone_path.read_bytes(), b"old-json")
        self.assertEqual(ref.read_bytes(), b"old-image")

    def test_hazard_only_file_is_not_filled_with_other_zones(self):
        self.client.post("/api/calibration/snapshot")
        response = self.client.post("/api/calibration/save", json={**self.payload(), "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["zone_types"], ["hazard"])
        self.assertEqual(response.json["zone_counts"], {"hazard": 1, "warning": 0, "safe": 0})
        self.assertFalse(response.json["complete"])
        self.client.post("/api/calibration/snapshot")
        loaded = self.client.get("/api/calibration/load?snapshot_id=test-snapshot").get_json()
        self.assertEqual([z["name"] for z in loaded["zones"]], ["hazard"])
        self.assertEqual(self.client.get("/api/status").json["zone_types"], ["hazard"])

    def test_capture_ui_requires_explicit_choice_and_does_not_clear_polygons(self):
        page = self.client.get("/calibration").get_data(as_text=True)
        handler = page.split("async function captureSnapshot()", 1)[1].split("async function releaseSnapshot()", 1)[0]
        self.assertIn("if(decision==='discard'){zones=[];draft=[]", handler)
        self.assertNotIn("zones=[];draft=[];", handler.replace("if(decision==='discard'){zones=[];draft=[];", ""))
        self.assertIn("chooseReplacement(sameResolution)", handler)
        self.assertIn("hasEditorState?(dirty||!sameResolution?'ask':'preserve'):'replace'", handler)
        self.assertIn("decision==='discard'", handler)
        self.assertIn("sameResolution", handler)
        for label in ("Lanjutkan tanpa mengganti snapshot", "buang perubahan", "pertahankan semua poligon"):
            self.assertIn(label, page)

    def test_zone_type_buttons_only_change_next_polygon_type(self):
        page = self.client.get("/calibration").get_data(as_text=True)
        handler = page.split("document.querySelectorAll('[data-zone]')", 1)[1].split("document.querySelector('#finish')", 1)[0]
        self.assertIn("selected=button.dataset.zone", handler)
        self.assertNotIn("zones.splice", handler)
        self.assertNotIn("zones=[]", handler)

    def test_save_ui_serializes_every_editor_polygon_and_shows_checklist(self):
        page = self.client.get("/calibration").get_data(as_text=True)
        self.assertIn("zones:zones.map(z=>({name:z.name", page)
        self.assertIn("zoneChecklist", page)
        self.assertIn("Kalibrasi belum lengkap", page)
        self.assertIn("zone_counts", page)

    def test_multiple_polygons_of_all_types_round_trip_and_report_counts_and_path(self):
        self.client.post("/api/calibration/snapshot")
        zones = [
            {"name": "hazard", "color": "#FF0000", "points": [[1, 1], [3, 1], [1, 3]]},
            {"name": "hazard", "color": "#FF0000", "points": [[1, 4], [3, 4], [1, 6]]},
            {"name": "warning", "color": "#FFFF00", "points": [[4, 1], [6, 1], [4, 3]]},
            {"name": "safe", "color": "#00FF00", "points": [[7, 7], [9, 7], [7, 9]]},
        ]
        saved = self.client.post("/api/calibration/save", json={"zones": zones, "snapshot_id": "test-snapshot"})
        self.assertEqual(saved.status_code, 200, saved.json)
        self.assertEqual(saved.json["zone_counts"], {"hazard": 2, "warning": 1, "safe": 1})
        self.assertEqual(saved.json["zone_total"], 4)
        self.assertTrue(saved.json["complete"])
        self.assertEqual(Path(saved.json["path"]), self.zone_path.resolve())
        self.assertEqual(self.client.get("/api/status").json["zone_calibration"], str(self.zone_path.resolve()))

        stored = json.loads(self.zone_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["zones"], zones)
        self.client.post("/api/calibration/snapshot")
        loaded = self.client.get("/api/calibration/load?snapshot_id=test-snapshot")
        self.assertEqual(loaded.status_code, 200, loaded.json)
        self.assertEqual(loaded.json["zones"], zones)

    def test_invalid_polygon_does_not_modify_existing_calibration_files(self):
        old_json = b"preserve existing calibration bytes"
        old_image = b"preserve existing reference bytes"
        self.zone_path.write_bytes(old_json)
        reference_path = self.zone_path.with_name("zones_reference.jpg")
        reference_path.write_bytes(old_image)
        self.client.post("/api/calibration/snapshot")
        invalid = [{"name": "hazard", "color": "#FF0000", "points": [[-1, 1], [8, 1], [8, 8]]}]
        response = self.client.post("/api/calibration/save", json={"zones": invalid, "snapshot_id": "test-snapshot"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.zone_path.read_bytes(), old_json)
        self.assertEqual(reference_path.read_bytes(), old_image)

    def test_lan_warning_is_in_page_data_and_api(self):
        app = create_app(self.pipeline, self.zone_path, host="0.0.0.0")
        client = app.test_client()
        self.assertTrue(client.get("/api/status").json["lan_warning"])
        self.assertIn("siapa pun", client.get("/").get_data(as_text=True))


class DistanceCalibrationAPITests(unittest.TestCase):
    def test_web_calibration_captures_depth_frame_samples_points_and_saves_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = CameraPipeline(
                {"source": "camera", "depth_model": "indoor", "depth_checkpoint": Path("metric-vits.pth"),
                 "min_depth_m": 0.1, "max_depth_m": 20.0},
                root / "zones.json", root / "distance-calibration.json",
            )
            pipeline.latest_depth_reference_frame = np.zeros((40, 60, 3), dtype=np.uint8)
            pipeline.latest_depth_map = np.full((40, 60), 2.5, dtype=np.float32)
            pipeline.latest_depth_frame_id = 25
            pipeline.latest_depth_reference_people = []
            client = create_app(pipeline, root / "zones.json").test_client()

            captured = client.post("/api/distance-calibration/snapshot")
            self.assertEqual(captured.status_code, 200, captured.json)
            metadata = captured.json
            self.assertEqual(metadata["frame_id"], 25)
            self.assertEqual(metadata["frame_size"], {"width": 60, "height": 40})
            sid = metadata["snapshot_id"]
            stored_snapshot = pipeline.distance_calibration_snapshots[sid]
            self.assertNotIn("frame", stored_snapshot)
            self.assertNotIn("depth_map", stored_snapshot)
            self.assertIn("depth_map_zlib", stored_snapshot)
            np.testing.assert_array_equal(pipeline._depth_map_from_snapshot(stored_snapshot), pipeline.latest_depth_map)
            self.assertIn("depth_jpeg", stored_snapshot)
            self.assertEqual(client.get(f"/api/distance-calibration/snapshot.jpg?snapshot_id={sid}").status_code, 200)
            depth_preview = client.get(f"/api/distance-calibration/depth.jpg?snapshot_id={sid}")
            self.assertEqual(depth_preview.status_code, 200)
            self.assertTrue(depth_preview.data)

            point = client.post("/api/distance-calibration/sample", json={
                "snapshot_id": sid, "pixel": [30, 20], "actual_distance_m": 4.0,
            })
            self.assertEqual(point.status_code, 200, point.json)
            self.assertAlmostEqual(point.json["raw_depth_m"], 2.5)
            saved = client.post("/api/distance-calibration/save", json={"samples": [point.json]})
            self.assertEqual(saved.status_code, 200, saved.json)
            self.assertAlmostEqual(saved.json["fit"]["scale"], 1.6)
            self.assertTrue((root / "distance-calibration.json").is_file())
            self.assertAlmostEqual(pipeline.distance_calibration["fit"]["scale"], 1.6)
            page = client.get("/calibration").get_data(as_text=True)
            self.assertIn("Kalibrasi jarak Depth AI", page)
            self.assertIn('id="distanceCanvas"', page)
            self.assertIn('id="distanceCalibrationStatus"', page)
            self.assertIn('id="distanceCalibrationPath"', page)
            self.assertIn("actual_distance_m", page)
            self.assertIn('id="aiResolutionMode"', page)
            self.assertIn("Rasio aspek dipertahankan", page)
            self.assertIn(">Settings</a>", page)


class AIResolutionSettingsTests(unittest.TestCase):
    def test_scaled_dimensions_keep_camera_aspect_ratio(self):
        self.assertEqual(scaled_ai_resolution(1280, 960, 640), (640, 480))
        self.assertEqual(scaled_ai_resolution(1920, 1080, 640), (640, 360))
        self.assertEqual(scaled_ai_resolution(1280, 960, 2000), (1280, 960))
        choices = ai_resolution_choices(1280, 960)
        self.assertIn({"mode": "640", "width": 640, "height": 480, "label": "640×480"}, choices)
        self.assertEqual(choices[-1]["mode"], "native")

    def test_settings_endpoint_applies_and_persists_both_model_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_file = root / "web_settings.json"
            pipeline = CameraPipeline({"source": "mock", "yolo_size": 640, "depth_size": 518,
                                       "web_settings_file": settings_file}, root / "zones.json")
            pipeline.latest_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            client = create_app(pipeline, root / "zones.json").test_client()
            initial = client.get("/api/settings/ai-resolution").get_json()
            self.assertEqual(initial["camera_size"], {"width": 1280, "height": 720})
            applied = client.post("/api/settings/ai-resolution", json={"mode": "640"})
            self.assertEqual(applied.status_code, 200, applied.json)
            self.assertEqual(applied.json["yolo_size"]["width"], 640)
            self.assertEqual(applied.json["yolo_size"]["height"], 360)
            self.assertEqual(pipeline.settings["yolo_size"], 640)
            self.assertEqual(pipeline.settings["depth_size"], 640)
            self.assertTrue(pipeline.ai_resolution_changed.is_set())
            self.assertTrue(pipeline.depth_model_id.endswith(":input640"))
            self.assertEqual(json.loads(settings_file.read_text(encoding="utf-8"))["ai_resolution_mode"], "640")
            self.assertEqual(client.post("/api/settings/ai-resolution", json={"mode": "333"}).status_code, 400)
            self.assertIn('id="aiResolutionMode"', client.get("/calibration").get_data(as_text=True))


class FrameSynchronizationTests(unittest.TestCase):
    def test_publish_reuses_frame_arrays_without_extra_copies(self):
        pipeline = CameraPipeline({"source": "mock"}, Path("unused.json"))
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        depth = np.ones((12, 12), dtype=np.float32)
        depth_frame = np.full((12, 12, 3), 127, dtype=np.uint8)
        pipeline._publish(frame, frame.copy(), [], {"frame_id": 1, "depth_stale": False}, depth, depth_frame, 1)
        self.assertIs(pipeline.latest_frame, frame)
        self.assertIs(pipeline.latest_depth_map, depth)
        self.assertIs(pipeline.latest_depth_reference_frame, depth_frame)

    def test_published_views_and_person_data_keep_the_same_frame_id(self):
        pipeline = CameraPipeline({"source": "mock"}, Path("unused.json"))
        first = np.zeros((12, 12, 3), dtype=np.uint8)
        second = np.full((12, 12, 3), 200, dtype=np.uint8)
        pipeline._publish(first, first, [{"id": 1, "bbox": [1, 1, 8, 10], "foot_point_px": [4, 9], "depth_m": None}], {"frame_id": 41, "depth_stale": False}, None)
        first_ai, _ = pipeline.view_snapshot("ai", 41)
        pipeline._publish(second, second, [{"id": 2, "bbox": [2, 2, 9, 11], "foot_point_px": [5, 10], "depth_m": None}], {"frame_id": 42, "depth_stale": False}, None)
        latest_ai, latest_status = pipeline.view_snapshot("ai", 42)
        retained_ai, _ = pipeline.view_snapshot("ai", 41)
        self.assertNotEqual(first_ai, latest_ai)
        self.assertEqual(first_ai, retained_ai)
        self.assertEqual(latest_status["frame_id"], 42)
        self.assertEqual(latest_status["people"][0]["id"], 2)

    def test_frame_buffer_is_bounded(self):
        pipeline = CameraPipeline({"source": "mock"}, Path("unused.json"))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        for frame_id in range(1, 12):
            pipeline._publish(frame, frame, [], {"frame_id": frame_id, "depth_stale": False}, None)
        self.assertLessEqual(len(pipeline.recent_frame_views), 8)
        self.assertIsNone(pipeline.view_snapshot("ai", 1)[0])
        self.assertIsNotNone(pipeline.view_snapshot("ai", 11)[0])


class VideoSourceRetentionTests(unittest.TestCase):
    def test_close_releases_cached_latest_frame(self):
        source = VideoSource.__new__(VideoSource)
        source._stop_reader = threading.Event()
        source.capture = None
        source._frame_condition = threading.Condition()
        source._latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        source._latest_observed = 1.0
        source._reader_thread = None
        source.close()
        self.assertIsNone(source._latest_frame)
        self.assertIsNone(source._latest_observed)

    def test_reader_does_not_retain_frame_returned_after_close(self):
        source = VideoSource.__new__(VideoSource)
        source._stop_reader = threading.Event()
        source._frame_condition = threading.Condition()
        source._latest_frame = None
        source._latest_observed = None
        source._frame_sequence = 0
        class ClosingCapture:
            def read(self):
                source._stop_reader.set()
                return True, np.zeros((8, 8, 3), dtype=np.uint8)
        source.capture = ClosingCapture()
        source._read_live_frames()
        self.assertIsNone(source._latest_frame)
        self.assertEqual(source._frame_sequence, 0)


class ShutdownTests(unittest.TestCase):
    def test_start_rejects_a_second_worker_for_the_same_pipeline(self):
        pipeline = CameraPipeline({"source": "fake"}, Path("unused.json"))
        release = threading.Event()
        pipeline._run = lambda: release.wait(2)
        pipeline.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                pipeline.start()
        finally:
            release.set()
            pipeline.thread.join(timeout=1)
        self.assertFalse(pipeline.thread.is_alive())

    def test_shutdown_stops_worker_and_closes_source_without_camera(self):
        released = threading.Event()

        class BlockingSource:
            def close(self):
                released.set()

        pipeline = CameraPipeline({"source": "fake"}, Path("unused.json"))
        pipeline.source = BlockingSource()
        pipeline._source_closed = False
        pipeline.thread = threading.Thread(target=lambda: released.wait(2), daemon=True)
        pipeline.thread.start()
        self.assertTrue(pipeline.shutdown(timeout=1))
        self.assertTrue(released.is_set())
        self.assertFalse(pipeline.thread.is_alive())
        self.assertEqual(pipeline.status["shutdown_status"], "stopped")

    def test_blocked_worker_is_reported_and_not_force_stopped(self):
        pipeline = CameraPipeline({"source": "fake"}, Path("unused.json"))
        wait = threading.Event()
        pipeline.thread = threading.Thread(target=wait.wait, daemon=True)
        pipeline.thread.start()
        self.assertFalse(pipeline.shutdown(timeout=0.01))
        self.assertTrue(pipeline.thread.is_alive())
        self.assertEqual(pipeline.status["shutdown_status"], "worker_still_running")
        wait.set()
        pipeline.thread.join(timeout=1)
        self.assertFalse(pipeline.thread.is_alive())


class MainLifecycleTests(unittest.TestCase):
    def run_main_with_server(self, run_side_effect=None, *, shutdown_result=True, shutdown_side_effect=None):
        pipeline = mock.Mock()
        pipeline.shutdown.return_value = shutdown_result
        if shutdown_side_effect is not None:
            pipeline.shutdown.side_effect = shutdown_side_effect
        server = mock.Mock()
        if run_side_effect is not None:
            server.run.side_effect = run_side_effect
        output = io.StringIO()
        with mock.patch.object(web_forklift, "load_settings", return_value={"source": "mock"}), \
             mock.patch.object(web_forklift, "CameraPipeline", return_value=pipeline) as pipeline_factory, \
             mock.patch.object(web_forklift, "create_app", return_value=server) as app_factory, \
             redirect_stdout(output):
            code = web_forklift.main(["--config", "mock.json", "--host", "0.0.0.0", "--port", "9123"])
        return code, pipeline, server, pipeline_factory, app_factory, output.getvalue()

    def test_main_runs_flask_once_with_requested_host_port_and_no_reloader(self):
        code, pipeline, server, pipeline_factory, app_factory, _ = self.run_main_with_server()
        self.assertEqual(code, 0)
        pipeline_factory.assert_called_once()
        pipeline.start.assert_called_once()
        app_factory.assert_called_once()
        server.run.assert_called_once_with(host="0.0.0.0", port=9123, debug=False, use_reloader=False, threaded=True)
        pipeline.shutdown.assert_called_once()

    def test_shutdown_failure_after_normal_server_exit_is_nonzero(self):
        code, pipeline, server, _, _, output = self.run_main_with_server(shutdown_result=False)
        self.assertEqual(code, 3)
        self.assertIn("shutdown was not clean", output.lower())
        server.run.assert_called_once()
        pipeline.shutdown.assert_called_once()

    def test_server_and_shutdown_failures_are_both_reported(self):
        code, pipeline, server, _, _, output = self.run_main_with_server(RuntimeError("bind failed"), shutdown_result=False)
        self.assertEqual(code, 2)
        self.assertIn("bind failed", output)
        self.assertIn("shutdown was not clean", output.lower())
        server.run.assert_called_once()
        pipeline.shutdown.assert_called_once()

    def test_ctrl_c_calls_shutdown_and_reports_shutdown_failure(self):
        code, pipeline, server, _, _, output = self.run_main_with_server(KeyboardInterrupt(), shutdown_result=False)
        self.assertEqual(code, 3)
        self.assertIn("shutdown was not clean", output.lower())
        pipeline.shutdown.assert_called_once()

    def test_shutdown_exception_after_server_exception_preserves_both_errors(self):
        code, pipeline, _server, _, _, output = self.run_main_with_server(
            RuntimeError("listen failed"), shutdown_side_effect=RuntimeError("cleanup failed"))
        self.assertEqual(code, 2)
        self.assertIn("listen failed", output)
        self.assertIn("cleanup failed", output)
        pipeline.shutdown.assert_called_once()

    def test_ctrl_c_with_clean_shutdown_returns_success(self):
        code, pipeline, _server, _, _, _ = self.run_main_with_server(KeyboardInterrupt(), shutdown_result=True)
        self.assertEqual(code, 0)
        pipeline.shutdown.assert_called_once()


class SnapshotRetentionTests(unittest.TestCase):
    def make_pipeline(self, *, maximum=8, ttl=900):
        pipeline = CameraPipeline({"source": "mock", "calibration_snapshot_max_count": maximum,
                                   "calibration_snapshot_ttl_s": ttl}, Path("unused.json"))
        pipeline.latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        pipeline.updated_at = time.time()
        return pipeline

    def test_expired_snapshots_are_pruned_during_live_publishing(self):
        pipeline = self.make_pipeline(maximum=2, ttl=1)
        with mock.patch.object(web_forklift.time, "monotonic", side_effect=[100.0, 102.0]):
            snapshot = pipeline.capture_calibration_snapshot()
            pipeline._publish(pipeline.latest_frame, pipeline.latest_frame.copy(), [], {"frame_id": 1, "depth_stale": False})
        self.assertNotIn(snapshot["snapshot_id"], pipeline.calibration_snapshots)

    def test_calibration_snapshot_retains_jpeg_without_full_resolution_frame_copy(self):
        pipeline = self.make_pipeline()
        metadata = pipeline.capture_calibration_snapshot()
        stored = pipeline.calibration_snapshots[metadata["snapshot_id"]]
        self.assertNotIn("frame", stored)
        self.assertIn("jpeg", stored)
        self.assertEqual(metadata["frame_size"], {"width": 8, "height": 8})

    def test_expired_snapshot_is_pruned_when_taking_next_snapshot(self):
        pipeline = self.make_pipeline(maximum=1, ttl=10)
        with mock.patch.object(web_forklift.time, "monotonic", side_effect=[100.0, 111.0]):
            old = pipeline.capture_calibration_snapshot()
            current = pipeline.capture_calibration_snapshot()
        self.assertNotIn(old["snapshot_id"], pipeline.calibration_snapshots)
        self.assertIn(current["snapshot_id"], pipeline.calibration_snapshots)
        self.assertEqual(len(pipeline.calibration_snapshots), 1)

    def test_reading_snapshot_does_not_extend_ttl(self):
        pipeline = self.make_pipeline(maximum=3, ttl=10)
        with mock.patch.object(web_forklift.time, "monotonic", side_effect=[100.0, 109.0, 111.0]):
            first = pipeline.capture_calibration_snapshot()
            jpeg, _ = pipeline.get_calibration_snapshot(first["snapshot_id"])
            pipeline.latest_frame[:] = 255
            second = pipeline.capture_calibration_snapshot()
        self.assertIsNotNone(jpeg)
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertNotIn(first["snapshot_id"], pipeline.calibration_snapshots)

    def test_snapshot_limit_rejects_new_frame_without_removing_active_snapshots(self):
        pipeline = self.make_pipeline(maximum=2)
        with mock.patch.object(web_forklift.time, "monotonic", side_effect=[1.0, 2.0, 3.0, 3.0]):
            first = pipeline.capture_calibration_snapshot()
            second = pipeline.capture_calibration_snapshot()
            with self.assertRaises(SnapshotCapacityError):
                pipeline.capture_calibration_snapshot()
            self.assertEqual(len(pipeline.calibration_snapshots), 2)
            pipeline.cancel_calibration_snapshot(first["snapshot_id"])
        self.assertNotIn(first["snapshot_id"], pipeline.calibration_snapshots)
        self.assertIn(second["snapshot_id"], pipeline.calibration_snapshots)
        self.assertEqual(len(pipeline.calibration_snapshots), 1)

    def test_snapshot_limit_returns_clear_http_error_and_cancel_frees_slot(self):
        pipeline = self.make_pipeline(maximum=1)
        client = create_app(pipeline, Path("unused.json")).test_client()
        first = client.post("/api/calibration/snapshot")
        self.assertEqual(first.status_code, 200)
        second = client.post("/api/calibration/snapshot")
        self.assertEqual(second.status_code, 429)
        self.assertIn("limit reached", second.json["error"].lower())
        sid = first.json["snapshot_id"]
        self.assertEqual(client.delete(f"/api/calibration/snapshot?snapshot_id={sid}").status_code, 200)
        self.assertEqual(len(pipeline.calibration_snapshots), 0)
        self.assertEqual(client.post("/api/calibration/snapshot").status_code, 200)

    def test_expired_snapshot_is_rejected_by_metadata_image_load_and_save(self):
        operations = ("metadata", "image", "load", "save")
        for operation in operations:
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                pipeline = self.make_pipeline(maximum=1, ttl=10)
                zone_path = Path(directory) / "zones.json"
                client = create_app(pipeline, zone_path).test_client()
                with mock.patch.object(web_forklift.time, "monotonic", side_effect=[100.0, 111.0, 112.0]):
                    snapshot = pipeline.capture_calibration_snapshot()
                    sid = snapshot["snapshot_id"]
                    if operation == "metadata":
                        response = client.get(f"/api/calibration/snapshot?snapshot_id={sid}")
                    elif operation == "image":
                        response = client.get(f"/api/calibration/snapshot.jpg?snapshot_id={sid}")
                    elif operation == "load":
                        response = client.get(f"/api/calibration/load?snapshot_id={sid}")
                    else:
                        response = client.post("/api/calibration/save", json={
                            "snapshot_id": sid,
                            "zones": [{"name": "hazard", "color": "#FF0000", "points": [[1, 1], [6, 1], [6, 6]]}],
                        })
                    replacement = pipeline.capture_calibration_snapshot()
                self.assertEqual(response.status_code, 410, response.get_json())
                self.assertIn("expired", response.get_json()["error"].lower())
                self.assertNotIn(sid, pipeline.calibration_snapshots)
                self.assertIn(replacement["snapshot_id"], pipeline.calibration_snapshots)
                self.assertFalse(zone_path.exists())
                self.assertEqual(len(pipeline.calibration_snapshots), 1)

    def test_live_snapshot_remains_usable_until_cancelled(self):
        pipeline = self.make_pipeline(maximum=2, ttl=10)
        client = create_app(pipeline, Path("unused.json")).test_client()
        with mock.patch.object(web_forklift.time, "monotonic", side_effect=[100.0, 101.0, 102.0]):
            snapshot = pipeline.capture_calibration_snapshot()
            sid = snapshot["snapshot_id"]
            metadata = client.get(f"/api/calibration/snapshot?snapshot_id={sid}")
            image = client.get(f"/api/calibration/snapshot.jpg?snapshot_id={sid}")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(client.delete(f"/api/calibration/snapshot?snapshot_id={sid}").status_code, 200)
        self.assertNotIn(sid, pipeline.calibration_snapshots)


if __name__ == "__main__":
    unittest.main()
