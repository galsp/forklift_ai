import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from forklift_vision.distance_calibration import (
    create_distance_calibration,
    fit_depth_scale,
    load_distance_calibration,
    sample_depth_at_point,
)


class DistanceCalibrationTests(unittest.TestCase):
    def test_one_point_calibration_produces_scale_and_error(self):
        result = fit_depth_scale([{"raw_depth_m": 2.0, "actual_distance_m": 3.0}])
        self.assertEqual(result["scale"], 1.5)
        self.assertEqual(result["sample_count"], 1)
        self.assertAlmostEqual(result["mae_m"], 0.0)

    def test_multiple_points_use_median_scale_ratio(self):
        result = fit_depth_scale([
            {"raw_depth_m": 2.0, "actual_distance_m": 3.0},
            {"raw_depth_m": 4.0, "actual_distance_m": 6.0},
            {"raw_depth_m": 8.0, "actual_distance_m": 80.0},
        ])
        self.assertEqual(result["scale"], 1.5)
        self.assertEqual(result["sample_count"], 3)

    def test_sample_depth_uses_valid_patch_and_rejects_invalid_point(self):
        depth = np.full((30, 30), 3.25, dtype=np.float32)
        depth[10, 10] = np.nan
        sample = sample_depth_at_point(depth, (10, 10), radius=2)
        self.assertIsNotNone(sample)
        self.assertAlmostEqual(sample["raw_depth_m"], 3.25, places=5)
        self.assertIsNone(sample_depth_at_point(depth, (-1, 10)))
        self.assertIsNone(sample_depth_at_point(np.full((30, 30), np.nan), (10, 10)))

    def test_legacy_checkpoint_calibration_is_assumed_to_use_default_518_input_only(self):
        legacy_model = "indoor:depth_anything_v2_metric_hypersim_vits.pth"
        payload = create_distance_calibration("camera", (100, 80), legacy_model, [
            {"pixel": [20, 30], "raw_depth_m": 2.0, "actual_distance_m": 2.5, "frame_id": 15}
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distance.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_distance_calibration(path, (100, 80), f"{legacy_model}:input518")
            self.assertEqual(loaded["fit"]["scale"], 1.25)
            self.assertEqual(loaded["legacy_input_size_assumed"], 518)
            with self.assertRaises(ValueError):
                load_distance_calibration(path, (100, 80), f"{legacy_model}:input640")

    def test_calibration_round_trip_and_resolution_model_checks(self):
        payload = create_distance_calibration("camera", (100, 80), "hypersim-vits:checkpoint", [
            {"pixel": [20, 30], "raw_depth_m": 2.0, "actual_distance_m": 2.5, "frame_id": 15}
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "distance.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_distance_calibration(path, (100, 80), "hypersim-vits:checkpoint")
            self.assertEqual(loaded["fit"]["scale"], 1.25)
            with self.assertRaises(ValueError):
                load_distance_calibration(path, (200, 80), "hypersim-vits:checkpoint")
            with self.assertRaises(ValueError):
                load_distance_calibration(path, (100, 80), "other-checkpoint")


if __name__ == "__main__":
    unittest.main()
