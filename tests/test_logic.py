import unittest

import numpy as np

from forklift_vision.logic import (
    DepthFilter, MotionHistory, WarningMachine, data_is_stale,
    estimate_foot_depth, estimate_roi_depth, homography_from_points,
    person_foot_point, project_ground_point, valid_roi,
)


class LogicTests(unittest.TestCase):
    def test_roi_clips_to_frame_and_rejects_invalid(self):
        self.assertIsNotNone(valid_roi([-20, 0, 40, 100], 100, 100))
        self.assertIsNone(valid_roi([10, 10, 10, 20], 100, 100))
        self.assertIsNone(valid_roi([0, 0, float("nan"), 2], 100, 100))

    def test_depth_robust_to_outlier_and_invalid_values(self):
        depth = np.full((100, 100), 3.0, dtype=np.float32)
        depth[45, 45] = 18.0
        depth[46, 46] = np.nan
        measurement = estimate_roi_depth(depth, [10, 5, 90, 95], min_pixels=10)
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement[0], 3.0)
        self.assertIsNone(estimate_roi_depth(np.full((100, 100), np.nan), [10, 5, 90, 95]))

    def test_foot_depth_uses_lower_center_region_and_reports_reference_point(self):
        depth = np.full((100, 100), 2.5, dtype=np.float32)
        depth[76:90, 41:59] = 4.2
        result = estimate_foot_depth(depth, [20, 10, 80, 90], min_pixels=12)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 4.2, places=5)
        self.assertEqual(person_foot_point([20, 10, 80, 90], 100, 100), (50, 89))

    def test_foot_point_rejects_invalid_or_out_of_frame_bbox(self):
        self.assertIsNone(person_foot_point([2, 2, 2, 10], 100, 100))
        self.assertIsNone(person_foot_point([-40, 2, -10, 20], 100, 100))
        self.assertIsNone(person_foot_point([0, 0, 10, 20], 0, 100))

    def test_depth_median_filter(self):
        filt = DepthFilter(3)
        self.assertEqual(filt.update(3), 3)
        filt.update(30)
        self.assertEqual(filt.update(3.2), 3.2)

    def test_closing_receding_stable_and_invalid_timestamp(self):
        history = MotionHistory(min_samples=3, speed_threshold_mps=0.3)
        history.update(1, 1.0, 5.0)
        history.update(1, 2.0, 4.5)
        closing = history.update(1, 3.0, 4.0)
        self.assertEqual(closing.status, "CLOSING")
        self.assertAlmostEqual(closing.closing_speed_mps, 0.5)
        self.assertAlmostEqual(closing.experimental_ttc_s, 8.0)
        self.assertEqual(history.update(1, 3.0, 3.8).status, "UNKNOWN")
        history.clear()
        history.update(2, 1, 4)
        history.update(2, 2, 4.1)
        self.assertEqual(history.update(2, 3, 4.0).status, "STABLE")
        history.clear()
        history.update(3, 1, 2)
        history.update(3, 2, 3)
        self.assertEqual(history.update(3, 3, 4).status, "RECEDING")

    def test_history_cleanup(self):
        history = MotionHistory(timeout_s=2, min_samples=2)
        history.update(1, 1, 4)
        history.cleanup(4)
        self.assertNotIn(1, history.samples)

    def test_warning_hysteresis_debounce_and_unknown(self):
        machine = WarningMachine(debounce_s=0.2)
        self.assertEqual(machine.update(0, 5, valid=True), "UNKNOWN")
        self.assertEqual(machine.update(0.1, 5, valid=True), "UNKNOWN")
        self.assertEqual(machine.update(0.25, 5, valid=True), "NORMAL")
        machine.update(1, 3.8, valid=True)
        self.assertEqual(machine.update(1.3, 3.8, valid=True), "WARNING")
        self.assertEqual(machine.update(1.4, 4.2, valid=True), "WARNING")
        self.assertEqual(machine.update(1.5, None, valid=False), "UNKNOWN")

    def test_stale_depth(self):
        self.assertFalse(data_is_stale(10, 10.5, 1))
        self.assertTrue(data_is_stale(10, 11.1, 1))
        self.assertTrue(data_is_stale(None, 10, 1))
        self.assertTrue(data_is_stale(11, 10, 1))

    def test_synthetic_homography(self):
        image = [(0, 0), (100, 0), (100, 100), (0, 100)]
        ground = [(0, 0), (2, 0), (2, 2), (0, 2)]
        matrix = homography_from_points(image, ground)
        result = project_ground_point(matrix, (50, 50))
        self.assertAlmostEqual(result[0], 1.0, places=4)
        self.assertAlmostEqual(result[1], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
