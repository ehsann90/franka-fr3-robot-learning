import math
import unittest

from expert_demo.thread_drive_preview import half_cosine_duration, preview_blockers


class ThreadDrivePreviewTests(unittest.TestCase):
    def test_duration_has_requested_peak_speed(self):
        duration = half_cosine_duration(math.pi / 2.0, 0.09)
        self.assertAlmostEqual(duration, math.pi * math.pi / (4.0 * 0.09))

    def test_valid_commissioned_window(self):
        blockers = preview_blockers(
            live_q7_rad=0.043,
            start_q7_rad=0.043015,
            start_tolerance_rad=0.03,
            target_q7_rad=0.043 + math.pi / 2.0,
            cable_interval_rad=(-0.1569, 2.9989),
            dynamic_model_interval_rad=(0.02, 1.7),
        )
        self.assertEqual(blockers, [])

    def test_rejects_unmodelled_target(self):
        blockers = preview_blockers(
            live_q7_rad=0.043,
            start_q7_rad=0.043015,
            start_tolerance_rad=0.03,
            target_q7_rad=2.0,
            cable_interval_rad=(-0.1569, 2.9989),
            dynamic_model_interval_rad=(0.02, 1.7),
        )
        self.assertIn("planned q7 interval leaves the calibrated dynamic torque model", blockers)


if __name__ == "__main__":
    unittest.main()
