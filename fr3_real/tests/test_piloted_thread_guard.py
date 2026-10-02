import unittest

import numpy as np

from expert_demo.piloted_thread_guard import PilotedThreadReplayGuard


class PilotedThreadGuardTests(unittest.TestCase):
    def make_guard(self):
        path = np.zeros((3, 7)); path[:, 0] = [0.0, 0.1, 0.2]; path[:, 6] = [0.0, 1.0, 2.0]
        return PilotedThreadReplayGuard(
            path_q=path,
            model_q7=np.array([0.0, 2.0]),
            model_K_Tz=np.array([0.1, 0.3]),
            baseline_Fz_N=-1.0,
            hard_absolute_axial_delta_N=2.0,
            hard_negative_residual_Tz_Nm=-0.5,
            q7_min_rad=-0.1,
            q7_max_rad=2.1,
            maximum_q1_q6_path_error_rad=0.05,
        )

    def test_nominal_interpolation(self):
        guard = self.make_guard()
        residual, axial, error = guard.update([0.1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0.1])
        self.assertAlmostEqual(residual, -0.1)
        self.assertAlmostEqual(axial, 1.0)
        self.assertAlmostEqual(error, 0.0)
        self.assertIsNone(guard.stop_reason)

    def test_force_stop_is_symmetric(self):
        guard = self.make_guard()
        guard.update([0, 0, 0, 0, 0, 0, 0], [0, 0, -3.1, 0, 0, 0.1])
        self.assertEqual(guard.stop_reason, "hard_absolute_axial_force_cutoff")

    def test_path_tracking_stop(self):
        guard = self.make_guard()
        guard.update([0.2, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0.2])
        self.assertEqual(guard.stop_reason, "q1_q6_path_tracking_limit")


if __name__ == "__main__":
    unittest.main()
