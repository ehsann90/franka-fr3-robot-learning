import unittest

import numpy as np

from expert_demo.bolt_free_spin import BoltFreeSpinGuard


class BoltFreeSpinGuardTests(unittest.TestCase):
    def guard(self):
        return BoltFreeSpinGuard(
            model_q7=np.array([0.0, 1.0]),
            model_K_Tz=np.array([0.2, 0.3]),
            baseline_Fz_N=-2.0,
            hard_negative_residual_Tz_Nm=-0.5,
            hard_positive_axial_delta_N=2.0,
            q7_min_rad=-0.1,
            q7_max_rad=1.1,
            start_xyz_m=np.zeros(3),
            maximum_tcp_translation_m=0.001,
        )

    def test_nominal_update_does_not_stop(self):
        guard = self.guard()
        residual, axial = guard.update(0.5, [0, 0, -1.5, 0, 0, 0.1], [0, 0, 0])
        self.assertAlmostEqual(residual, -0.15)
        self.assertAlmostEqual(axial, 0.5)
        self.assertIsNone(guard.stop_reason)

    def test_residual_torque_cutoff(self):
        guard = self.guard()
        guard.update(0.5, [0, 0, -2, 0, 0, -0.3], [0, 0, 0])
        self.assertEqual(guard.stop_reason, "hard_residual_torque_cutoff")

    def test_axial_cutoff(self):
        guard = self.guard()
        guard.update(0.5, [0, 0, 0.1, 0, 0, 0.25], [0, 0, 0])
        self.assertEqual(guard.stop_reason, "hard_axial_force_cutoff")

    def test_translation_cutoff(self):
        guard = self.guard()
        guard.update(0.5, [0, 0, -2, 0, 0, 0.25], [0.002, 0, 0])
        self.assertEqual(guard.stop_reason, "tcp_translation_limit")


if __name__ == "__main__":
    unittest.main()
