import unittest

import numpy as np

from expert_demo.bolt_axial_preload import (
    AxialPreloadGuard,
    decompose_translation_error,
    rotation_angle_rad,
    tool_axis_tilt_rad,
)


class AxialPreloadGuardTests(unittest.TestCase):
    def guard(self):
        return AxialPreloadGuard(
            baseline_Fz_N=-1.5,
            axis_base=np.array([0.0, 0.0, -1.0]),
            start_xyz_m=np.zeros(3),
            hard_absolute_axial_delta_N=2.0,
            wrong_direction_threshold_N=0.5,
            wrong_direction_persistence_s=0.1,
            maximum_axial_travel_m=0.003,
            maximum_lateral_translation_m=0.0003,
            maximum_duration_s=8.0,
        )

    def test_nominal_update_does_not_stop(self):
        guard = self.guard()
        value = guard.update(1.0, [0, 0, -0.3, 0, 0, 0], [0, 0, -0.0002])
        self.assertAlmostEqual(value, 1.2)
        self.assertIsNone(guard.stop_reason)

    def test_positive_force_cutoff(self):
        guard = self.guard()
        guard.update(1.0, [0, 0, 0.6, 0, 0, 0], [0, 0, 0])
        self.assertEqual(guard.stop_reason, "hard_absolute_axial_force_cutoff")

    def test_wrong_sign_force_cutoff(self):
        guard = self.guard()
        guard.update(1.0, [0, 0, -3.6, 0, 0, 0], [0, 0, 0])
        self.assertEqual(guard.stop_reason, "hard_absolute_axial_force_cutoff")

    def test_persistent_wrong_direction_force(self):
        guard = self.guard()
        guard.update(1.0, [0, 0, -2.1, 0, 0, 0], [0, 0, 0])
        self.assertIsNone(guard.stop_reason)
        guard.update(1.11, [0, 0, -2.1, 0, 0, 0], [0, 0, 0])
        self.assertEqual(guard.stop_reason, "persistent_wrong_direction_force")

    def test_axial_travel_cutoff(self):
        guard = self.guard()
        guard.update(1.0, [0, 0, -1.5, 0, 0, 0], [0, 0, -0.0031])
        self.assertEqual(guard.stop_reason, "maximum_axial_travel")

    def test_lateral_travel_cutoff(self):
        guard = self.guard()
        guard.update(1.0, [0, 0, -1.5, 0, 0, 0], [0.00031, 0, 0])
        self.assertEqual(guard.stop_reason, "maximum_lateral_translation")

    def test_timeout(self):
        guard = self.guard()
        guard.update(8.0, [0, 0, -1.5, 0, 0, 0], [0, 0, 0])
        self.assertEqual(guard.stop_reason, "timeout")

    def test_tool_axis_tilt_ignores_yaw(self):
        yaw = np.deg2rad(76.25)
        rotation_about_z = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.assertAlmostEqual(tool_axis_tilt_rad(np.eye(3), rotation_about_z), 0.0)
        self.assertAlmostEqual(rotation_angle_rad(np.eye(3), rotation_about_z), yaw)

    def test_translation_error_is_split_by_axis(self):
        axial, lateral = decompose_translation_error(
            np.array([0.003, 0.004, -0.002]),
            np.zeros(3),
            np.array([0.0, 0.0, -1.0]),
        )
        self.assertAlmostEqual(axial, 0.002)
        self.assertAlmostEqual(lateral, 0.005)


if __name__ == "__main__":
    unittest.main()
