"""Synthetic tests for the expert-demonstration transform convention."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.transforms import (  # noqa: E402
    TransformError,
    inverse_transform,
    make_transform,
    object_pose_from_ee_pose,
    quaternion_xyzw_from_rotation,
    required_ee_pose,
    rotation_matrix_from_axis_angle,
    transform_between,
    validate_transform,
)


class TransformTests(unittest.TestCase):
    def assertTransformClose(self, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-9) -> None:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=0.0)

    def test_inverse_round_trip_with_rotation_and_translation(self) -> None:
        rotation = rotation_matrix_from_axis_angle([0.2, -0.4, 0.8], math.radians(37.0))
        transform = make_transform([0.31, -0.22, 0.74], rotation)
        self.assertTransformClose(transform @ inverse_transform(transform), np.eye(4))
        self.assertTransformClose(inverse_transform(inverse_transform(transform)), transform)

    def test_rotation_to_xyzw_quaternion(self) -> None:
        rotation = rotation_matrix_from_axis_angle([1.0, 0.0, 0.0], math.pi)
        quaternion = quaternion_xyzw_from_rotation(rotation)
        np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-9, rtol=0.0)

    def test_transform_between_reconstructs_child_pose(self) -> None:
        T_base_ee = make_transform(
            [0.45, -0.10, 0.32],
            rotation_matrix_from_axis_angle([0.0, 1.0, 0.0], math.radians(15.0)),
        )
        T_base_bracket = make_transform(
            [0.49, -0.08, 0.27],
            rotation_matrix_from_axis_angle([0.0, 0.0, 1.0], math.radians(-8.0)),
        )
        T_ee_bracket = transform_between(T_base_ee, T_base_bracket)
        self.assertTransformClose(T_base_ee @ T_ee_bracket, T_base_bracket)

    def test_lower_bracket_grasp_changes_ee_pose_but_not_wall_bracket_pose(self) -> None:
        T_base_bracket_wall = make_transform([0.62, 0.18, 0.41])

        # In the LOW grasp, EE is 30 mm below the bracket frame. Therefore the
        # bracket is +30 mm along EE Z, expressed by T_ee_bracket_low.
        T_ee_bracket_center = make_transform()
        T_ee_bracket_low = make_transform([0.0, 0.0, 0.030])

        T_base_ee_center = required_ee_pose(T_base_bracket_wall, T_ee_bracket_center)
        T_base_ee_low = required_ee_pose(T_base_bracket_wall, T_ee_bracket_low)

        self.assertAlmostEqual(T_base_ee_low[2, 3], T_base_ee_center[2, 3] - 0.030)
        self.assertTransformClose(
            object_pose_from_ee_pose(T_base_ee_center, T_ee_bracket_center),
            T_base_bracket_wall,
        )
        self.assertTransformClose(
            object_pose_from_ee_pose(T_base_ee_low, T_ee_bracket_low),
            T_base_bracket_wall,
        )

    def test_grasp_orientation_is_compensated_at_wall(self) -> None:
        T_base_bracket_wall = make_transform(
            [0.60, 0.20, 0.40],
            rotation_matrix_from_axis_angle([1.0, 0.0, 0.0], math.radians(90.0)),
        )
        T_ee_bracket = make_transform(
            [0.01, -0.025, 0.04],
            rotation_matrix_from_axis_angle([0.0, 0.0, 1.0], math.radians(6.0)),
        )
        T_base_ee_wall = required_ee_pose(T_base_bracket_wall, T_ee_bracket)
        self.assertTransformClose(T_base_ee_wall @ T_ee_bracket, T_base_bracket_wall)

    def test_screw_preinsert_reconstruction(self) -> None:
        T_base_screw_preinsert = make_transform(
            [0.55, -0.14, 0.36],
            rotation_matrix_from_axis_angle([0.0, 1.0, 0.0], math.radians(90.0)),
        )
        T_ee_screw = make_transform([0.0, 0.0, 0.018])
        T_base_ee_preinsert = required_ee_pose(T_base_screw_preinsert, T_ee_screw)
        self.assertTransformClose(T_base_ee_preinsert @ T_ee_screw, T_base_screw_preinsert)

    def test_validation_rejects_non_rigid_and_non_finite_matrices(self) -> None:
        scaled = np.eye(4)
        scaled[0, 0] = 2.0
        with self.assertRaises(TransformError):
            validate_transform(scaled)

        reflected = np.eye(4)
        reflected[0, 0] = -1.0
        with self.assertRaises(TransformError):
            validate_transform(reflected)

        non_finite = np.eye(4)
        non_finite[1, 3] = np.nan
        with self.assertRaises(TransformError):
            validate_transform(non_finite)

    def test_validation_accepts_measured_franky_rotation_precision(self) -> None:
        measured = make_transform(
            [0.5, -0.2, 0.3],
            rotation_matrix_from_axis_angle([0.1, 0.7, -0.2], 0.4),
        )
        measured[0, 0] += 5e-8
        validated = validate_transform(measured, name="measured_franky_pose")
        np.testing.assert_array_equal(validated, measured)

    def test_inputs_are_not_aliased(self) -> None:
        source = np.eye(4)
        validated = validate_transform(source)
        validated[0, 3] = 1.0
        self.assertEqual(source[0, 3], 0.0)


if __name__ == "__main__":
    unittest.main()
