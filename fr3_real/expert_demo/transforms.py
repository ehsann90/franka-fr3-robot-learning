"""Rigid-transform utilities for FR3 expert demonstrations.

Convention
----------
``T_A_B`` is the pose of frame B expressed in frame A. It transforms a
homogeneous coordinate from frame B into frame A::

    p_A = T_A_B @ p_B

Transforms are authoritative 4x4 matrices. Quaternions or Euler angles may be
stored as descriptive metadata, but are not used as the calibration source of
truth.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


Transform = npt.NDArray[np.float64]
# Franky's measured transforms are exposed with approximately float32 rotation
# precision. Across 23,800 samples from the existing physical FR3 recordings,
# the maximum orthonormality residual was 9.74e-8 and determinant error was
# 1.22e-7. A 1e-6 tolerance accepts that measurement noise while still
# rejecting malformed/scaled transforms by orders of magnitude.
DEFAULT_RIGID_TRANSFORM_ATOL = 1e-6


class TransformError(ValueError):
    """Raised when a value is not a finite rigid homogeneous transform."""


def validate_transform(
    transform: npt.ArrayLike,
    *,
    name: str = "transform",
    atol: float = DEFAULT_RIGID_TRANSFORM_ATOL,
) -> Transform:
    """Return a defensive float64 copy after validating a rigid transform."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise TransformError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise TransformError(f"{name} contains NaN or infinite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise TransformError(f"{name} has an invalid homogeneous last row")

    rotation = matrix[:3, :3]
    orthonormality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if orthonormality_error > atol:
        raise TransformError(
            f"{name} rotation is not orthonormal "
            f"(max residual {orthonormality_error:.3e}, tolerance {atol:.3e})"
        )
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=atol, rel_tol=0.0):
        raise TransformError(
            f"{name} rotation determinant must be +1, got {determinant:.12g} "
            f"(tolerance {atol:.3e})"
        )
    return matrix.copy()


def make_transform(
    translation: Sequence[float] | npt.ArrayLike = (0.0, 0.0, 0.0),
    rotation: npt.ArrayLike | None = None,
) -> Transform:
    """Construct and validate a transform from XYZ and a 3x3 rotation."""

    xyz = np.asarray(translation, dtype=np.float64)
    if xyz.shape != (3,):
        raise TransformError(f"translation must have shape (3,), got {xyz.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = xyz
    if rotation is not None:
        rotation_matrix = np.asarray(rotation, dtype=np.float64)
        if rotation_matrix.shape != (3, 3):
            raise TransformError(f"rotation must have shape (3, 3), got {rotation_matrix.shape}")
        matrix[:3, :3] = rotation_matrix
    return validate_transform(matrix)


def rotation_matrix_from_axis_angle(axis: npt.ArrayLike, angle_radians: float) -> npt.NDArray[np.float64]:
    """Return a 3x3 right-handed rotation matrix using Rodrigues' formula."""

    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise TransformError("axis must be a finite vector with shape (3,)")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise TransformError("axis must have non-zero length")
    x, y, z = vector / norm
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    rotation = identity + math.sin(angle_radians) * skew + (1.0 - math.cos(angle_radians)) * (skew @ skew)
    # Validate through a complete transform so all rotation checks stay in one place.
    return make_transform(rotation=rotation)[:3, :3]


def quaternion_xyzw_from_rotation(rotation: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""

    matrix = make_transform(rotation=rotation)[:3, :3]
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def inverse_transform(transform: npt.ArrayLike) -> Transform:
    """Invert a rigid transform without using a generic matrix inverse."""

    matrix = validate_transform(transform)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def transform_between(T_base_a: npt.ArrayLike, T_base_b: npt.ArrayLike) -> Transform:
    """Calculate ``T_A_B`` from poses ``T_base_A`` and ``T_base_B``."""

    T_base_a_valid = validate_transform(T_base_a, name="T_base_a")
    T_base_b_valid = validate_transform(T_base_b, name="T_base_b")
    return validate_transform(inverse_transform(T_base_a_valid) @ T_base_b_valid, name="T_a_b")


def required_ee_pose(T_base_object_target: npt.ArrayLike, T_ee_object: npt.ArrayLike) -> Transform:
    """Calculate the EE pose that realizes a desired object pose.

    Given the grasp relationship ``T_ee_object`` and fixed target
    ``T_base_object_target``::

        T_base_ee_target = T_base_object_target @ inverse(T_ee_object)
    """

    target = validate_transform(T_base_object_target, name="T_base_object_target")
    grasp = validate_transform(T_ee_object, name="T_ee_object")
    return validate_transform(target @ inverse_transform(grasp), name="T_base_ee_target")


def object_pose_from_ee_pose(T_base_ee: npt.ArrayLike, T_ee_object: npt.ArrayLike) -> Transform:
    """Reconstruct ``T_base_object`` from an EE pose and grasp relationship."""

    ee_pose = validate_transform(T_base_ee, name="T_base_ee")
    grasp = validate_transform(T_ee_object, name="T_ee_object")
    return validate_transform(ee_pose @ grasp, name="T_base_object")
