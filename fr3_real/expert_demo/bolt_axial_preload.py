"""Control-rate guards for stationary bolt axial-preload commissioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def rotation_angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    """Return the full relative rotation angle between two 3x3 matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("rotations must both be 3x3")
    cosine = float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def tool_axis_tilt_rad(a: np.ndarray, b: np.ndarray) -> float:
    """Return tilt between +Z axes while intentionally ignoring yaw about Z."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("tool-axis rotations must both be 3x3")
    cosine = float(np.clip(np.dot(a[:, 2], b[:, 2]), -1.0, 1.0))
    return float(np.arccos(cosine))


def decompose_translation_error(
    live_xyz: np.ndarray, reference_xyz: np.ndarray, axis_base: np.ndarray
) -> tuple[float, float]:
    """Return signed axial and nonnegative lateral error magnitudes."""
    live = np.asarray(live_xyz, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_xyz, dtype=np.float64).reshape(-1)
    axis = np.asarray(axis_base, dtype=np.float64).reshape(-1)
    if live.size != 3 or reference.size != 3 or axis.size != 3:
        raise ValueError("translation inputs must all contain three values")
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("axis must be finite and nonzero")
    axis = axis / norm
    delta = live - reference
    signed_axial = float(np.dot(delta, axis))
    lateral = float(np.linalg.norm(delta - signed_axial * axis))
    return signed_axial, lateral


@dataclass
class AxialPreloadGuard:
    baseline_Fz_N: float
    axis_base: np.ndarray
    start_xyz_m: np.ndarray
    hard_absolute_axial_delta_N: float
    wrong_direction_threshold_N: float
    wrong_direction_persistence_s: float
    maximum_axial_travel_m: float
    maximum_lateral_translation_m: float
    maximum_duration_s: float
    stop_reason: str | None = None
    minimum_axial_delta_N: float = np.inf
    maximum_axial_delta_N: float = -np.inf
    maximum_absolute_axial_delta_N: float = 0.0
    maximum_axial_travel_observed_m: float = 0.0
    maximum_lateral_translation_observed_m: float = 0.0
    wrong_direction_since_s: float | None = None

    def update(
        self,
        elapsed_s: float,
        wrench_K: Sequence[float],
        xyz_m: Sequence[float],
    ) -> float:
        wrench = np.asarray(wrench_K, dtype=np.float64).reshape(-1)
        xyz = np.asarray(xyz_m, dtype=np.float64).reshape(-1)
        axis = np.asarray(self.axis_base, dtype=np.float64).reshape(-1)
        if (
            wrench.size != 6
            or xyz.size != 3
            or axis.size != 3
            or not np.isfinite(wrench).all()
            or not np.isfinite(xyz).all()
            or not np.isfinite(axis).all()
            or not np.isfinite(elapsed_s)
        ):
            self.stop_reason = self.stop_reason or "invalid_control_state"
            return np.nan

        axial_delta = float(wrench[2] - self.baseline_Fz_N)
        displacement = xyz - np.asarray(self.start_xyz_m, dtype=np.float64)
        signed_axial_travel = float(np.dot(displacement, axis))
        lateral = displacement - signed_axial_travel * axis
        axial_travel = abs(signed_axial_travel)
        lateral_translation = float(np.linalg.norm(lateral))

        self.minimum_axial_delta_N = min(self.minimum_axial_delta_N, axial_delta)
        self.maximum_axial_delta_N = max(self.maximum_axial_delta_N, axial_delta)
        self.maximum_absolute_axial_delta_N = max(
            self.maximum_absolute_axial_delta_N, abs(axial_delta)
        )
        self.maximum_axial_travel_observed_m = max(
            self.maximum_axial_travel_observed_m, axial_travel
        )
        self.maximum_lateral_translation_observed_m = max(
            self.maximum_lateral_translation_observed_m, lateral_translation
        )

        if axial_delta <= -self.wrong_direction_threshold_N:
            if self.wrong_direction_since_s is None:
                self.wrong_direction_since_s = elapsed_s
        else:
            self.wrong_direction_since_s = None

        if self.stop_reason is None:
            if abs(axial_delta) >= self.hard_absolute_axial_delta_N:
                self.stop_reason = "hard_absolute_axial_force_cutoff"
            elif (
                self.wrong_direction_since_s is not None
                and elapsed_s - self.wrong_direction_since_s
                >= self.wrong_direction_persistence_s
            ):
                self.stop_reason = "persistent_wrong_direction_force"
            elif axial_travel >= self.maximum_axial_travel_m:
                self.stop_reason = "maximum_axial_travel"
            elif lateral_translation >= self.maximum_lateral_translation_m:
                self.stop_reason = "maximum_lateral_translation"
            elif elapsed_s >= self.maximum_duration_s:
                self.stop_reason = "timeout"
        return axial_delta
