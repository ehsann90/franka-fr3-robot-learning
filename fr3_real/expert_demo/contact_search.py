"""Pure runtime guard logic for a bounded +Kz bolt contact search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactSearchGuard:
    baseline_force_n: float
    contact_delta_n: float
    contact_persistence_s: float
    hard_delta_n: float
    maximum_travel_m: float
    maximum_duration_s: float
    start_xyz_m: np.ndarray
    axis_base: np.ndarray
    first_above_time_s: float | None = None
    stop_reason: str | None = None
    peak_force_delta_n: float = float("-inf")
    maximum_travel_observed_m: float = 0.0

    def __post_init__(self) -> None:
        self.start_xyz_m = np.asarray(self.start_xyz_m, dtype=np.float64)
        self.axis_base = np.asarray(self.axis_base, dtype=np.float64)
        if self.start_xyz_m.shape != (3,) or self.axis_base.shape != (3,):
            raise ValueError("start position and axis must each contain three values")
        norm = float(np.linalg.norm(self.axis_base))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("axis must be finite and non-zero")
        self.axis_base = self.axis_base / norm
        for name in (
            "contact_delta_n",
            "contact_persistence_s",
            "hard_delta_n",
            "maximum_travel_m",
            "maximum_duration_s",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hard_delta_n <= self.contact_delta_n:
            raise ValueError("hard force delta must exceed the contact delta")

    def update(self, elapsed_s: float, force_kz_n: float, xyz_m: np.ndarray) -> str | None:
        if self.stop_reason is not None:
            return self.stop_reason
        delta = float(force_kz_n) - float(self.baseline_force_n)
        self.peak_force_delta_n = max(self.peak_force_delta_n, delta)
        travel = float(np.dot(np.asarray(xyz_m, dtype=np.float64) - self.start_xyz_m, self.axis_base))
        self.maximum_travel_observed_m = max(self.maximum_travel_observed_m, travel)
        if delta >= self.hard_delta_n:
            self.stop_reason = "hard_axial_force_cutoff"
        elif travel >= self.maximum_travel_m:
            self.stop_reason = "maximum_travel"
        elif delta >= self.contact_delta_n:
            if self.first_above_time_s is None:
                self.first_above_time_s = float(elapsed_s)
            elif float(elapsed_s) - self.first_above_time_s >= self.contact_persistence_s:
                self.stop_reason = "contact_detected"
        else:
            self.first_above_time_s = None
        if self.stop_reason is None and float(elapsed_s) >= self.maximum_duration_s:
            self.stop_reason = "timeout"
        return self.stop_reason


def rotation_angle_rad(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))
