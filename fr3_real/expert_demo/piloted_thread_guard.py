"""Control-rate safety guard for a guarded Pilot thread-drive replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PilotedThreadReplayGuard:
    path_q: np.ndarray
    model_q7: np.ndarray
    model_K_Tz: np.ndarray
    baseline_Fz_N: float
    hard_absolute_axial_delta_N: float
    hard_negative_residual_Tz_Nm: float
    q7_min_rad: float
    q7_max_rad: float
    maximum_q1_q6_path_error_rad: float
    stop_reason: str | None = None
    minimum_residual_Tz_Nm: float = np.inf
    minimum_axial_delta_N: float = np.inf
    maximum_axial_delta_N: float = -np.inf
    maximum_q1_q6_path_error_observed_rad: float = 0.0

    def __post_init__(self) -> None:
        self.path_q = np.asarray(self.path_q, dtype=np.float64)
        self.model_q7 = np.asarray(self.model_q7, dtype=np.float64)
        self.model_K_Tz = np.asarray(self.model_K_Tz, dtype=np.float64)
        if self.path_q.ndim != 2 or self.path_q.shape[1] != 7 or len(self.path_q) < 2:
            raise ValueError("path_q must be Nx7")
        if np.any(np.diff(self.path_q[:, 6]) < -0.01):
            raise ValueError("thread replay q7 path must be approximately monotonic")
        if self.model_q7.ndim != 1 or self.model_K_Tz.shape != self.model_q7.shape:
            raise ValueError("dynamic torque model dimensions are invalid")

    def update(
        self,
        q: Sequence[float],
        wrench_K: Sequence[float],
    ) -> tuple[float, float, float]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        wrench = np.asarray(wrench_K, dtype=np.float64).reshape(-1)
        if q.size != 7 or wrench.size != 6 or not np.all(np.isfinite(q)) or not np.all(np.isfinite(wrench)):
            self.stop_reason = self.stop_reason or "invalid_control_state"
            return np.nan, np.nan, np.nan
        expected_Tz = float(np.interp(q[6], self.model_q7, self.model_K_Tz))
        residual_Tz = float(wrench[5] - expected_Tz)
        axial_delta = float(wrench[2] - self.baseline_Fz_N)
        expected_q1_q6 = np.asarray(
            [np.interp(q[6], self.path_q[:, 6], self.path_q[:, joint]) for joint in range(6)]
        )
        path_error = float(np.max(np.abs(q[:6] - expected_q1_q6)))
        self.minimum_residual_Tz_Nm = min(self.minimum_residual_Tz_Nm, residual_Tz)
        self.minimum_axial_delta_N = min(self.minimum_axial_delta_N, axial_delta)
        self.maximum_axial_delta_N = max(self.maximum_axial_delta_N, axial_delta)
        self.maximum_q1_q6_path_error_observed_rad = max(
            self.maximum_q1_q6_path_error_observed_rad, path_error
        )
        if self.stop_reason is None:
            if abs(axial_delta) >= self.hard_absolute_axial_delta_N:
                self.stop_reason = "hard_absolute_axial_force_cutoff"
            elif residual_Tz <= self.hard_negative_residual_Tz_Nm:
                self.stop_reason = "hard_residual_torque_cutoff"
            elif q[6] < self.q7_min_rad or q[6] > self.q7_max_rad:
                self.stop_reason = "q7_cable_limit"
            elif path_error >= self.maximum_q1_q6_path_error_rad:
                self.stop_reason = "q1_q6_path_tracking_limit"
        return residual_Tz, axial_delta, path_error
