"""Control-rate guards for the first bolt-held, unengaged wrist rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BoltFreeSpinGuard:
    model_q7: np.ndarray
    model_K_Tz: np.ndarray
    baseline_Fz_N: float
    hard_negative_residual_Tz_Nm: float
    hard_positive_axial_delta_N: float
    q7_min_rad: float
    q7_max_rad: float
    start_xyz_m: np.ndarray
    maximum_tcp_translation_m: float
    stop_reason: str | None = None
    minimum_residual_Tz_Nm: float = np.inf
    maximum_axial_delta_N: float = -np.inf
    maximum_tcp_translation_observed_m: float = 0.0

    def update(self, q7_rad: float, wrench_K: Sequence[float], xyz_m: Sequence[float]) -> tuple[float, float]:
        wrench = np.asarray(wrench_K, dtype=np.float64).reshape(-1)
        xyz = np.asarray(xyz_m, dtype=np.float64).reshape(-1)
        if wrench.size != 6 or xyz.size != 3 or not np.isfinite(wrench).all() or not np.isfinite(xyz).all():
            self.stop_reason = self.stop_reason or "invalid_control_state"
            return np.nan, np.nan
        expected_Tz = float(np.interp(q7_rad, self.model_q7, self.model_K_Tz))
        residual_Tz = float(wrench[5] - expected_Tz)
        axial_delta = float(wrench[2] - self.baseline_Fz_N)
        travel = float(np.linalg.norm(xyz - self.start_xyz_m))
        self.minimum_residual_Tz_Nm = min(self.minimum_residual_Tz_Nm, residual_Tz)
        self.maximum_axial_delta_N = max(self.maximum_axial_delta_N, axial_delta)
        self.maximum_tcp_translation_observed_m = max(self.maximum_tcp_translation_observed_m, travel)
        if self.stop_reason is None:
            if residual_Tz <= self.hard_negative_residual_Tz_Nm:
                self.stop_reason = "hard_residual_torque_cutoff"
            elif axial_delta >= self.hard_positive_axial_delta_N:
                self.stop_reason = "hard_axial_force_cutoff"
            elif q7_rad < self.q7_min_rad or q7_rad > self.q7_max_rad:
                self.stop_reason = "q7_cable_limit"
            elif travel >= self.maximum_tcp_translation_m:
                self.stop_reason = "tcp_translation_limit"
        return residual_Tz, axial_delta
