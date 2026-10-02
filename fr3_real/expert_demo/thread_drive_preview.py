"""Pure validation helpers for the first guarded bolt thread-drive preview."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def half_cosine_duration(angle_rad: float, peak_speed_rad_s: float) -> float:
    """Return duration for theta(t)=a/2*(1-cos(pi*t/T))."""
    if not math.isfinite(angle_rad) or angle_rad <= 0.0:
        raise ValueError("drive angle must be finite and positive")
    if not math.isfinite(peak_speed_rad_s) or peak_speed_rad_s <= 0.0:
        raise ValueError("peak speed must be finite and positive")
    return angle_rad * math.pi / (2.0 * peak_speed_rad_s)


def preview_blockers(
    *,
    live_q7_rad: float,
    start_q7_rad: float,
    start_tolerance_rad: float,
    target_q7_rad: float,
    cable_interval_rad: Sequence[float],
    dynamic_model_interval_rad: Sequence[float],
) -> list[str]:
    """Validate only already calibrated q7/model constraints; command no motion."""
    cable_min, cable_max = map(float, cable_interval_rad)
    model_min, model_max = map(float, dynamic_model_interval_rad)
    values = np.asarray(
        [live_q7_rad, start_q7_rad, start_tolerance_rad, target_q7_rad,
         cable_min, cable_max, model_min, model_max],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        return ["q7 preview contains NaN/Inf"]
    blockers: list[str] = []
    if abs(live_q7_rad - start_q7_rad) > start_tolerance_rad:
        blockers.append("live q7 is not near the commissioned drive start")
    if live_q7_rad < cable_min or target_q7_rad > cable_max:
        blockers.append("planned q7 interval leaves the operator-calibrated cable-safe range")
    if live_q7_rad < model_min or target_q7_rad > model_max:
        blockers.append("planned q7 interval leaves the calibrated dynamic torque model")
    return blockers
