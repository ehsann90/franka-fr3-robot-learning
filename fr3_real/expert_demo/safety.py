"""Pure safety validation used before any expert motion is commissioned."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .transforms import validate_transform


NOT_CALIBRATED = "NOT_CALIBRATED"


class SafetyError(ValueError):
    """Raised when a target violates configured safety bounds."""


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum_xyz_m: tuple[float, float, float]
    maximum_xyz_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum_xyz_m, dtype=float)
        maximum = np.asarray(self.maximum_xyz_m, dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise SafetyError("workspace bounds must contain three-axis minimum and maximum values")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise SafetyError("workspace bounds must be finite")
        if np.any(minimum >= maximum):
            raise SafetyError("every workspace minimum must be smaller than its maximum")

    def validate_pose(self, transform: Any, *, label: str) -> None:
        matrix = validate_transform(transform, name=label)
        xyz = matrix[:3, 3]
        minimum = np.asarray(self.minimum_xyz_m)
        maximum = np.asarray(self.maximum_xyz_m)
        if np.any(xyz < minimum) or np.any(xyz > maximum):
            raise SafetyError(
                f"{label} xyz={xyz.tolist()} is outside workspace "
                f"min={minimum.tolist()} max={maximum.tolist()}"
            )


def workspace_from_config(config: dict[str, Any]) -> WorkspaceBounds | None:
    bounds = config.get("safety", {}).get("workspace_bounds_m", NOT_CALIBRATED)
    if bounds in (None, NOT_CALIBRATED):
        return None
    if not isinstance(bounds, dict):
        raise SafetyError("safety.workspace_bounds_m must be NOT_CALIBRATED or an object")
    return WorkspaceBounds(tuple(bounds["min_xyz"]), tuple(bounds["max_xyz"]))


def execution_readiness_blockers(config: dict[str, Any]) -> list[str]:
    """Return configuration reasons that prohibit real bracket execution."""

    blockers: list[str] = []
    if not config.get("execution_enabled", False):
        blockers.append("execution_enabled is false")
    if workspace_from_config(config) is None:
        blockers.append("Cartesian workspace bounds are NOT_CALIBRATED")
    motion = config.get("motion", {})
    for name in (
        "dynamics_factor",
        "max_cartesian_travel_per_state_m",
        "state_timeout_s",
        "grasp_width_m",
        "grasp_speed_m_s",
        "grasp_force_n",
        "open_speed_m_s",
    ):
        if motion.get(name, NOT_CALIBRATED) in (None, NOT_CALIBRATED):
            blockers.append(f"motion.{name} is NOT_CALIBRATED")
    return blockers
