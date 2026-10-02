"""Pure validation helpers for guarded piloted-trajectory stage execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from .replay_plan import ReplayStagePlan


NOT_CALIBRATED = "NOT_CALIBRATED"


def smoothed_position_velocities(
    plan: ReplayStagePlan,
    time_scale: float,
    smoothing_passes: int,
) -> np.ndarray:
    """Derive velocity targets from q positions and smooth measurement noise.

    The authoritative replay remains the recorded joint-position path. Boundary
    velocities are explicitly zero because every semantic stage was recorded
    while the arm was stationary.
    """
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes must be non-negative")
    scaled_times = plan.command_times_s * float(time_scale)
    velocities = np.gradient(plan.q_targets, scaled_times, axis=0)
    velocities[0] = 0.0
    velocities[-1] = 0.0
    for _ in range(smoothing_passes):
        filtered = velocities.copy()
        filtered[1:-1] = (
            velocities[:-2] + 2.0 * velocities[1:-1] + velocities[2:]
        ) / 4.0
        velocities = filtered
        velocities[0] = 0.0
        velocities[-1] = 0.0
    return velocities


def stage_config_blockers(stage: dict[str, Any], plan: ReplayStagePlan) -> list[str]:
    """Return static blockers without importing Franky or connecting to hardware."""
    blockers: list[str] = []
    if stage.get("enabled") is not True:
        blockers.append("stage execution is disabled")
    if stage.get("empty_workspace_only") is not True:
        blockers.append("first stage replay must be explicitly empty-workspace-only")
    if stage.get("gripper_commands_enabled") is not False:
        blockers.append("gripper commands must remain disabled for first stage replay")
    for key in (
        "playback_time_scale",
        "relative_dynamics_factor",
        "joint_margin_rad",
        "start_joint_tolerance_rad",
        "terminal_joint_tolerance_rad",
        "maximum_scaled_target_velocity_rad_s",
        "maximum_target_step_rad",
        "workspace_bounds_m",
        "maximum_fk_path_error_m",
        "state_timeout_s",
    ):
        value = stage.get(key)
        if value is None or value == NOT_CALIBRATED:
            blockers.append(f"stage.{key} is NOT_CALIBRATED")

    numeric_positive = (
        "playback_time_scale",
        "start_joint_tolerance_rad",
        "terminal_joint_tolerance_rad",
        "maximum_scaled_target_velocity_rad_s",
        "maximum_target_step_rad",
        "maximum_fk_path_error_m",
        "state_timeout_s",
    )
    for key in numeric_positive:
        value = stage.get(key)
        if isinstance(value, (int, float)) and float(value) <= 0.0:
            blockers.append(f"stage.{key} must be positive")
    dynamics = stage.get("relative_dynamics_factor")
    if isinstance(dynamics, (int, float)) and not 0.0 < float(dynamics) <= 1.0:
        blockers.append("stage.relative_dynamics_factor must be in (0, 1]")
    margin = stage.get("joint_margin_rad")
    if isinstance(margin, (int, float)) and not 0.0 <= float(margin) < 0.5:
        blockers.append("stage.joint_margin_rad must be in [0, 0.5)")

    configured_trajectory = stage.get("trajectory")
    if configured_trajectory and configured_trajectory != plan.trajectory_name:
        blockers.append(
            f"stage is commissioned for trajectory {configured_trajectory}, not {plan.trajectory_name}"
        )
    time_scale = stage.get("playback_time_scale")
    velocity_cap = stage.get("maximum_scaled_target_velocity_rad_s")
    if isinstance(time_scale, (int, float)) and isinstance(velocity_cap, (int, float)):
        scaled_peak = float(np.max(np.abs(plan.dq_targets))) / float(time_scale)
        if scaled_peak > float(velocity_cap):
            blockers.append(
                f"scaled target velocity {scaled_peak:.6f} rad/s exceeds "
                f"configured maximum {float(velocity_cap):.6f} rad/s"
            )
    step_cap = stage.get("maximum_target_step_rad")
    if isinstance(step_cap, (int, float)) and plan.max_target_step_rad > float(step_cap):
        blockers.append(
            f"target step {plan.max_target_step_rad:.6f} rad exceeds configured "
            f"maximum {float(step_cap):.6f} rad"
        )
    workspace = stage.get("workspace_bounds_m")
    if isinstance(workspace, dict):
        minimum = np.asarray(workspace.get("min_xyz"), dtype=np.float64)
        maximum = np.asarray(workspace.get("max_xyz"), dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            blockers.append("stage.workspace_bounds_m is invalid")
        elif np.any(plan.xyz_targets < minimum) or np.any(plan.xyz_targets > maximum):
            blockers.append("recorded stage path leaves stage.workspace_bounds_m")
    return blockers
