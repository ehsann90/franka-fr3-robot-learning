"""Pure commissioning checks for a piloted bracket object episode."""

from __future__ import annotations

from typing import Any

import numpy as np

from .replay_execution import smoothed_position_velocities
from .replay_plan import ReplayStagePlan


REQUIRED_STAGES = ("approach_open", "transport_place_closed", "retreat_open")


def automatic_reset_blockers(
    reset: dict[str, Any],
    current_q: np.ndarray,
    target_q: np.ndarray,
    path_xyz: np.ndarray,
    target_fk_error_m: float,
) -> list[str]:
    """Validate an automatically computed empty-gripper reset path."""
    blockers: list[str] = []
    if reset.get("enabled") is not True:
        blockers.append("automatic reset is disabled")
    for key in (
        "relative_dynamics_factor",
        "joint_margin_rad",
        "maximum_joint_delta_rad",
        "workspace_bounds_m",
        "target_fk_error_limit_m",
        "terminal_joint_tolerance_rad",
    ):
        if reset.get(key) is None or reset.get(key) == "NOT_CALIBRATED":
            blockers.append(f"automatic_reset_to_start.{key} is NOT_CALIBRATED")
    if reset.get("requires_empty_gripper") is not True:
        blockers.append("automatic reset must require an empty gripper")
    object_removed = reset.get(
        "requires_object_removed", reset.get("requires_bracket_removed")
    )
    if object_removed is not True:
        blockers.append("automatic reset must require the task object to be removed")
    dynamics = reset.get("relative_dynamics_factor")
    if isinstance(dynamics, (int, float)) and not 0.0 < float(dynamics) <= 1.0:
        blockers.append("automatic reset dynamics factor must be in (0, 1]")
    margin = reset.get("joint_margin_rad")
    if isinstance(margin, (int, float)) and not 0.0 <= float(margin) < 0.5:
        blockers.append("automatic reset joint margin must be in [0, 0.5)")
    maximum_delta = reset.get("maximum_joint_delta_rad")
    if isinstance(maximum_delta, (int, float)):
        actual_delta = float(np.max(np.abs(target_q - current_q)))
        if actual_delta > float(maximum_delta):
            blockers.append(
                f"reset joint delta {actual_delta:.6f} rad exceeds {float(maximum_delta):.6f} rad"
            )
    workspace = reset.get("workspace_bounds_m")
    if isinstance(workspace, dict):
        minimum = np.asarray(workspace.get("min_xyz"), dtype=np.float64)
        maximum = np.asarray(workspace.get("max_xyz"), dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            blockers.append("automatic reset workspace is invalid")
        elif np.any(path_xyz < minimum) or np.any(path_xyz > maximum):
            blockers.append("computed automatic reset path leaves its configured workspace")
    elif workspace != "NOT_CALIBRATED":
        blockers.append("automatic reset workspace is invalid")
    fk_limit = reset.get("target_fk_error_limit_m")
    if isinstance(fk_limit, (int, float)) and target_fk_error_m > float(fk_limit):
        blockers.append(
            f"reset target FK error {target_fk_error_m:.6f} m exceeds {float(fk_limit):.6f} m"
        )
    return blockers


def object_episode_blockers(config: dict[str, Any], plans: dict[str, ReplayStagePlan]) -> list[str]:
    blockers: list[str] = []
    if config.get("enabled") is not True:
        blockers.append("object episode execution is disabled")
    for name in REQUIRED_STAGES:
        if name not in plans:
            blockers.append(f"missing replay plan {name}")
    if blockers:
        return blockers
    trajectory = config.get("trajectory")
    if any(plan.trajectory_name != trajectory for plan in plans.values()):
        blockers.append("object episode is not commissioned for this trajectory")
    for key in (
        "playback_time_scale",
        "stage_robot_global_dynamics_factor",
        "relative_dynamics_factor",
        "joint_margin_rad",
        "start_joint_tolerance_rad",
        "terminal_joint_tolerance_rad",
        "maximum_scaled_target_velocity_rad_s",
        "maximum_command_velocity_jump_rad_s",
        "command_velocity_source",
        "maximum_target_step_rad",
        "command_velocity_smoothing_passes",
        "stage_lead_in_ms",
        "stage_lead_in_hold_ms",
        "stage_lead_in_enabled",
        "workspace_bounds_m",
        "maximum_fk_path_error_m",
        "state_timeout_s",
        "automatic_reset_to_start",
    ):
        if config.get(key) is None or config.get(key) == "NOT_CALIBRATED":
            blockers.append(f"object_episode_execution.{key} is NOT_CALIBRATED")
    scale = config.get("playback_time_scale")
    dynamics = config.get("relative_dynamics_factor")
    global_dynamics = config.get("stage_robot_global_dynamics_factor")
    if isinstance(scale, (int, float)) and float(scale) <= 0:
        blockers.append("playback_time_scale must be positive")
    if isinstance(dynamics, (int, float)) and not 0 < float(dynamics) <= 1:
        blockers.append("relative_dynamics_factor must be in (0, 1]")
    if isinstance(global_dynamics, (int, float)) and not 0 < float(global_dynamics) <= 1:
        blockers.append("stage_robot_global_dynamics_factor must be in (0, 1]")
    if not isinstance(config.get("stage_lead_in_enabled"), bool):
        blockers.append("stage_lead_in_enabled must be boolean")
    if isinstance(scale, (int, float)):
        smoothing_passes = config.get("command_velocity_smoothing_passes")
        if not isinstance(smoothing_passes, int) or smoothing_passes < 0:
            blockers.append("command_velocity_smoothing_passes must be a non-negative integer")
            smoothing_passes = 0
        velocity_source = config.get("command_velocity_source")
        if velocity_source == "recorded_scaled":
            command_velocities = [plan.dq_targets / float(scale) for plan in plans.values()]
        elif velocity_source == "position_derivative_smoothed":
            command_velocities = [
                smoothed_position_velocities(plan, float(scale), smoothing_passes)
                for plan in plans.values()
            ]
        else:
            blockers.append(f"unknown command_velocity_source {velocity_source!r}")
            command_velocities = [np.full_like(plan.dq_targets, np.inf) for plan in plans.values()]
        scaled_peak = max(float(np.max(np.abs(velocities))) for velocities in command_velocities)
        cap = config.get("maximum_scaled_target_velocity_rad_s")
        if isinstance(cap, (int, float)) and scaled_peak > float(cap):
            blockers.append(f"scaled target velocity {scaled_peak:.6f} rad/s exceeds {float(cap):.6f}")
        maximum_jump = max(
            float(np.max(np.abs(np.diff(velocities, axis=0))))
            for velocities in command_velocities
        )
        jump_cap = config.get("maximum_command_velocity_jump_rad_s")
        if isinstance(jump_cap, (int, float)) and maximum_jump > float(jump_cap):
            blockers.append(
                f"command velocity jump {maximum_jump:.6f} rad/s exceeds {float(jump_cap):.6f}"
            )
    step_cap = config.get("maximum_target_step_rad")
    if isinstance(step_cap, (int, float)):
        maximum_step = max(plan.max_target_step_rad for plan in plans.values())
        if maximum_step > float(step_cap):
            blockers.append(f"target step {maximum_step:.6f} rad exceeds {float(step_cap):.6f}")
    gripper = config.get("gripper", {})
    for key in (
        "open_speed_m_s",
        "minimum_open_width_m",
        "grasp_target_width_m",
        "grasp_speed_m_s",
        "grasp_force_n",
        "epsilon_inner_m",
        "epsilon_outer_m",
        "held_width_min_m",
        "held_width_max_m",
        "monitor_hz",
    ):
        value = gripper.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            blockers.append(f"object_episode_execution.gripper.{key} must be positive")
    held_min = gripper.get("held_width_min_m")
    held_max = gripper.get("held_width_max_m")
    if isinstance(held_min, (int, float)) and isinstance(held_max, (int, float)):
        if float(held_min) >= float(held_max):
            blockers.append("held gripper width range is invalid")
    if config.get("camera_recording_enabled") is not False:
        blockers.append("camera recording must remain disabled for first live-object run")
    return blockers
