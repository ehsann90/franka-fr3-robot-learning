"""Offline-only staging of piloted bracket trajectories for later replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_EVENTS = (
    "trajectory_started",
    "grasp_pose_reached",
    "gripper_closed",
    "place_pose_reached",
    "gripper_opened",
    "retreat_complete",
)

STAGE_EVENTS = {
    "full_arm_no_gripper": ("trajectory_started", "retreat_complete"),
    "approach_open": ("trajectory_started", "grasp_pose_reached"),
    "transport_place_closed": ("gripper_closed", "place_pose_reached"),
    "retreat_open": ("gripper_opened", "retreat_complete"),
}


class ReplayPlanError(ValueError):
    """Raised when a raw trajectory cannot be staged safely offline."""


@dataclass(frozen=True)
class ReplayStagePlan:
    trajectory_name: str
    name: str
    start_event: str
    stop_event: str
    source_start_index: int
    source_stop_index: int
    source_duration_s: float
    command_rate_hz: float
    command_times_s: np.ndarray
    q_targets: np.ndarray
    dq_targets: np.ndarray
    xyz_targets: np.ndarray
    expected_start_q: np.ndarray
    expected_stop_q: np.ndarray
    source_xyz_min: np.ndarray
    source_xyz_max: np.ndarray
    max_source_abs_dq_rad_s: float
    max_target_step_rad: float
    implied_peak_velocity_rad_s: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReplayPlanError(f"missing {path}")
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayPlanError(f"cannot read {path}: {error}") from error


def _nearest_index(times: np.ndarray, query: int) -> int:
    return int(np.argmin(np.abs(times - int(query))))


def build_replay_stage(
    trajectory_dir: str | Path,
    stage: str,
    *,
    command_rate_hz: float,
    expected_events: tuple[str, ...] = EXPECTED_EVENTS,
    stage_events: dict[str, tuple[str, str]] = STAGE_EVENTS,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
) -> ReplayStagePlan:
    trajectory_dir = Path(trajectory_dir).expanduser().resolve()
    if stage not in stage_events:
        raise ReplayPlanError(f"unknown replay stage {stage!r}; choices={sorted(stage_events)}")
    if command_rate_hz <= 0:
        raise ReplayPlanError("command_rate_hz must be positive")
    if trim_start_s < 0 or trim_end_s < 0:
        raise ReplayPlanError("stationary trim durations must be non-negative")
    try:
        metadata = json.loads((trajectory_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayPlanError(f"cannot read trajectory metadata: {error}") from error
    if not metadata.get("accepted_by_operator", False):
        raise ReplayPlanError("trajectory was not accepted by the operator")
    if metadata.get("sample_error_count") != 0:
        raise ReplayPlanError("trajectory contains state sampling errors")

    samples = read_jsonl(trajectory_dir / "trajectory.jsonl")
    events = read_jsonl(trajectory_dir / "events.jsonl")
    event_names = tuple(event.get("event") for event in events)
    if event_names != expected_events:
        raise ReplayPlanError(f"event sequence is {event_names}; expected {expected_events}")
    if len(samples) < 2:
        raise ReplayPlanError("trajectory has fewer than two samples")

    time_ns = np.asarray([sample["monotonic_time_ns"] for sample in samples], dtype=np.int64)
    if np.any(np.diff(time_ns) <= 0):
        raise ReplayPlanError("trajectory sample times are not strictly increasing")
    q = np.asarray([sample["q"] for sample in samples], dtype=np.float64)
    dq = np.asarray([sample["dq"] for sample in samples], dtype=np.float64)
    xyz = np.asarray([sample["translation_xyz_m"] for sample in samples], dtype=np.float64)
    if q.shape != (len(samples), 7) or dq.shape != q.shape or xyz.shape != (len(samples), 3):
        raise ReplayPlanError("trajectory state arrays have invalid dimensions")
    if not all(np.isfinite(array).all() for array in (q, dq, xyz)):
        raise ReplayPlanError("trajectory contains NaN or infinite state")

    event_by_name = {event["event"]: event for event in events}
    start_event, stop_event = stage_events[stage]
    start_query_ns = int(event_by_name[start_event]["monotonic_time_ns"]) + int(
        float(trim_start_s) * 1e9
    )
    stop_query_ns = int(event_by_name[stop_event]["monotonic_time_ns"]) - int(
        float(trim_end_s) * 1e9
    )
    start_index = _nearest_index(time_ns, start_query_ns)
    stop_index = _nearest_index(time_ns, stop_query_ns)
    if stop_index <= start_index:
        raise ReplayPlanError(f"stage {stage} has a non-positive sample range")

    segment_time_s = (time_ns[start_index : stop_index + 1] - time_ns[start_index]) / 1e9
    duration_s = float(segment_time_s[-1])
    command_times_s = np.arange(0.0, duration_s, 1.0 / command_rate_hz)
    if not len(command_times_s) or command_times_s[-1] < duration_s:
        command_times_s = np.append(command_times_s, duration_s)
    segment_q = q[start_index : stop_index + 1]
    segment_dq = dq[start_index : stop_index + 1]
    segment_xyz = xyz[start_index : stop_index + 1]
    q_targets = np.column_stack(
        [np.interp(command_times_s, segment_time_s, segment_q[:, joint]) for joint in range(7)]
    )
    dq_targets = np.column_stack(
        [np.interp(command_times_s, segment_time_s, segment_dq[:, joint]) for joint in range(7)]
    )
    xyz_targets = np.column_stack(
        [np.interp(command_times_s, segment_time_s, segment_xyz[:, axis]) for axis in range(3)]
    )
    target_steps = np.diff(q_targets, axis=0)
    max_target_step = float(np.max(np.abs(target_steps))) if len(target_steps) else 0.0

    return ReplayStagePlan(
        trajectory_name=trajectory_dir.name,
        name=stage,
        start_event=start_event,
        stop_event=stop_event,
        source_start_index=start_index,
        source_stop_index=stop_index,
        source_duration_s=duration_s,
        command_rate_hz=float(command_rate_hz),
        command_times_s=command_times_s,
        q_targets=q_targets,
        dq_targets=dq_targets,
        xyz_targets=xyz_targets,
        expected_start_q=q_targets[0].copy(),
        expected_stop_q=q_targets[-1].copy(),
        source_xyz_min=xyz[start_index : stop_index + 1].min(axis=0),
        source_xyz_max=xyz[start_index : stop_index + 1].max(axis=0),
        max_source_abs_dq_rad_s=float(np.max(np.abs(dq[start_index : stop_index + 1]))),
        max_target_step_rad=max_target_step,
        implied_peak_velocity_rad_s=max_target_step * command_rate_hz,
    )


def replay_readiness_blockers(config: dict[str, Any], plan: ReplayStagePlan) -> list[str]:
    replay = config.get("replay", {})
    blockers: list[str] = []
    if replay.get("enabled") is not True:
        blockers.append("replay.enabled is false")
    for key in (
        "workspace_bounds_m",
        "joint_velocity_cap_rad_s",
        "start_joint_tolerance_rad",
    ):
        if replay.get(key) == "NOT_CALIBRATED" or replay.get(key) is None:
            blockers.append(f"replay.{key} is NOT_CALIBRATED")
    workspace = replay.get("workspace_bounds_m")
    if isinstance(workspace, dict):
        minimum = np.asarray(workspace.get("min_xyz"), dtype=np.float64)
        maximum = np.asarray(workspace.get("max_xyz"), dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            blockers.append("replay.workspace_bounds_m is invalid")
        elif np.any(plan.source_xyz_min < minimum) or np.any(plan.source_xyz_max > maximum):
            blockers.append(
                "stage source EE envelope leaves configured workspace: "
                f"min={plan.source_xyz_min.tolist()} max={plan.source_xyz_max.tolist()}"
            )
    velocity_cap = replay.get("joint_velocity_cap_rad_s")
    if isinstance(velocity_cap, (int, float)) and plan.implied_peak_velocity_rad_s > float(velocity_cap):
        blockers.append(
            f"stage implied peak joint velocity {plan.implied_peak_velocity_rad_s:.4f} rad/s "
            f"exceeds configured cap {float(velocity_cap):.4f} rad/s"
        )
    return blockers
