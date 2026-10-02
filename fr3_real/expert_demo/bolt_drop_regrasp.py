"""Capture contract for mechanically normalizing a bolt before threading."""

from __future__ import annotations

from typing import Any

from .replay_plan import ReplayStagePlan, build_replay_stage


EXPECTED_BOLT_DROP_REGRASP_EVENTS = (
    "trajectory_started_preinsert_held",
    "drop_pose_reached_held",
    "bolt_released_upright",
    "thread_regrasp_pose_reached_open",
    "thread_ready_regrasped",
)

BOLT_DROP_REGRASP_STAGE_EVENTS = {
    "lower_held": ("trajectory_started_preinsert_held", "drop_pose_reached_held"),
    "reposition_regrasp_open": (
        "bolt_released_upright",
        "thread_regrasp_pose_reached_open",
    ),
}


def build_bolt_drop_regrasp_stage(
    trajectory_dir: str,
    stage: str,
    *,
    command_rate_hz: float,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
) -> ReplayStagePlan:
    return build_replay_stage(
        trajectory_dir,
        stage,
        command_rate_hz=command_rate_hz,
        expected_events=EXPECTED_BOLT_DROP_REGRASP_EVENTS,
        stage_events=BOLT_DROP_REGRASP_STAGE_EVENTS,
        trim_start_s=trim_start_s,
        trim_end_s=trim_end_s,
    )


def bolt_drop_regrasp_metadata_blockers(metadata: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if metadata.get("task_name") != "bolt":
        blockers.append("task_name must be bolt")
    if metadata.get("capture_scope") != "preinsert_through_thread_ready_only":
        blockers.append("capture_scope must be preinsert_through_thread_ready_only")
    if metadata.get("mechanical_normalization_method") != "release_into_guide_then_regrasp_head":
        blockers.append("mechanical normalization method must release then regrasp")
    if metadata.get("bolt_upright_at_endpoint") is not True:
        blockers.append("bolt must be upright at the endpoint")
    if metadata.get("bolt_regrasped_at_endpoint") is not True:
        blockers.append("bolt must be regrasped at the endpoint")
    if metadata.get("rotation_or_threading_included") is not False:
        blockers.append("rotation/threading must be excluded")
    if metadata.get("cameras_recorded") is not False:
        blockers.append("calibration capture must remain camera-free")
    return blockers
