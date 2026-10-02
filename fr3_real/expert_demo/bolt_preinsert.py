"""Schema constants and pure validation for piloted bolt PREINSERT captures."""

from __future__ import annotations

from typing import Any

from .replay_plan import ReplayStagePlan, build_replay_stage


EXPECTED_BOLT_PREINSERT_EVENTS = (
    "trajectory_started",
    "grasp_pose_reached",
    "gripper_closed",
    "preinsert_pose_reached",
)

BOLT_PREINSERT_STAGE_EVENTS = {
    "approach_open": ("trajectory_started", "grasp_pose_reached"),
    "transport_preinsert_closed": ("gripper_closed", "preinsert_pose_reached"),
}


def build_bolt_preinsert_stage(
    trajectory_dir: str,
    stage: str,
    *,
    command_rate_hz: float,
) -> ReplayStagePlan:
    return build_replay_stage(
        trajectory_dir,
        stage,
        command_rate_hz=command_rate_hz,
        expected_events=EXPECTED_BOLT_PREINSERT_EVENTS,
        stage_events=BOLT_PREINSERT_STAGE_EVENTS,
    )


def bolt_capture_metadata_blockers(metadata: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if metadata.get("task_name") != "bolt":
        blockers.append("task_name must be bolt")
    if metadata.get("capture_scope") != "pickup_through_preinsert_only":
        blockers.append("capture_scope must stop at PREINSERT")
    if metadata.get("contact_or_threading_included") is not False:
        blockers.append("contact/threading must be excluded from the PREINSERT capture")
    if metadata.get("cameras_recorded") is not False:
        blockers.append("piloted calibration capture must remain camera-free")
    return blockers
