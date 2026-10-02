#!/usr/bin/env python3
"""Record a commissioned bolt demo, optionally after bracket placement."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_drop_regrasp import (  # noqa: E402
    BOLT_DROP_REGRASP_STAGE_EVENTS,
    build_bolt_drop_regrasp_stage,
)
from expert_demo.bolt_preinsert import (  # noqa: E402
    BOLT_PREINSERT_STAGE_EVENTS,
    build_bolt_preinsert_stage,
)
from expert_demo.object_episode_execution import (  # noqa: E402
    REQUIRED_STAGES,
    automatic_reset_blockers,
    object_episode_blockers,
)
from expert_demo.replay_plan import (  # noqa: E402
    ReplayPlanError,
    build_replay_stage,
)
from expert_demo.retimed_trajectory import (  # noqa: E402
    build_retimed_trajectory,
    write_retimed_csv,
)


DEFAULT_RECORDING_CONFIG = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def build_configured_2x_trajectory(plan, config: dict):
    """Apply the exact commissioned 2x retiming and idle-edge trim profile."""

    return build_configured_accelerated_trajectory(plan, config)


def build_configured_accelerated_trajectory(
    plan,
    config: dict,
    stage_name: str | None = None,
):
    """Apply one guarded accelerated retiming and idle-edge trim profile.

    The passed 2x profile uses one global source-time scale.  Faster production
    profiles may specify a per-stage scale so a requested speedup is reduced
    only where the existing velocity, acceleration, or jerk cap requires it.
    """

    idle_trim = config.get("idle_edge_trim", {})
    scale_by_stage = config.get("playback_time_scale_by_stage", {})
    if stage_name is not None and stage_name in scale_by_stage:
        playback_time_scale = float(scale_by_stage[stage_name])
    else:
        playback_time_scale = float(config["playback_time_scale"])
    return build_retimed_trajectory(
        plan,
        playback_time_scale=playback_time_scale,
        validation_rate_hz=float(config["offline_validation_rate_hz"]),
        idle_trim_velocity_threshold_rad_s=(
            float(idle_trim["interval_velocity_threshold_rad_s"])
            if idle_trim.get("enabled") is True else 0.0
        ),
        idle_trim_buffer_s=(
            float(idle_trim["source_time_buffer_s"])
            if idle_trim.get("enabled") is True else 0.0
        ),
    )


def pickup_grasp_evidence(
    grasp_config: dict,
    *,
    accepted: bool,
    is_grasped: bool,
    width_m: float,
) -> tuple[bool, bool]:
    """Return strict Franka success and bounded physical-obstruction evidence.

    The printed shaft has repeatedly stopped the fingers at a real nonzero
    width while Franka's epsilon-based boolean remained false.  A width above
    the empty-Hand target is independent evidence that the shaft obstructs the
    closing fingers; the narrower bound also detects a subsequently dropped
    bolt because the Hand then closes to its target width.
    """

    strict = bool(accepted and is_grasped)
    bounded_obstruction = bool(
        float(grasp_config["physical_hold_width_min_m"])
        <= float(width_m)
        <= float(grasp_config["physical_hold_width_max_m"])
    )
    return strict, bounded_obstruction


def completed_two_x_episode_count(output_root: Path) -> int:
    """Count completed guarded 2x chains, never pre-motion/partial failures."""

    completed = 0
    for metadata_path in output_root.glob("*/metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            metadata.get("two_x_profile_enabled") is True
            and metadata.get("guarded_sequence_completed") is True
        ):
            completed += 1
    return completed


def load_inputs(recording_config: Path) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    collection = json.loads(recording_config.expanduser().resolve().read_text())
    task = collection["tasks"]["bolt_insert_engage"]
    preinsert_root = json.loads(project_path(task["preinsert_config"]).read_text())
    drop_root = json.loads(project_path(task["drop_regrasp_config"]).read_text())
    engagement = json.loads(project_path(task["engagement_config"]).read_text())
    commissioning = json.loads(project_path(task["commissioning_record"]).read_text())
    preinsert_episode = preinsert_root["replay"]["object_preinsert_execution"]
    drop_episode = drop_root["replay"]["object_normalization_execution"]
    preinsert_dir = project_path(task["preinsert_trajectory_dir"])
    drop_dir = project_path(task["drop_regrasp_trajectory_dir"])
    preinsert_plans = {
        name: build_bolt_preinsert_stage(
            str(preinsert_dir),
            name,
            command_rate_hz=float(preinsert_root["replay"]["command_rate_hz"]),
        )
        for name in BOLT_PREINSERT_STAGE_EVENTS
    }
    trims = drop_root["replay"]["stage_stationary_trim_s"]
    drop_plans = {
        name: build_bolt_drop_regrasp_stage(
            str(drop_dir),
            name,
            command_rate_hz=float(drop_root["replay"]["command_rate_hz"]),
            trim_start_s=float(trims[name]["start_s"]),
            trim_end_s=float(trims[name]["end_s"]),
        )
        for name in BOLT_DROP_REGRASP_STAGE_EVENTS
    }
    return (
        collection,
        task,
        preinsert_episode,
        drop_episode,
        engagement,
        commissioning,
        {"preinsert": preinsert_plans, "drop": drop_plans},
    )


def load_bracket_inputs(collection: dict) -> tuple[dict, dict, dict]:
    task = collection["tasks"]["bracket_placement"]
    root = json.loads(project_path(task["commissioned_replay_config"]).read_text())
    episode = root["replay"]["object_episode_execution"]
    trajectory_dir = project_path(task["trajectory_dir"])
    plans = {
        name: build_replay_stage(
            trajectory_dir,
            name,
            command_rate_hz=float(root["replay"]["command_rate_hz"]),
        )
        for name in REQUIRED_STAGES
    }
    return task, episode, plans


def resolve_success_endpoint(
    *,
    bracket_task: dict | None,
    full_task: dict | None,
    task: dict,
    collection: dict,
    bracket_only: bool,
    full_assembly: bool,
) -> str:
    """Resolve metadata endpoint across canonical and generated task configs."""
    if bracket_only:
        configured = (bracket_task or {}).get("success_endpoint")
        if configured:
            return str(configured)
        # Generated per-cell manual configs intentionally carry only motion
        # inputs; the canonical task config remains the source of prompt text.
        return str(collection["tasks"]["bracket_placement"]["success_endpoint"])
    if full_assembly:
        if full_task is None:
            raise KeyError("full-assembly task configuration is unavailable")
        return str(full_task["success_endpoint"])
    return str(task["success_endpoint"])


def select_stage_arm_config(
    stage_key: str,
    bracket_arm_config: dict | None,
    production_config: dict | None,
    *,
    accelerated: bool,
) -> dict | None:
    """Keep cell-specific path inputs separate from the selected speed profile."""
    if stage_key.startswith("bracket_") and bracket_arm_config is not None:
        if accelerated and production_config is not None:
            return production_config
        return bracket_arm_config
    return production_config


def build_cap_limited_accelerated_trajectory(plan, config: dict, stage_name: str):
    """Use the selected profile's speed as a floor, slowing only to meet its caps."""
    scales = dict(config.get("playback_time_scale_by_stage", {}))
    base_scale = float(scales.get(stage_name, config["playback_time_scale"]))
    utilization = float(config.get("planning_cap_utilization", 1.0))
    limits = (
        ("peak_abs_velocity_rad_s", float(config["velocity_cap_rad_s"])),
        ("peak_abs_acceleration_rad_s2", float(config["acceleration_cap_rad_s2"])),
        ("peak_abs_jerk_rad_s3", float(config["jerk_cap_rad_s3"])),
    )

    def make(scale: float):
        trial_scales = dict(scales)
        trial_scales[stage_name] = scale
        trial_config = {**config, "playback_time_scale_by_stage": trial_scales}
        return build_configured_accelerated_trajectory(plan, trial_config, stage_name)

    def within_limits(trajectory) -> bool:
        return all(
            float(getattr(trajectory, metric)) <= cap * utilization
            for metric, cap in limits
        )

    trajectory = make(base_scale)
    if within_limits(trajectory):
        return trajectory

    low = base_scale
    high = base_scale
    for _ in range(32):
        high *= 1.1
        trajectory = make(high)
        if within_limits(trajectory):
            break
        low = high
    else:
        raise ValueError(f"{stage_name}: could not retime within the configured motion caps")

    for _ in range(28):
        mid = (low + high) / 2.0
        candidate = make(mid)
        if within_limits(candidate):
            high = mid
            trajectory = candidate
        else:
            low = mid
    return make(high)


DEFAULT_MANUAL_BRACKET_CANDIDATE = (
    FR3_ROOT / "recordings" / "bracket_manual_replay_ik_preview_v1"
    / "20260816_145621_bracket_manual_exact_pose_rebalanced_ik_candidate.csv"
)
DEFAULT_MANUAL_BRACKET_CONFIG = (
    FR3_ROOT / "configs" / "bracket_manual_replay_2x_commissioning_v1.json"
)


def load_manual_bracket_inputs(
    candidate_path: Path = DEFAULT_MANUAL_BRACKET_CANDIDATE,
    bracket_config_path: Path = DEFAULT_MANUAL_BRACKET_CONFIG,
) -> tuple[dict, dict, dict]:
    """Same (task, episode, plans) shape as load_bracket_inputs(), built from
    the 2026-08-16 manual (COMMISSIONED wrist-side) bracket candidate instead
    of the original piloted trajectory. Reuses build_manual_stage_plan() so
    the plan objects are the same ReplayStagePlan type the rest of this file
    already expects."""
    from prepare_manual_bracket_replay import build_manual_stage_plan  # noqa: PLC0415

    bracket_config = json.loads(bracket_config_path.read_text())
    plans = {
        name: build_manual_stage_plan(
            candidate_path, name, command_rate_hz=float(bracket_config["command_rate_hz"])
        )
        for name in REQUIRED_STAGES
    }
    stage_bounds = [
        bracket_config["stages"][name]["workspace_bounds_m"] for name in REQUIRED_STAGES
    ]
    workspace_min = np.min([b["min_xyz"] for b in stage_bounds], axis=0).tolist()
    workspace_max = np.max([b["max_xyz"] for b in stage_bounds], axis=0).tolist()
    gripper = dict(bracket_config["gripper"])
    trajectory_name = next(iter(plans.values())).trajectory_name
    episode = {
        "enabled": True,
        "trajectory": trajectory_name,
        # command_velocity_source/maximum_*_rad_s below only gate the legacy
        # non-accelerated run_stage() fallback, which --four-x never takes
        # (it always uses run_retimed_2x_stage's own caps instead); they are
        # still evaluated as a static blocker check regardless of speed mode,
        # so they're derived generously above this bracket_config's own
        # velocity_cap_rad_s (the caps its binary is actually commissioned
        # for) rather than a number hardcoded for a specific past candidate --
        # B3's raised 1.9 rad/s cap blocked here on 2026-08-17 when this was
        # still a flat 1.5 tuned for B1/B2's ~1.08 rad/s peak.
        "playback_time_scale": bracket_config["playback_time_scale"],
        "stage_robot_global_dynamics_factor": 1.0,
        "relative_dynamics_factor": 0.3,
        "joint_margin_rad": 0.12,
        "start_joint_tolerance_rad": bracket_config["controller_start_tolerance_rad"],
        "terminal_joint_tolerance_rad": 0.02,
        "command_velocity_source": "recorded_scaled",
        "maximum_scaled_target_velocity_rad_s": max(
            1.5, 1.1 * float(bracket_config.get("velocity_cap_rad_s", 1.2))
        ),
        "maximum_command_velocity_jump_rad_s": max(
            0.5, 0.35 * float(bracket_config.get("velocity_cap_rad_s", 1.2))
        ),
        "maximum_target_step_rad": 0.5,
        "command_velocity_smoothing_passes": 6,
        "stage_lead_in_enabled": False,
        "stage_lead_in_ms": 750,
        "stage_lead_in_hold_ms": 250,
        "workspace_bounds_m": {"min_xyz": workspace_min, "max_xyz": workspace_max},
        "maximum_fk_path_error_m": 0.005,
        "state_timeout_s": 140.0,
        "camera_recording_enabled": False,
        "automatic_reset_to_start": {
            "enabled": True,
            "relative_dynamics_factor": bracket_config["reset_relative_dynamics_factor"],
            "joint_margin_rad": bracket_config.get("reset_joint_margin_rad", 0.12),
            "maximum_joint_delta_rad": bracket_config["reset_maximum_joint_delta_rad"],
            "workspace_bounds_m": bracket_config["reset_workspace_bounds_m"],
            "target_fk_error_limit_m": 0.005,
            "terminal_joint_tolerance_rad": bracket_config["reset_terminal_tolerance_rad"],
            "requires_empty_gripper": True,
            "requires_object_removed": True,
        },
        "gripper": gripper,
    }
    task = {
        "trajectory_dir": str(candidate_path),
        "candidate_source": "manual_commissioned_2026_08_16",
    }
    return task, episode, plans


def static_blockers(
    collection: dict,
    task: dict,
    preinsert: dict,
    drop: dict,
    engagement: dict,
    commissioning: dict,
    plans: dict,
) -> list[str]:
    blockers: list[str] = []
    roles = [entry.get("role") for entry in collection["camera"]["serials_in_dataset_order"]]
    if collection.get("schema_version") != 1:
        blockers.append("autonomous collection schema_version must be 1")
    if roles != ["exterior", "wrist"]:
        blockers.append("camera order must be exactly exterior, wrist")
    if collection["dataset"].get("schema") != "droid_joint_position":
        blockers.append("production bolt demos require droid_joint_position")
    pickup_grasp = task["pickup_shaft_grasp"]
    if not (
        float(pickup_grasp["accepted_width_min_m"])
        < float(pickup_grasp["target_width_m"])
        < float(pickup_grasp["accepted_width_max_m"])
    ):
        blockers.append("pickup target width is outside its accepted shaft-width band")
    head_grasp = task["thread_ready_head_grasp"]
    head_calibration = json.loads(project_path(head_grasp["calibration"]).read_text())
    if head_calibration.get("all_samples_is_grasped") is not True:
        blockers.append("thread-ready head calibration did not verify every sample")
    if abs(
        float(head_grasp["target_width_m"])
        - float(head_calibration["median_width_m"])
    ) > 0.000001:
        blockers.append("head grasp target does not match its passed calibration")
    if task.get("automatic_retreat_enabled") is not False:
        blockers.append("uncommissioned automatic retreat must remain disabled")
    if task.get("operator_final_engagement_confirmation_required") is not True:
        blockers.append("final visible engagement confirmation must remain required")
    if preinsert.get("enabled") is not True or drop.get("enabled") is not True:
        blockers.append("both commissioned object replay stages must remain enabled")
    if preinsert.get("contact_threading_enabled") is not False:
        blockers.append("preinsert replay must not contain contact/threading")
    if drop.get("contact_threading_enabled") is not False:
        blockers.append("drop/regrasp replay must not contain contact/threading")
    if any(p.trajectory_name != preinsert.get("trajectory") for p in plans["preinsert"].values()):
        blockers.append("preinsert trajectory does not match its commissioned profile")
    if any(p.trajectory_name != drop.get("trajectory") for p in plans["drop"].values()):
        blockers.append("drop/regrasp trajectory does not match its commissioned profile")
    if commissioning.get("commissioning_status") != "PASSED" or commissioning.get("objective_achieved") is not True:
        blockers.append("bolt threading commissioning record is not passed")
    evidence = commissioning.get("automatic_engagement_evidence", {})
    if evidence.get("stop_reason") != "motion_completed":
        blockers.append("commissioning lacks a completed automatic engagement trial")
    if evidence.get("operator_physically_confirmed_thread_engagement_after_opening_hand") is not True:
        blockers.append("commissioning lacks physical engagement confirmation")
    sequence = task["engagement_drive_sequence"]
    angles = [float(stage["angle_deg"]) for stage in sequence]
    if angles != [85.0, 85.0, 85.0, 50.0, 50.0, 50.0]:
        blockers.append(
            "setup engagement sequence must be 85+85+85+50+50+50 degrees"
        )
    if abs(sum(angles) - float(task["total_engagement_rotation_deg"])) > 1e-12:
        blockers.append("engagement drive angles do not sum to the configured total")
    if sequence[0].get("empty_wrist_ratchet_before") is True or any(
        stage.get("empty_wrist_ratchet_before") is not True
        for stage in sequence[1:]
    ):
        blockers.append("every tightening window after the first must begin with an empty-wrist ratchet")
    expected_binaries = {
        45.0: {"fr3_bolt_combined_thread_45"},
        50.0: {"fr3_bolt_combined_thread_50", "fr3_bolt_combined_thread_50_widelateral"},
        85.0: {"fr3_bolt_combined_thread_85", "fr3_bolt_combined_thread_85_widelateral"},
        40.0: {"fr3_bolt_combined_thread_40"},
        5.0: {"fr3_bolt_combined_thread_5"},
    }
    for stage in sequence:
        binary = project_path(stage["binary"])
        if not binary.is_file():
            blockers.append(f"compiled controller is missing for {stage['name']}")
        elif binary.name not in expected_binaries.get(float(stage["angle_deg"]), set()):
            blockers.append(f"controller binary does not match angle for {stage['name']}")
    q7_start = float(task["engagement_start_q7_rad"])
    dynamic = json.loads(project_path(task["dynamic_model_calibration"]).read_text())
    dynamic_min = float(dynamic["knots"][0][0]) - 0.02
    dynamic_max = float(dynamic["knots"][-1][0])
    planned_q7 = q7_start
    for stage in sequence:
        if stage.get("empty_wrist_ratchet_before"):
            planned_q7 = q7_start
        stage_stop = planned_q7 + math.radians(float(stage["angle_deg"]))
        # Reserve the worst-case angular distance added while the fastest
        # compiled setup controller slews a guard stop from 0.120 rad/s to zero.
        maximum_guard_stop_extra_rad = 0.12**2 / (2.0 * 0.5)
        if (
            planned_q7 < dynamic_min
            or stage_stop + maximum_guard_stop_extra_rad > dynamic_max
        ):
            blockers.append(f"planned q7 interval leaves the dynamic model for {stage['name']}")
        planned_q7 = stage_stop
    ratchet = commissioning.get("automatic_ratchet_evidence", {})
    if ratchet.get("accepted_by_operator") is not True:
        blockers.append("automatic empty-wrist reset has not been physically accepted")
    if engagement.get("collision_reflex_settings") != "UNCHANGED":
        blockers.append("Franka collision/reflex settings must remain unchanged")
    if not project_path(task["dynamic_model_calibration"]).is_file():
        blockers.append("dynamic wrist-torque model is missing")
    return blockers


def make_waypoint_motion(plan, episode: dict):
    from franky import Duration, JointState, JointWaypoint, JointWaypointMotion  # noqa: PLC0415

    scale = float(episode["playback_time_scale"])
    waypoint_ms = int(math.ceil(1000.0 * scale / plan.command_rate_hz))
    waypoints = [
        JointWaypoint(
            JointState(q, dq / scale),
            minimum_time=Duration(waypoint_ms),
            max_total_duration=Duration(2000),
        )
        for q, dq in zip(plan.q_targets[1:], plan.dq_targets[1:], strict=True)
    ]
    return JointWaypointMotion(
        waypoints,
        relative_dynamics_factor=float(episode["relative_dynamics_factor"]),
    )


def dynamic_model_csv(path: Path, source: Path) -> None:
    document = json.loads(source.read_text())
    knots = np.asarray(document["knots"], dtype=np.float64)
    np.savetxt(path, knots[:, :2], delimiter=",", header="q7_rad,median_K_Tz_Nm", comments="")


def direct_samples(
    csv_path: Path,
    width: float,
    is_grasped: bool,
    command_width: float,
) -> list[dict]:
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "wall_time_ns" not in rows[0]:
        raise RuntimeError("direct engagement telemetry lacks absolute timestamps")
    # The controller logs at 1 kHz. Thirty-hertz labels are sufficient for the
    # 30-fps raw video while retaining the final controller state exactly.
    selected = rows[::33]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    samples = []
    for row in selected:
        samples.append(
            {
                "time_ns": int(row["wall_time_ns"]),
                "robot_time": float(row["time_s"]),
                "q": [float(row[f"q{i}"]) for i in range(7)],
                "dq": [float(row[f"dq{i}"]) for i in range(7)],
                "ee_translation": [float(row[name]) for name in ("x", "y", "z")],
                "gripper_state_time_ns": int(row["wall_time_ns"]),
                "gripper_width": float(width),
                "gripper_is_grasped": bool(is_grasped),
                "gripper_command_width": float(command_width),
                "state_source": "direct_libfranka_engagement_telemetry",
            }
        )
    return samples


def retimed_direct_samples(
    csv_path: Path,
    width: float,
    is_grasped: bool,
    command_width: float,
    profile_label: str = "2x",
) -> list[dict]:
    """Convert 1 kHz smooth-arm telemetry into recorder state samples."""

    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "wall_time_ns" not in rows[0]:
        raise RuntimeError("retimed arm telemetry lacks absolute timestamps")
    selected = rows[::33]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return [
        {
            "time_ns": int(row["wall_time_ns"]),
            "robot_time": float(row["time_s"]),
            "q": [float(row[f"q{i}"]) for i in range(7)],
            "dq": [float(row[f"dq{i}"]) for i in range(7)],
            "ee_translation": [float(row[name]) for name in ("x", "y", "z")],
            "gripper_state_time_ns": int(row["wall_time_ns"]),
            "gripper_width": float(width),
            "gripper_is_grasped": bool(is_grasped),
            "gripper_command_width": float(command_width),
            "state_source": f"direct_libfranka_retimed_{profile_label}_telemetry",
        }
        for row in selected
    ]


def stationary_state_bridge(
    source_sample: dict,
    stop_time_ns: int,
    sample_hz: float,
    width: float,
    is_grasped: bool,
    command_width: float,
) -> list[dict]:
    """Fill an exclusive-controller connection handoff with stationary states.

    The direct controller spends three seconds acquiring a baseline before its
    1 kHz control telemetry begins. The arm and Hand are deliberately
    stationary during that gap, so repeated achieved states prevent camera
    frames from being paired across a multi-second timestamp hole.
    """
    period_ns = int(round(1.0e9 / float(sample_hz)))
    start_ns = int(source_sample["time_ns"]) + period_ns
    if stop_time_ns <= start_ns:
        return []
    bridge: list[dict] = []
    for time_ns in range(start_ns, int(stop_time_ns), period_ns):
        sample = dict(source_sample)
        sample["time_ns"] = time_ns
        sample["dq"] = [0.0] * 7
        sample["dq_d"] = [0.0] * 7
        sample["ddq_d"] = [0.0] * 7
        sample["gripper_state_time_ns"] = time_ns
        sample["gripper_width"] = float(width)
        sample["gripper_is_grasped"] = bool(is_grasped)
        sample["gripper_command_width"] = float(command_width)
        sample["state_source"] = "stationary_direct_controller_handoff_bridge"
        bridge.append(sample)
    return bridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--recording-preview", action="store_true")
    mode.add_argument("--record-demo", action="store_true")
    parser.add_argument(
        "--full-assembly",
        action="store_true",
        help="record bracket placement and bolt engagement as one continuous episode",
    )
    parser.add_argument(
        "--bracket-only",
        action="store_true",
        help="run and record only the commissioned bracket pick/place stages; requires --full-assembly",
    )
    parser.add_argument(
        "--training-data",
        action="store_true",
        help=(
            "make an accepted full-assembly capture eligible for derived bracket, "
            "full-policy bolt, and hybrid-prefix training views"
        ),
    )
    speed = parser.add_mutually_exclusive_group()
    speed.add_argument(
        "--two-x",
        action="store_true",
        help="use the separately commissioned smooth 2x arm and 0.060 rad/s threading profile",
    )
    speed.add_argument(
        "--three-x",
        action="store_true",
        help="target 3x production speed, cap-limiting individual arm stages and using 0.090 rad/s threading",
    )
    speed.add_argument(
        "--four-x",
        action="store_true",
        help="target recorded source speed, cap-limiting individual arm stages and using 0.120 rad/s threading",
    )
    parser.add_argument(
        "--threading-speed-override",
        choices=["2x", "3x", "4x"],
        default=None,
        help=(
            "run only the threading drive chain at this speed profile, independent of the "
            "--two-x/--three-x/--four-x profile selected above for the bracket/bolt arm stages. "
            "The underlying per-angle threading binaries and force/admittance targets are "
            "identical across speeds -- only the commanded rotation rate changes -- and each "
            "speed's threading chain has its own separate commissioning gate "
            "(threading_2x/3x/4x.combined_production_chain_enabled), so this does not bypass "
            "or weaken any guard, it only decouples which already-commissioned threading speed "
            "is used from which already-commissioned arm speed is used. Requires one of "
            "--two-x/--three-x/--four-x to also be passed."
        ),
    )
    parser.add_argument(
        "--bracket-source",
        choices=["piloted", "manual_2026_08_16"],
        default="piloted",
        help=(
            "which bracket trajectory source to use for --full-assembly: 'piloted' is the "
            "original commissioned source; 'manual_2026_08_16' is the new manual (COMMISSIONED "
            "wrist-side) candidate placed and proven live on 2026-08-16"
        ),
    )
    parser.add_argument(
        "--bracket-arm-config",
        type=Path,
        default=None,
        help=(
            "override just the bracket_* stages' retiming config/binary, independent of "
            "whichever --two-x/--three-x/--four-x profile governs bolt_* stages. Used for the "
            "2026-08-16 bracket-only high-cap (true 1.0x speed) binary, which has not been "
            "validated for bolt stages and must not silently apply to them."
        ),
    )
    parser.add_argument(
        "--bracket-slot",
        default=None,
        help=(
            "bracket slot label recorded in the episode folder name and metadata. Normally "
            "auto-derived from the 'bracket_slot' field of --bracket-arm-config (or of the "
            "default manual bracket config when --bracket-source manual_2026_08_16 is used "
            "without an override); pass this explicitly only to force a mismatch check to fail "
            "loudly instead of silently trusting the config. Previously hardcoded to 'B1' "
            "regardless of which bracket source/config was actually used, which mislabeled "
            "B2 recordings on 2026-08-16 (episode 2) -- auto-derivation exists so that bug "
            "class cannot recur just because an operator forgot to type a flag."
        ),
    )
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recording-config", type=Path, default=DEFAULT_RECORDING_CONFIG)
    args = parser.parse_args()
    if args.bracket_only and not args.full_assembly:
        raise SystemExit("--bracket-only requires --full-assembly so the commissioned bracket path is loaded")
    if args.episode_index < 1:
        raise SystemExit("--episode-index must be at least 1")
    if args.training_data and not (args.record_demo and args.full_assembly and args.four_x):
        raise SystemExit(
            "--training-data requires --record-demo --full-assembly --four-x; "
            "the separately passed 4x setup is the commissioned collection path"
        )
    accelerated = bool(args.two_x or args.three_x or args.four_x)
    speed_label = (
        "4x" if args.four_x else "3x" if args.three_x
        else "2x" if args.two_x else "1x"
    )
    if accelerated and not args.full_assembly:
        flag = (
            "--four-x" if args.four_x else "--three-x"
            if args.three_x else "--two-x"
        )
        raise SystemExit(f"{flag} is only defined for the continuous --full-assembly flow")
    if args.threading_speed_override is not None and not accelerated:
        raise SystemExit(
            "--threading-speed-override requires --two-x, --three-x, or --four-x to also "
            "select the arm/bolt profile it runs alongside"
        )
    threading_speed_label = args.threading_speed_override or speed_label

    bracket_arm_config = None
    if args.bracket_arm_config is not None:
        if args.bracket_source != "manual_2026_08_16":
            raise SystemExit("--bracket-arm-config requires --bracket-source manual_2026_08_16")
        bracket_arm_config = json.loads(args.bracket_arm_config.expanduser().resolve().read_text())

    if bracket_arm_config is not None:
        derived_bracket_slot = bracket_arm_config.get("bracket_slot")
        if derived_bracket_slot is None:
            raise SystemExit(
                f"--bracket-arm-config {args.bracket_arm_config} is missing a 'bracket_slot' "
                "field -- add one (e.g. \"bracket_slot\": \"B2\") so the episode folder name "
                "and metadata cannot silently default to the wrong slot"
            )
    elif args.bracket_source == "manual_2026_08_16":
        derived_bracket_slot = json.loads(DEFAULT_MANUAL_BRACKET_CONFIG.read_text()).get(
            "bracket_slot", "B1"
        )
    else:
        derived_bracket_slot = "B1"
    if args.bracket_slot is not None and args.bracket_slot != derived_bracket_slot:
        raise SystemExit(
            f"--bracket-slot {args.bracket_slot} was passed explicitly but does not match the "
            f"bracket_slot '{derived_bracket_slot}' declared by the actual bracket config in "
            "use -- this mismatch is exactly the class of bug that mislabeled episode 2 on "
            "2026-08-16; fix the config's bracket_slot field or drop --bracket-slot and let it "
            "auto-derive"
        )
    args.bracket_slot = derived_bracket_slot

    def arm_config_for(stage_key: str) -> dict:
        """Use the selected cap-aware profile for accelerated bracket and bolt stages."""
        selected = select_stage_arm_config(
            stage_key,
            bracket_arm_config,
            two_x_config,
            accelerated=accelerated,
        )
        if selected is None:
            raise RuntimeError(f"no arm profile is configured for {stage_key}")
        return selected

    def retime_stage(plan, stage_key: str):
        config = arm_config_for(stage_key)
        if accelerated and bracket_arm_config is not None and stage_key.startswith("bracket_"):
            return build_cap_limited_accelerated_trajectory(plan, config, stage_key)
        return build_configured_accelerated_trajectory(plan, config, stage_key)

    try:
        collection, task, preinsert, drop, engagement, commissioning, plans = load_inputs(
            args.recording_config
        )
        blockers = static_blockers(
            collection, task, preinsert, drop, engagement, commissioning, plans
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ReplayPlanError) as error:
        raise SystemExit(f"[bolt production] invalid input: {error}") from error
    full_task = collection["tasks"].get("full_bracket_bolt_assembly") if args.full_assembly else None
    # ``two_x_config`` is retained as the local variable name to keep the
    # recorder diff small; for --three-x it contains the cap-aware 3x profile.
    two_x_config = None
    bracket_task = None
    bracket_episode = None
    bracket_plans = None
    if args.full_assembly:
        try:
            bracket_task, bracket_episode, bracket_plans = (
                (
                    load_manual_bracket_inputs(
                        candidate_path=project_path(bracket_arm_config["candidate"]),
                        bracket_config_path=project_path(args.bracket_arm_config),
                    )
                    if bracket_arm_config is not None
                    else load_manual_bracket_inputs()
                )
                if args.bracket_source == "manual_2026_08_16"
                else load_bracket_inputs(collection)
            )
            bracket_blockers = object_episode_blockers(bracket_episode, bracket_plans)
            if accelerated:
                # maximum_scaled_target_velocity_rad_s / maximum_command_velocity_jump_rad_s
                # only gate the legacy non-accelerated run_stage() fallback (see the
                # comment on those fields in load_manual_bracket_inputs/load_bracket_inputs);
                # --two-x/--three-x/--four-x always take run_retimed_2x_stage's own
                # commissioned velocity/acceleration/jerk caps instead, which are
                # independently offline-validated per stage before execution_enabled is
                # ever set true. Raw command-velocity-jump on the unretimed recorded
                # samples is not representative of what actually gets commanded in this
                # mode, so it's not a real safety signal here and shouldn't block.
                bracket_blockers = [
                    item for item in bracket_blockers
                    if not item.startswith("scaled target velocity")
                    and not item.startswith("command velocity jump")
                ]
            blockers.extend(bracket_blockers)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ReplayPlanError) as error:
            raise SystemExit(f"[full assembly] invalid bracket input: {error}") from error
        if full_task is None:
            blockers.append("full-assembly task is missing from the recording config")
        elif args.record_demo and full_task.get("enabled") is not True:
            blockers.append("full-assembly recording remains disabled pending preview")
        if accelerated:
            profile_key = (
                "four_x_profile" if args.four_x else "three_x_profile"
                if args.three_x else "two_x_profile"
            )
            profile = full_task.get(profile_key, {}) if full_task is not None else {}
            if args.record_demo and profile.get("enabled") is not True:
                blockers.append(f"full-assembly {speed_label} profile remains disabled")
            try:
                two_x_config = json.loads(
                    project_path(profile["commissioning_config"]).read_text()
                )
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
                blockers.append(f"cannot load the {speed_label} production config: {error}")
            if bracket_arm_config is not None:
                if not project_path(bracket_arm_config["binary"]).is_file():
                    blockers.append("compiled bracket-arm-config controller is missing")
            if two_x_config is not None:
                if not project_path(two_x_config["binary"]).is_file():
                    blockers.append(f"compiled smooth {speed_label} arm controller is missing")
                for stage_name, stage_config in two_x_config["stages"].items():
                    source_plan = {
                        "bracket_approach_open": bracket_plans["approach_open"],
                        "bracket_transport_place_closed": bracket_plans["transport_place_closed"],
                        "bracket_retreat_open": bracket_plans["retreat_open"],
                        "bolt_approach_open": plans["preinsert"]["approach_open"],
                        "bolt_transport_preinsert_closed": plans["preinsert"]["transport_preinsert_closed"],
                        "bolt_lower_held": plans["drop"]["lower_held"],
                        "bolt_reposition_regrasp_open": plans["drop"]["reposition_regrasp_open"],
                    }[stage_name]
                    stage_arm_config = arm_config_for(stage_name)
                    stage_speed_label = (
                        "bracket-arm-config" if stage_arm_config is bracket_arm_config else speed_label
                    )
                    retimed = retime_stage(source_plan, stage_name)
                    for label, measured, cap in (
                        ("velocity", retimed.peak_abs_velocity_rad_s, stage_arm_config["velocity_cap_rad_s"]),
                        ("acceleration", retimed.peak_abs_acceleration_rad_s2, stage_arm_config["acceleration_cap_rad_s2"]),
                        ("jerk", retimed.peak_abs_jerk_rad_s3, stage_arm_config["jerk_cap_rad_s3"]),
                    ):
                        if measured > float(cap):
                            blockers.append(
                                f"{stage_speed_label} {stage_name} {label} exceeds its configured cap"
                            )
                    if stage_name.startswith("bracket_") and args.bracket_source == "manual_2026_08_16":
                        # Authorization for a manual-source bracket stage must come from that
                        # stage's OWN commissioning config (bracket_arm_config when set, else
                        # the default 2x manual config) -- never from two_x_config's stages,
                        # whose flags were earned commissioning the *old* piloted trajectory.
                        manual_bracket_config = (
                            bracket_arm_config
                            if bracket_arm_config is not None
                            else json.loads(DEFAULT_MANUAL_BRACKET_CONFIG.read_text())
                        )
                        manual_stage_name = stage_name.removeprefix("bracket_")
                        manual_stage = manual_bracket_config["stages"].get(manual_stage_name, {})
                        stage_trial_passed = manual_stage.get("execution_enabled") is True
                        if args.record_demo and not stage_trial_passed:
                            blockers.append(
                                f"{stage_name} has no live-proven authorization in "
                                f"{'bracket_arm_config' if bracket_arm_config is not None else 'the default manual bracket config'}"
                            )
                        continue
                    stage_trial_passed = stage_config.get(
                        "production_execution_authorized"
                    ) is True or (
                        stage_config.get("object_aware_physical_trial_passed") is True
                        if stage_name == "bolt_reposition_regrasp_open"
                        else stage_config.get("empty_hand_physical_trial_passed") is True
                    )
                    if args.record_demo and not stage_trial_passed:
                        blockers.append(f"{speed_label} {stage_name} execution remains locked")
                enabled_key = (
                    "final_full_assembly_4x_enabled" if args.four_x
                    else "final_full_assembly_3x_enabled" if args.three_x
                    else "final_full_assembly_2x_enabled"
                )
                threading_key = f"threading_{threading_speed_label}"
                if args.record_demo and two_x_config.get(enabled_key) is not True:
                    blockers.append(f"full-assembly {speed_label} production execution remains locked")
                # The threading commissioning gate (threading_2x/3x/4x) lives inside each
                # arm-speed-specific commissioning config. When threading is overridden to a
                # different speed than the arm/bolt profile, that gate has to be read from
                # the config file that actually belongs to the overridden threading speed,
                # not from two_x_config (which belongs to the arm/bolt speed).
                if threading_speed_label == speed_label:
                    threading_config = two_x_config
                else:
                    threading_profile_key = (
                        "four_x_profile" if threading_speed_label == "4x"
                        else "three_x_profile" if threading_speed_label == "3x"
                        else "two_x_profile"
                    )
                    threading_profile = full_task.get(threading_profile_key, {}) if full_task is not None else {}
                    if args.record_demo and threading_profile.get("enabled") is not True:
                        blockers.append(f"full-assembly {threading_speed_label} profile remains disabled")
                    try:
                        threading_config = json.loads(
                            project_path(threading_profile["commissioning_config"]).read_text()
                        )
                    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
                        threading_config = None
                        blockers.append(
                            f"cannot load the {threading_speed_label} threading config: {error}"
                        )
                if threading_config is not None and threading_key not in threading_config:
                    blockers.append(
                        f"{threading_speed_label} threading config is missing '{threading_key}'"
                    )
                elif (
                    args.record_demo and threading_config is not None
                    and threading_config[threading_key].get("combined_production_chain_enabled") is not True
                ):
                    blockers.append(
                        f"{float(threading_config[threading_key]['peak_angular_speed_rad_s']):.3f} "
                        "rad/s production threading chain remains locked"
                    )
            if full_task is not None and args.record_demo:
                maximum_trials = profile.get(
                    "maximum_supervised_production_trials"
                )
                if maximum_trials is not None:
                    output_root = project_path(full_task["output_dir"])
                    existing_two_x = completed_two_x_episode_count(output_root)
                    if existing_two_x >= int(maximum_trials):
                        blockers.append(
                            "authorized supervised full-assembly 2x trial allowance is exhausted"
                        )
    if args.record_demo and task.get("enabled") is not True:
        blockers.append("autonomous bolt recording remains disabled pending recording preview")

    active_engagement_sequence = [dict(stage) for stage in task["engagement_drive_sequence"]]
    if accelerated:
        suffix = threading_speed_label
        accelerated_binary_by_angle = {
            5.0: "model-running/streaming/build/fr3_bolt_combined_thread_5_2x",
            40.0: "model-running/streaming/build/fr3_bolt_combined_thread_40_2x",
            45.0: "model-running/streaming/build/fr3_bolt_combined_thread_45_2x",
            50.0: "model-running/streaming/build/fr3_bolt_combined_thread_50_2x",
            85.0: "model-running/streaming/build/fr3_bolt_combined_thread_85_2x",
        }
        if threading_speed_label != "2x":
            accelerated_binary_by_angle = {
                angle: path.replace("_2x", f"_{threading_speed_label}")
                for angle, path in accelerated_binary_by_angle.items()
            }
        for stage in active_engagement_sequence:
            stage["binary"] = accelerated_binary_by_angle[float(stage["angle_deg"])]
            if not project_path(stage["binary"]).is_file():
                blockers.append(
                    f"compiled {suffix} thread controller is missing for {stage['name']}"
                )

    threading_speed = (
        0.12 if threading_speed_label == "4x" else 0.09 if threading_speed_label == "3x"
        else 0.06 if threading_speed_label == "2x"
        else float(engagement["peak_angular_speed_rad_s"])
    )
    if accelerated and two_x_config is not None:
        expected_duration = sum(
            build_configured_accelerated_trajectory(
                plan,
                two_x_config,
                {
                    "approach_open": "bolt_approach_open",
                    "transport_preinsert_closed": "bolt_transport_preinsert_closed",
                }[name],
            ).duration_s
            for name, plan in plans["preinsert"].items()
        ) + sum(
            build_configured_accelerated_trajectory(
                plan,
                two_x_config,
                {
                    "lower_held": "bolt_lower_held",
                    "reposition_regrasp_open": "bolt_reposition_regrasp_open",
                }[name],
            ).duration_s
            for name, plan in plans["drop"].items()
        )
    else:
        expected_duration = sum(
            plan.source_duration_s * float(preinsert["playback_time_scale"])
            for plan in plans["preinsert"].values()
        ) + sum(
            plan.source_duration_s * float(drop["playback_time_scale"])
            for plan in plans["drop"].values()
        )
    expected_duration += sum(
        float(engagement["force_acquisition_duration_s"])
        + math.pi * math.radians(float(stage["angle_deg"]))
        / (2.0 * threading_speed)
        for stage in active_engagement_sequence
    )
    if args.full_assembly:
        assert bracket_episode is not None and bracket_plans is not None
        if accelerated and two_x_config is not None:
            expected_duration += sum(
            retime_stage(
                plan,
                {
                    "approach_open": "bracket_approach_open",
                    "transport_place_closed": "bracket_transport_place_closed",
                    "retreat_open": "bracket_retreat_open",
                }[name],
                ).duration_s
                for name, plan in bracket_plans.items()
            )
        else:
            expected_duration += sum(
                plan.source_duration_s * float(bracket_episode["playback_time_scale"])
                for plan in bracket_plans.values()
            )
    if args.bracket_only:
        assert bracket_episode is not None and bracket_plans is not None
        expected_duration = (
            sum(
                retime_stage(
                    plan,
                    {
                        "approach_open": "bracket_approach_open",
                        "transport_place_closed": "bracket_transport_place_closed",
                        "retreat_open": "bracket_retreat_open",
                    }[name],
                ).duration_s
                for name, plan in bracket_plans.items()
            )
            if accelerated and two_x_config is not None else
            sum(
                plan.source_duration_s * float(bracket_episode["playback_time_scale"])
                for plan in bracket_plans.values()
            )
        )
    print(
        "=== AUTONOMOUS BRACKET PLACEMENT DEMO ==="
        if args.bracket_only else
        "=== AUTONOMOUS FULL BRACKET + BOLT ASSEMBLY DEMO ==="
        if args.full_assembly
        else "=== AUTONOMOUS BOLT INSERT + ENGAGEMENT DEMO ==="
    )
    print(f"mode:                     {'RECORDING_PREVIEW' if args.recording_preview else 'RECORD_DEMO'}")
    print(f"robot IP:                 {collection['robot_ip']}")
    active_prompt = (
        collection["tasks"]["bracket_placement"]["language_prompt"]
        if args.bracket_only else
        full_task["language_prompt"] if args.full_assembly else task["language_prompt"]
    )
    print(f"task:                     {active_prompt}")
    drive_tokens: list[str] = []
    for index, stage in enumerate(active_engagement_sequence):
        if index and stage.get("empty_wrist_ratchet_before"):
            drive_tokens.append("ratchet")
        drive_tokens.append(f"{float(stage['angle_deg']):.0f}")
    drive_chain = " -> ".join(drive_tokens)
    prefix = (
        "bracket pickup/place -> cross-task transition -> bolt pickup/preinsert"
        if args.full_assembly else "pickup -> preinsert"
    )
    print(
        "sequence:                 bracket approach -> grasp -> wall placement -> release -> retreat"
        if args.bracket_only else
        f"sequence:                 {prefix} -> stand/regrasp -> {drive_chain} -> release"
    )
    print(f"expected commanded time:  {expected_duration:.1f} s plus short verification holds")
    print("camera recording:         exterior + wrist, continuous across all stages")
    print("dataset schema:           15 Hz droid_joint_position")
    if args.bracket_only:
        print("data purpose:             BRACKET PLACEMENT TRAINING CANDIDATE")
    elif args.full_assembly:
        print(
            "data purpose:             TRAINING CANDIDATE; final visual acceptance required"
            if args.training_data
            else "data purpose:             SETUP VALIDATION ONLY; not training-eligible"
        )
    print(
        "success endpoint:         bracket mounted on pins; Hand open; arm retreated"
        if args.bracket_only else
        "success endpoint:         visibly engaged/self-supported; Hand open; arm stationary"
    )
    if not args.bracket_only:
        print(f"empty-wrist prewind:      q7 -> {float(task['engagement_start_q7_rad']):+.6f} rad before head grasp")
        print(f"total tightening rotation:{float(task['total_engagement_rotation_deg']):.1f} deg")
        print(
            "threading axial load:     +1.200360 N relative target via admittance; "
            "0.500 mm/s cap"
        )
    if not args.bracket_only:
        planned_q7 = float(task["engagement_start_q7_rad"])
        for stage in active_engagement_sequence:
            if stage.get("empty_wrist_ratchet_before"):
                planned_q7 = float(task["engagement_start_q7_rad"])
            stop_q7 = planned_q7 + math.radians(float(stage["angle_deg"]))
            print(
                f"  {stage['name']}: {planned_q7:+.3f} -> {stop_q7:+.3f} rad "
                f"(+{float(stage['angle_deg']):.0f} deg)"
            )
            planned_q7 = stop_q7
    print("automatic retreat:        DISABLED (manual only after recording stops)")
    print("collision/reflex settings:UNCHANGED")
    if args.full_assembly:
        if accelerated:
            requested = 4.0 if args.four_x else 3.0 if args.three_x else 2.0
            print(
                f"arm replay speed:          up to {requested:.1f}x current, via direct "
                "1 kHz clamped-cubic controller"
            )
            if args.three_x or args.four_x:
                print("arm cap policy:            per-stage limited to unchanged 1.2 / 4.0 / 40.0 caps")
            if bracket_arm_config is not None:
                print(
                    f"bracket path source:       {Path(bracket_arm_config['candidate']).name}; "
                    f"timing/controller use the cap-aware {speed_label} profile"
                )
            print(
                f"threading peak speed:      {threading_speed:.3f} rad/s "
                f"(separately compiled {threading_speed_label} binaries"
                f"{', overriding the ' + speed_label + ' arm/bolt profile' if threading_speed_label != speed_label else ''})"
            )
            enabled_key = (
                "final_full_assembly_4x_enabled" if args.four_x
                else "final_full_assembly_3x_enabled" if args.three_x
                else "final_full_assembly_2x_enabled"
            )
            print(
                f"{speed_label} production enabled:    "
                f"{bool(two_x_config and two_x_config[enabled_key])}"
            )
        else:
            print(
                f"arm replay speed:          {float(full_task['arm_playback_speed_multiplier']):.1f}x "
                "the physically passed replay"
            )
            print("2x profile:                available behind --two-x; commissioning remains separate")
    if blockers:
        print("\nStatic blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
        if args.record_demo:
            raise SystemExit("[SAFETY] autonomous bolt recording refused")

    if args.full_assembly and args.record_demo:
        print("Connecting read-only and validating the complete commissioned path...")
    else:
        input(
            "Press Enter to connect read-only and validate the complete commissioned path "
            "(Ctrl+C cancels): "
        )
    from franky import Frame, Gripper, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(collection["robot_ip"])
    gripper = Gripper(collection["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    if args.full_assembly:
        assert bracket_plans is not None
        target_q = bracket_plans["approach_open"].expected_start_q
        target_xyz = bracket_plans["approach_open"].xyz_targets[0]
    else:
        target_q = plans["preinsert"]["approach_open"].expected_start_q
        target_xyz = plans["preinsert"]["approach_open"].xyz_targets[0]
    start_error = float(np.max(np.abs(current_q - target_q)))
    initial_width = float(gripper.state.width)
    initial_grasped = bool(gripper.is_grasped)
    runtime_blockers = list(blockers)
    if initial_grasped:
        runtime_blockers.append("Hand reports a held object; remove it before reset")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    all_plans = list(plans["preinsert"].values()) + list(plans["drop"].values())
    if args.full_assembly:
        assert bracket_plans is not None
        all_plans = list(bracket_plans.values()) + all_plans
    all_q = np.concatenate([plan.q_targets for plan in all_plans])
    predicted_xyz = np.asarray(
        [
            (
                np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
                @ joint7_to_tcp
            )[:3, 3]
            for q in all_q
        ]
    )
    margin = max(float(preinsert["joint_margin_rad"]), float(drop["joint_margin_rad"]))
    if np.any(all_q < JOINT_MIN + margin) or np.any(all_q > JOINT_MAX - margin):
        runtime_blockers.append("combined replay violates guarded joint ranges")
    bracket_count = (
        sum(plan.q_targets.shape[0] for plan in bracket_plans.values())
        if args.full_assembly
        else 0
    )
    pre_count = sum(plan.q_targets.shape[0] for plan in plans["preinsert"].values())
    bracket_xyz = predicted_xyz[:bracket_count]
    pre_xyz = predicted_xyz[bracket_count : bracket_count + pre_count]
    drop_xyz = predicted_xyz[bracket_count + pre_count :]
    if args.full_assembly:
        assert bracket_episode is not None
        minimum = np.asarray(bracket_episode["workspace_bounds_m"]["min_xyz"])
        maximum = np.asarray(bracket_episode["workspace_bounds_m"]["max_xyz"])
        if np.any(bracket_xyz < minimum) or np.any(bracket_xyz > maximum):
            runtime_blockers.append("bracket path leaves its commissioned workspace")
    for name, xyz, episode in (("preinsert", pre_xyz, preinsert), ("drop/regrasp", drop_xyz, drop)):
        minimum = np.asarray(episode["workspace_bounds_m"]["min_xyz"])
        maximum = np.asarray(episode["workspace_bounds_m"]["max_xyz"])
        if np.any(xyz < minimum) or np.any(xyz > maximum):
            runtime_blockers.append(f"{name} path leaves its commissioned workspace")

    reset_required = start_error > float(preinsert["start_joint_tolerance_rad"])
    reset_q = np.linspace(current_q, target_q, num=101)
    reset_xyz = np.asarray(
        [
            (
                np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
                @ joint7_to_tcp
            )[:3, 3]
            for q in reset_q
        ]
    )
    reset_fk_error = float(
        np.linalg.norm(reset_xyz[-1] - target_xyz)
    )
    if args.full_assembly:
        assert bracket_episode is not None
        reset_profiles = [("commissioned_bracket_reset", bracket_episode["automatic_reset_to_start"])]
    else:
        reset_profiles = [
            ("cross_task_bracket_retreat", task["automatic_reset_to_start"]),
            ("commissioned_preinsert_recovery", preinsert["automatic_reset_to_start"]),
        ]
    reset_name, reset = reset_profiles[0]
    reset_segment_count = 1
    reset_failures: list[tuple[str, list[str]]] = []
    if reset_required:
        for candidate_name, candidate in reset_profiles:
            candidate_delta = float(np.max(np.abs(target_q - current_q)))
            candidate_segment_count = 1
            candidate_for_validation = candidate
            if args.full_assembly:
                candidate_segment_count = max(
                    1,
                    int(
                        math.ceil(
                            candidate_delta
                            / float(candidate["maximum_joint_delta_rad"])
                        )
                    ),
                )
                maximum_segments = int(full_task["initial_reset_maximum_segments"])
                if candidate_segment_count <= maximum_segments:
                    # The configured delta remains the per-segment bound. The
                    # full path is unchanged; only its slow JointMotion
                    # execution is divided into equal bounded segments.
                    candidate_for_validation = {
                        **candidate,
                        "maximum_joint_delta_rad": candidate_delta + 1e-12,
                    }
            candidate_blockers = automatic_reset_blockers(
                {
                    **candidate_for_validation,
                    "requires_empty_gripper": True,
                    "requires_object_removed": True,
                },
                current_q,
                target_q,
                reset_xyz,
                reset_fk_error,
            )
            if args.full_assembly and candidate_segment_count > int(
                full_task["initial_reset_maximum_segments"]
            ):
                candidate_blockers.append(
                    f"requires {candidate_segment_count} reset segments, exceeding "
                    f"the configured maximum {int(full_task['initial_reset_maximum_segments'])}"
                )
            if not candidate_blockers:
                reset_name, reset = candidate_name, candidate
                reset_segment_count = candidate_segment_count
                break
            reset_failures.append((candidate_name, candidate_blockers))
        else:
            for candidate_name, candidate_blockers in reset_failures:
                runtime_blockers.extend(
                    f"automatic reset {candidate_name}: {item}"
                    for item in candidate_blockers
                )

    transition_q = None
    transition_xyz = None
    transition = None
    if args.full_assembly:
        assert bracket_plans is not None and full_task is not None
        transition = full_task["cross_task_transition"]
        transition_start = bracket_plans["retreat_open"].expected_stop_q
        transition_stop = plans["preinsert"]["approach_open"].expected_start_q
        transition_q = np.linspace(transition_start, transition_stop, num=101)
        transition_xyz = np.asarray(
            [
                (
                    np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
                    @ joint7_to_tcp
                )[:3, 3]
                for q in transition_q
            ]
        )
        transition_fk_error = float(
            np.linalg.norm(
                transition_xyz[-1]
                - plans["preinsert"]["approach_open"].xyz_targets[0]
            )
        )
        transition_blockers = []
        transition_delta = float(np.max(np.abs(transition_stop - transition_start)))
        if transition_delta > float(transition["maximum_joint_delta_rad"]):
            transition_blockers.append(
                f"joint delta {transition_delta:.6f} rad exceeds "
                f"{float(transition['maximum_joint_delta_rad']):.6f} rad"
            )
        transition_min = np.asarray(transition["workspace_bounds_m"]["min_xyz"])
        transition_max = np.asarray(transition["workspace_bounds_m"]["max_xyz"])
        if np.any(transition_xyz < transition_min) or np.any(transition_xyz > transition_max):
            transition_blockers.append("path leaves its configured workspace")
        if transition_fk_error > float(transition["target_fk_error_limit_m"]):
            transition_blockers.append(
                f"target FK error {transition_fk_error:.6f} m exceeds "
                f"{float(transition['target_fk_error_limit_m']):.6f} m"
            )
        transition_margin = float(transition["joint_margin_rad"])
        if np.any(transition_q < JOINT_MIN + transition_margin) or np.any(
            transition_q > JOINT_MAX - transition_margin
        ):
            transition_blockers.append("path violates guarded joint range")
        runtime_blockers.extend(
            f"cross-task transition: {item}" for item in transition_blockers
        )

    print(f"\nlive start error:         {start_error:.6f} rad")
    print(f"live Hand width/grasped:  {initial_width:.6f} m / {initial_grasped}")
    print(f"combined path xyz min:    {np.array2string(predicted_xyz.min(0), precision=5)}")
    print(f"combined path xyz max:    {np.array2string(predicted_xyz.max(0), precision=5)}")
    if transition_xyz is not None:
        print(
            f"cross-task reset xyz:      {np.array2string(transition_xyz.min(0), precision=5)} / "
            f"{np.array2string(transition_xyz.max(0), precision=5)}"
        )
    print(f"automatic reset:          {'REQUIRED' if reset_required else 'NOT NEEDED'}")
    if reset_required:
        print(f"reset profile selected:   {reset_name if len(reset_failures) < len(reset_profiles) else 'NONE'}")
        print(f"reset maximum q delta:    {float(np.max(np.abs(target_q - current_q))):.6f} rad")
        print(
            f"reset segments:           {reset_segment_count} x max "
            f"{float(np.max(np.abs(target_q - current_q))) / reset_segment_count:.6f} rad"
        )
        print(f"reset path xyz min:       {np.array2string(reset_xyz.min(0), precision=5)}")
        print(f"reset path xyz max:       {np.array2string(reset_xyz.max(0), precision=5)}")
        print(f"reset target FK error:    {reset_fk_error:.6f} m")
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")

    sys.path.insert(0, str(FR3_ROOT / "franka-tests"))
    from realsense_recorder import RealSenseEpisodeRecorder  # noqa: PLC0415

    camera = collection["camera"]
    serials = [entry["serial"] for entry in camera["serials_in_dataset_order"]]
    if args.recording_preview:
        with tempfile.TemporaryDirectory(prefix="fr3_bolt_recording_preview_") as temp:
            recorder = RealSenseEpisodeRecorder(
                temp,
                serials=serials,
                width=int(camera["width"]),
                height=int(camera["height"]),
                fps=int(camera["source_fps"]),
            )
            try:
                recorder.attach_robot(
                    robot,
                    gripper=None,
                    sample_hz=float(collection["robot_state_sample_hz"]),
                    gripper_command_source="unavailable",
                )
                recorder.set_achieved_gripper_state(
                    initial_width,
                    initial_grasped,
                    "recording_preview_initial_hand_state",
                    command_width=initial_width,
                )
                recorder.start_episode(
                    "full_assembly_recording_preview" if args.full_assembly else "bolt_recording_preview",
                    {
                        "task_name": "bracket_placement" if args.bracket_only else "full_bracket_bolt_assembly" if args.full_assembly else "bolt",
                        "bc_training_eligible": False,
                    },
                )
                time.sleep(2.0)
                frame_counts = dict(recorder.camera_frame_counts)
                state_count = len(recorder.state_samples)
                recorder.stop_episode(False, {"preview_only": True})
            finally:
                recorder.close()
        camera_failures = [
            name for name, count in frame_counts.items() if count < int(camera["source_fps"])
        ]
        if runtime_blockers or camera_failures or state_count < int(collection["robot_state_sample_hz"]):
            if camera_failures:
                print("Recording preview blocker: insufficient frames from " + ", ".join(camera_failures))
            if state_count < int(collection["robot_state_sample_hz"]):
                print(f"Recording preview blocker: only {state_count} robot states")
            raise SystemExit("[SAFETY] autonomous bolt recording preview failed")
        print(f"\nrecording preview frames: {frame_counts}")
        print(f"recording preview states: {state_count}")
        print("Recording preview passed. No arm or Hand command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] autonomous bolt demo execution refused")

    robot_ref = [robot]
    gripper_ref = [gripper]
    stop_requested = threading.Event()

    def stop_all() -> None:
        stop_requested.set()
        if robot_ref[0] is not None:
            try:
                robot_ref[0].stop()
            except BaseException:
                pass
        if gripper_ref[0] is not None:
            try:
                gripper_ref[0].stop()
            except BaseException:
                pass

    def interrupt(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; stopping available arm and Hand clients")
        stop_all()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    if reset_required:
        input(
            (
                "Remove the bracket from B1/frame and the bolt from the Hand/S1, "
                "clear the full path, then press Enter for the slow arm-only reset: "
            )
            if args.full_assembly
            else "Remove the bolt from the Hand and S1, clear the full path, then press Enter for the slow arm-only reset: "
        )
        reset_start_q = np.array(current_q, copy=True)
        for segment_index in range(1, reset_segment_count + 1):
            segment_target = reset_start_q + (
                float(segment_index) / float(reset_segment_count)
            ) * (target_q - reset_start_q)
            robot.move(
                JointMotion(
                    JointState(segment_target),
                    relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
                )
            )
            segment_error = float(
                np.max(np.abs(np.asarray(robot.state.q) - segment_target))
            )
            print(
                f"[auto-reset {segment_index}/{reset_segment_count}] "
                f"terminal error={segment_error:.6f} rad"
            )
            if segment_error > float(reset["terminal_joint_tolerance_rad"]):
                raise SystemExit("[SAFETY] automatic reset segment tolerance failed")
        error = float(np.max(np.abs(np.asarray(robot.state.q) - target_q)))
        print(f"[auto-reset] final terminal error={error:.6f} rad")
        if error > float(reset["terminal_joint_tolerance_rad"]):
            raise SystemExit("[SAFETY] automatic reset terminal tolerance failed")

    output_dir = full_task["output_dir"] if args.full_assembly else task["output_dir"]
    recorder = RealSenseEpisodeRecorder(
        project_path(output_dir),
        serials=serials,
        width=int(camera["width"]),
        height=int(camera["height"]),
        fps=int(camera["source_fps"]),
    )
    recorder.attach_robot(
        robot,
        gripper=None,
        sample_hz=float(collection["robot_state_sample_hz"]),
        gripper_command_source="unavailable",
    )
    recorder.set_achieved_gripper_state(
        float(gripper.state.width),
        bool(gripper.is_grasped),
        "initial_hand_state",
        command_width=float(gripper.state.width),
    )
    if args.bracket_only:
        print(f"\nPlace the bracket precisely at {args.bracket_slot}; leave the bolt at S1.")
    elif args.full_assembly:
        print(f"\nPlace the bracket precisely at {args.bracket_slot} and the bolt precisely at S1.")
        print("Both objects must remain in their recorded orientations from the start.")
    else:
        print("\nPlace the bolt precisely at S1 in the recorded orientation.")
    print("The narrow SHAFT must be between the fingertips; the wide circular HEAD must remain outside the jaws.")
    print("Keep yourself outside both camera views.")
    input(
        (
            f"Press Enter when the bracket is at {args.bracket_slot}, the pins and path are clear, "
            "and cameras are ready to record: "
        )
        if args.bracket_only else
        "Press Enter when B1, S1, the frame, and the complete path are ready to begin recording: "
        if args.full_assembly
        else "Press Enter when the bolt, mounted bracket, and complete path are ready to begin recording: "
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    episode_name = (
        f"{stamp}_full_bracket_bolt_autonomous_{args.episode_index:04d}_{args.bracket_slot}_S1_CENTER"
        if args.full_assembly
        else f"{stamp}_bolt_engagement_autonomous_{args.episode_index:04d}_S1_CENTER"
    )
    episode_dir = recorder.start_episode(
        episode_name,
        {
            "task_name": "bracket_placement" if args.bracket_only else "full_bracket_bolt_assembly" if args.full_assembly else "bolt",
            "collection_subtask": "bracket_placement" if args.bracket_only else None,
            "task": active_prompt,
            "language_prompt": active_prompt,
            "collection_mode": collection["collection_mode"],
            "robot_ip": collection["robot_ip"],
            "bracket_slot": args.bracket_slot,
            "screw_slot": task["slot"],
            "screw_grasp_style": task["grasp_style"],
            "episode_index": args.episode_index,
            "random_seed": args.seed,
            "dataset_target_fps": collection["dataset"]["target_fps"],
            "dataset_schema": collection["dataset"]["schema"],
            "action_dimension": collection["dataset"]["action_dimension"],
            "action_semantics": collection["dataset"]["action_semantics"],
            "expert_command_available": True,
            "gripper_command_available": True,
            "gripper_state_source": "automatic_command_measured_at_transition",
            "two_x_profile_enabled": bool(args.two_x),
            "three_x_profile_enabled": bool(args.three_x),
            "four_x_profile_enabled": bool(args.four_x),
            "accelerated_profile": speed_label if accelerated else None,
            "idle_edge_trim_enabled": bool(
                accelerated
                and two_x_config is not None
                and two_x_config.get("idle_edge_trim", {}).get("enabled") is True
            ),
            "trimmed_arm_duration_s": (
                expected_duration if args.bracket_only else
                float(two_x_config["offline_validation"]["retimed_total_arm_stage_duration_s"])
                if accelerated and two_x_config is not None else None
            ),
            "setup_validation_only": bool(args.full_assembly and not args.training_data and not args.bracket_only),
            # Eligibility is granted only by stop_episode after the operator's
            # terminal visual review. A crash can therefore never leave a raw
            # capture looking training-ready.
            "bc_training_eligible": False,
            "derived_training_views_requested": bool(args.training_data),
            "success_endpoint": resolve_success_endpoint(
                bracket_task=bracket_task,
                full_task=full_task,
                task=task,
                collection=collection,
                bracket_only=args.bracket_only,
                full_assembly=args.full_assembly,
            ),
            "source_trajectories": (
                [bracket_task["trajectory_dir"]] if args.bracket_only else [
                    bracket_task["trajectory_dir"],
                    task["preinsert_trajectory_dir"],
                    task["drop_regrasp_trajectory_dir"],
                ]
                if args.full_assembly
                else [task["preinsert_trajectory_dir"], task["drop_regrasp_trajectory_dir"]]
            ),
            **({} if args.bracket_only else {
                "engagement_controllers": [
                    stage["binary"] for stage in active_engagement_sequence
                ],
                "total_engagement_rotation_deg": task["total_engagement_rotation_deg"],
                "threading_axial_force_target_N": 1.200360,
                "threading_maximum_axial_speed_m_s": 0.000500,
            }),
            "recording_config": str(args.recording_config.expanduser().resolve()),
            "camera_roles": {
                entry["role"]: entry["serial"] for entry in camera["serials_in_dataset_order"]
            },
        },
    )
    may_hold = False

    def operator_gate(prompt: str, rejection: str) -> None:
        """Keep standalone commissioning prompts, but never pause a full flow."""
        if args.full_assembly:
            return
        if input(prompt):
            raise RuntimeError(rejection)

    def run_retimed_2x_stage(name: str, episode: dict, plan, monitor_grasp: bool) -> None:
        """Run one guarded accelerated object stage through the 1 kHz executor."""

        nonlocal robot
        assert two_x_config is not None
        stage_key = {
            "bracket_approach_open": "bracket_approach_open",
            "bracket_transport_place_closed": "bracket_transport_place_closed",
            "bracket_retreat_open": "bracket_retreat_open",
            "approach_open": "bolt_approach_open",
            "transport_preinsert_closed": "bolt_transport_preinsert_closed",
            "lower_held": "bolt_lower_held",
            "reposition_regrasp_open": "bolt_reposition_regrasp_open",
        }[name]
        stage_arm_config = arm_config_for(stage_key)
        retimed = retime_stage(plan, stage_key)
        current_width = float(gripper.state.width)
        current_grasped = hand_still_holds_commanded_object()
        recorder.add_event(
            f"{name}_started",
            {
                "controller": f"direct_libfranka_retimed_{speed_label}",
                "duration_s": retimed.duration_s,
                "source_trim_start_s": retimed.source_trim_start_s,
                "source_trim_end_s": retimed.source_trim_end_s,
                "peak_velocity_rad_s": retimed.peak_abs_velocity_rad_s,
                "peak_acceleration_rad_s2": retimed.peak_abs_acceleration_rad_s2,
                "peak_jerk_rad_s3": retimed.peak_abs_jerk_rad_s3,
            },
        )
        recorder.pause_robot_state_sampling()
        last_python_state = max(
            recorder.state_samples,
            key=lambda sample: int(sample["time_ns"]),
        )
        robot_ref[0] = None
        old_robot, robot = robot, None
        del old_robot
        gc.collect()
        time.sleep(0.25)

        telemetry_path = episode_dir / f"retimed_{speed_label}_{stage_key}.csv"
        lost = threading.Event()
        monitor_stop = threading.Event()
        with tempfile.TemporaryDirectory(prefix=f"fr3_retimed_{speed_label}_{stage_key}_") as temp:
            plan_path = Path(temp) / "plan.csv"
            write_retimed_csv(plan_path, retimed)
            process = subprocess.Popen(
                [
                    str(project_path(stage_arm_config["binary"])),
                    "execute",
                    collection["robot_ip"],
                    str(plan_path),
                    str(telemetry_path),
                    "RUN_RETIMED_ACCELERATED_STAGE",
                    stage_key,
                    *(str(value) for value in episode["workspace_bounds_m"]["min_xyz"]),
                    *(str(value) for value in episode["workspace_bounds_m"]["max_xyz"]),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            def monitor() -> None:
                while not monitor_stop.wait(1.0 / float(episode["gripper"]["monitor_hz"])):
                    try:
                        if not hand_still_holds_commanded_object():
                            lost.set()
                            process.send_signal(signal.SIGINT)
                            return
                    except BaseException:
                        lost.set()
                        process.send_signal(signal.SIGINT)
                        return

            thread = threading.Thread(target=monitor, daemon=True)
            if monitor_grasp:
                thread.start()
            stdout, stderr = process.communicate(input="\n\n")
            monitor_stop.set()
            if monitor_grasp:
                thread.join(timeout=1.0)
        print(stdout, end="")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if lost.is_set():
            raise RuntimeError(f"{name} lost the object grasp during {speed_label} replay")
        if process.returncode != 0 or "[result] stop reason:      motion_completed" not in stdout:
            raise RuntimeError(
                f"{name} smooth {speed_label} controller did not complete normally"
            )

        external = retimed_direct_samples(
            telemetry_path,
            current_width,
            current_grasped,
            current_width,
            speed_label,
        )
        bridge = stationary_state_bridge(
            last_python_state,
            int(external[0]["time_ns"]),
            float(collection["robot_state_sample_hz"]),
            current_width,
            current_grasped,
            current_width,
        )
        recorder.append_external_state_samples(bridge + external)
        robot = Robot(collection["robot_ip"])
        robot_ref[0] = robot
        post_sample = recorder.attach_robot(
            robot,
            gripper=None,
            sample_hz=float(collection["robot_state_sample_hz"]),
            gripper_command_source="unavailable",
        )
        post_bridge = stationary_state_bridge(
            external[-1],
            int(post_sample["time_ns"]),
            float(collection["robot_state_sample_hz"]),
            current_width,
            current_grasped,
            current_width,
        )
        post_sample.update(
            {
                "gripper_state_time_ns": int(post_sample["time_ns"]),
                "gripper_width": current_width,
                "gripper_is_grasped": current_grasped,
                "gripper_command_width": current_width,
                "state_source": f"post_retimed_{speed_label}_controller_reconnection",
            }
        )
        recorder.append_external_state_samples(post_bridge + [post_sample])
        error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_stop_q)))
        print(f"[{name}] retimed {speed_label} terminal error={error:.6f} rad")
        if error > float(episode["terminal_joint_tolerance_rad"]):
            raise RuntimeError(f"{name} {speed_label} terminal tolerance failed")
        recorder.add_event(
            f"{name}_completed",
            {
                "terminal_error_rad": error,
                "controller_samples_imported": len(external),
                "stationary_handoff_samples_inserted": len(bridge) + len(post_bridge),
            },
        )

    def run_stage(name: str, episode: dict, plan, monitor_grasp: bool) -> None:
        nonlocal may_hold
        if accelerated:
            run_retimed_2x_stage(name, episode, plan, monitor_grasp)
            return
        robot.relative_dynamics_factor = float(episode["stage_robot_global_dynamics_factor"])
        lost = threading.Event()
        monitor_stop = threading.Event()

        def monitor() -> None:
            while not monitor_stop.wait(1.0 / float(episode["gripper"]["monitor_hz"])):
                try:
                    if not hand_still_holds_commanded_object():
                        lost.set()
                        robot.stop()
                        return
                except BaseException:
                    lost.set()
                    robot.stop()
                    return

        thread = threading.Thread(target=monitor, daemon=True)
        if monitor_grasp:
            thread.start()
        timer = threading.Timer(float(episode["state_timeout_s"]), stop_all)
        timer.daemon = True
        timer.start()
        recorder.add_event(f"{name}_started")
        try:
            robot.move(make_waypoint_motion(plan, episode))
        finally:
            timer.cancel()
            monitor_stop.set()
            if monitor_grasp:
                thread.join(timeout=1.0)
        if stop_requested.is_set() or lost.is_set():
            raise RuntimeError(f"{name} stopped or lost the bolt grasp")
        error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_stop_q)))
        print(f"[{name}] terminal error={error:.6f} rad")
        if error > float(episode["terminal_joint_tolerance_rad"]):
            raise RuntimeError(f"{name} terminal tolerance failed")
        recorder.add_event(f"{name}_completed")

    open_speed = float(drop["gripper"]["open_speed_m_s"])
    pickup_grasp = task["pickup_shaft_grasp"]
    head_grasp = task["thread_ready_head_grasp"]
    held_width = float("nan")
    pickup_width_fallback_active = False

    def hand_still_holds_commanded_object() -> bool:
        """Use Franka state, with a bounded shaft-width fallback when needed."""

        if bool(gripper.is_grasped):
            return True
        if not pickup_width_fallback_active:
            return False
        width = float(gripper.state.width)
        return (
            float(pickup_grasp["physical_hold_width_min_m"])
            <= width
            <= float(pickup_grasp["physical_hold_width_max_m"])
        )

    def check_head_grasp_result(
        event_prefix: str,
        *,
        accepted: bool | None,
        is_grasped: bool,
        width: float,
    ) -> bool:
        """Validate a head grasp without discarding known-good visual evidence.

        Franky has returned accepted=False/is_grasped=False for physically held
        printed parts at a valid nonzero width.  Keep that raw result, enforce a
        bounded physical width, and require the existing operator confirmation
        before labeling a non-strict result as held.
        """
        strict_width = (
            float(head_grasp["accepted_width_min_m"])
            <= width
            <= float(head_grasp["accepted_width_max_m"])
        )
        accepted_ok = accepted is not False
        strict = bool(accepted_ok and is_grasped and strict_width)
        physical_sanity = (
            float(head_grasp["operator_confirmed_width_min_m"])
            <= width
            <= float(head_grasp["operator_confirmed_width_max_m"])
        )
        bounded_obstruction = (
            float(head_grasp["physical_obstruction_width_min_m"])
            <= width
            <= float(head_grasp["physical_obstruction_width_max_m"])
        )
        # accepted/is_grasped is sufficient inside the broad physical envelope.
        # If Franka's epsilon boolean is false, require the fingers to remain
        # stopped measurably wider than the empty-Hand target.  This prevents a
        # target-width empty Hand from being mislabeled as holding the bolt.
        physical_evidence = bool(
            physical_sanity
            and ((accepted_ok and is_grasped) or bounded_obstruction)
        )
        print(
            f"[{event_prefix}] head state accepted="
            f"{accepted if accepted is not None else 'not_applicable'} "
            f"is_grasped={is_grasped} width={width:.6f} m "
            f"strict_validation={strict}"
        )
        recorder.log_action(
            f"{event_prefix}_head_grasp_command_result",
            {
                "target_width": float(head_grasp["target_width_m"]),
                "achieved_width": width,
                "accepted": accepted,
                "is_grasped": bool(is_grasped),
                "strict_calibrated_width": bool(strict_width),
                "physical_width_sanity": bool(physical_sanity),
                "bounded_physical_obstruction": bool(bounded_obstruction),
                "physical_hold_evidence": bool(physical_evidence),
            },
        )
        if not physical_evidence:
            raise RuntimeError(
                f"{event_prefix} head state does not provide bounded physical "
                f"hold evidence at width {width:.6f} m"
            )
        if not strict:
            print(
                f"[{event_prefix}] Franky strict validation disagrees with the "
                "bounded physical width; mandatory visual confirmation will decide"
            )
        return strict

    def regrasp_stationary_head(event_prefix: str, *, ratchet_reset: bool) -> None:
        nonlocal may_hold, held_width
        print(f"[{event_prefix}] opening the Hand while the bolt remains stationary")
        gripper.open(open_speed)
        may_hold = False
        open_width = float(gripper.state.width)
        if open_width < float(drop["gripper"]["minimum_open_width_m"]):
            raise RuntimeError(f"{event_prefix} failed to open the Hand")
        recorder.set_achieved_gripper_state(
            open_width,
            False,
            f"{event_prefix}_hand_opened",
            command_width=0.08,
        )
        check = "" if args.full_assembly else input(
            "Confirm the bolt stayed upright/fixed and the open-finger sweep is clear. "
            "Press Enter to continue; type anything to abort: "
        )
        if check:
            raise RuntimeError(f"operator rejected {event_prefix} upright-bolt check")
        if ratchet_reset:
            target = np.array(robot.state.q, dtype=np.float64, copy=True)
            start_q7 = float(target[6])
            target[6] = float(task["engagement_start_q7_rad"])
            recorder.add_event(
                f"{event_prefix}_empty_wrist_reset_started",
                {"start_q7_rad": start_q7, "target_q7_rad": float(target[6])},
            )
            robot.move(
                JointMotion(
                    JointState(target),
                    relative_dynamics_factor=float(
                        task["empty_wrist_prewind_relative_dynamics_factor"]
                    ),
                )
            )
            final_q = np.asarray(robot.state.q, dtype=np.float64)
            error = float(np.max(np.abs(final_q - target)))
            print(
                f"[{event_prefix}] empty q7 reset "
                f"{start_q7:+.6f} -> {final_q[6]:+.6f} rad; error={error:.6f}"
            )
            if error > 0.005:
                raise RuntimeError(f"{event_prefix} empty-wrist reset tolerance failed")
            recorder.add_event(
                f"{event_prefix}_empty_wrist_reset_completed",
                {"terminal_error_rad": error},
            )
        accepted = gripper.grasp(
            float(head_grasp["target_width_m"]),
            float(head_grasp["speed_m_s"]),
            float(head_grasp["force_N"]),
            float(head_grasp["epsilon_inner_m"]),
            float(head_grasp["epsilon_outer_m"]),
        )
        may_hold = True
        held_width = float(gripper.state.width)
        held = bool(gripper.is_grasped)
        strict = check_head_grasp_result(
            event_prefix,
            accepted=bool(accepted),
            is_grasped=held,
            width=held_width,
        )
        check = "" if args.full_assembly else input(
            "Confirm the bolt is upright, securely head-held, and unloaded. "
            "Press Enter to continue; type anything to abort: "
        )
        if check:
            raise RuntimeError(f"operator rejected {event_prefix} head regrasp")
        recorder.set_achieved_gripper_state(
            held_width,
            True,
            f"{event_prefix}_head_regrasped",
            command_width=float(head_grasp["target_width_m"]),
            operator_confirmed=not strict and not args.full_assembly,
        )

    def run_direct_drive(stage: dict) -> None:
        nonlocal robot, gripper, held_width
        stage_name = str(stage["name"])
        angle_deg = float(stage["angle_deg"])
        recorder.add_event(
            f"{stage_name}_connection_handoff_started",
            {"angle_deg": angle_deg},
        )
        recorder.pause_robot_state_sampling()
        last_python_state = max(
            recorder.state_samples,
            key=lambda sample: int(sample["time_ns"]),
        )
        robot_ref[0] = None
        gripper_ref[0] = None
        old_robot, old_gripper = robot, gripper
        robot = None
        gripper = None
        del old_robot
        del old_gripper
        gc.collect()
        time.sleep(0.25)

        telemetry_path = episode_dir / f"direct_{stage_name}.csv"
        with tempfile.TemporaryDirectory(prefix=f"fr3_production_{stage_name}_") as temp:
            model_path = Path(temp) / "dynamic_model.csv"
            dynamic_model_csv(model_path, project_path(task["dynamic_model_calibration"]))
            print(
                f"Keep one hand near emergency stop. "
                f"The commissioned +{angle_deg:.0f}-degree drive begins now."
            )
            completed = subprocess.run(
                [
                    str(project_path(stage["binary"])),
                    "execute",
                    collection["robot_ip"],
                    str(telemetry_path),
                    "COMBINED_THREAD_AUTHORIZED",
                    str(model_path),
                ],
                input="\n\n",
                text=True,
                capture_output=True,
                check=False,
            )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if (
            completed.returncode != 0
            or "[result] stop reason:      motion_completed" not in completed.stdout
        ):
            raise RuntimeError(
                f"direct +{angle_deg:.0f}-degree drive did not complete normally"
            )
        external = direct_samples(
            telemetry_path,
            held_width,
            True,
            float(head_grasp["target_width_m"]),
        )
        bridge = stationary_state_bridge(
            last_python_state,
            int(external[0]["time_ns"]),
            float(collection["robot_state_sample_hz"]),
            held_width,
            True,
            float(head_grasp["target_width_m"]),
        )
        recorder.append_external_state_samples(bridge + external)
        recorder.add_event(
            f"{stage_name}_completed",
            {
                "angle_deg": angle_deg,
                "controller_samples_imported": len(external),
                "stationary_handoff_samples_inserted": len(bridge),
            },
        )

        robot = Robot(collection["robot_ip"])
        gripper = Gripper(collection["robot_ip"])
        robot_ref[0] = robot
        gripper_ref[0] = gripper
        post_sample = recorder.attach_robot(
            robot,
            gripper=None,
            sample_hz=float(collection["robot_state_sample_hz"]),
            gripper_command_source="unavailable",
        )
        post_bridge = stationary_state_bridge(
            external[-1],
            int(post_sample["time_ns"]),
            float(collection["robot_state_sample_hz"]),
            held_width,
            True,
            float(head_grasp["target_width_m"]),
        )
        post_sample.update(
            {
                "gripper_state_time_ns": int(post_sample["time_ns"]),
                "gripper_width": float(held_width),
                "gripper_is_grasped": True,
                "gripper_command_width": float(head_grasp["target_width_m"]),
                "state_source": "post_direct_controller_reconnection",
            }
        )
        recorder.append_external_state_samples(post_bridge + [post_sample])
        post_width = float(gripper.state.width)
        post_grasped = bool(gripper.is_grasped)
        post_strict = check_head_grasp_result(
            f"{stage_name}_post_drive",
            accepted=None,
            is_grasped=post_grasped,
            width=post_width,
        )
        if not post_strict and not args.full_assembly and input(
            "Franky strict post-drive validation disagrees with the bounded "
            "physical width. Visually confirm the Hand still securely holds the "
            "upright bolt; press Enter to continue, type anything to abort: "
        ):
            raise RuntimeError(f"operator rejected {stage_name} post-drive grasp")
        held_width = post_width
        recorder.set_achieved_gripper_state(
            post_width,
            True,
            f"{stage_name}_hand_still_closed",
            command_width=float(head_grasp["target_width_m"]),
            operator_confirmed=not post_strict and not args.full_assembly,
        )

    try:
        if args.full_assembly:
            assert bracket_episode is not None and bracket_plans is not None
            bracket_grasp = bracket_episode["gripper"]
            bracket_open_speed = float(bracket_grasp["open_speed_m_s"])
            gripper.open(bracket_open_speed)
            bracket_open_width = float(gripper.state.width)
            recorder.set_achieved_gripper_state(
                bracket_open_width,
                False,
                "bracket_hand_opened",
                command_width=0.08,
            )
            run_stage(
                "bracket_approach_open",
                bracket_episode,
                bracket_plans["approach_open"],
                False,
            )
            bracket_accepted = gripper.grasp(
                float(bracket_grasp["grasp_target_width_m"]),
                float(bracket_grasp["grasp_speed_m_s"]),
                float(bracket_grasp["grasp_force_n"]),
                float(bracket_grasp["epsilon_inner_m"]),
                float(bracket_grasp["epsilon_outer_m"]),
            )
            may_hold = True
            bracket_width = float(gripper.state.width)
            bracket_grasped = bool(gripper.is_grasped)
            recorder.set_achieved_gripper_state(
                bracket_width,
                bracket_grasped,
                "bracket_grasped",
                command_width=float(bracket_grasp["grasp_target_width_m"]),
            )
            print(
                f"[bracket grasp] accepted={bracket_accepted} "
                f"is_grasped={bracket_grasped} width={bracket_width:.6f} m"
            )
            if not bracket_accepted or not bracket_grasped or not (
                float(bracket_grasp["held_width_min_m"])
                <= bracket_width
                <= float(bracket_grasp["held_width_max_m"])
            ):
                raise RuntimeError("automatic bracket grasp validation failed")
            if not args.full_assembly and input(
                "Visually confirm the bracket is securely held and S1 remains clear. "
                "Press Enter to continue; type anything to abort: "
            ):
                raise RuntimeError("operator rejected bracket grasp")
            run_stage(
                "bracket_transport_place_closed",
                bracket_episode,
                bracket_plans["transport_place_closed"],
                True,
            )
            gripper.open(bracket_open_speed)
            may_hold = False
            bracket_release_width = float(gripper.state.width)
            if bracket_release_width < float(bracket_grasp["minimum_open_width_m"]):
                raise RuntimeError("Hand failed to open after bracket placement")
            recorder.set_achieved_gripper_state(
                bracket_release_width,
                False,
                "bracket_mounted_released",
                command_width=0.08,
            )
            run_stage(
                "bracket_retreat_open",
                bracket_episode,
                bracket_plans["retreat_open"],
                False,
            )
            if args.bracket_only:
                recorder.add_event(
                    "training_motion_completed",
                    {
                        "task": "bracket",
                        "bracket_mounted": True,
                        "bolt_left_in_fixed_scene": True,
                        "terminal_context_s": 1.0,
                        "operator_review_follows": True,
                    },
                )
                input("Confirm bracket mounted and Hand open; press Enter to continue: ")
                recorder.add_event(
                    "autonomous_bracket_episode_finished",
                    {"accepted_by_operator": True},
                )
                recorder.stop_episode(
                    True,
                    {
                        "accepted_by_operator": True,
                        "pending_visual_review": False,
                        "guarded_sequence_completed": True,
                        "termination_reason": "verified_bracket_placement",
                        "failure_reason": None,
                        "bc_training_eligible": True,
                        "automatic_retreat_issued": False,
                    },
                )
                recorder.close()
                print(f"[SUCCESS] bracket-only training capture saved: {episode_dir}")
                return
            if not args.full_assembly and input(
                "Confirm the bracket is mounted, the Hand is open, and the bolt remains "
                "correctly staged at S1. Press Enter for the automatic cross-task "
                "transition; type anything to abort: "
            ):
                raise RuntimeError("operator rejected mounted bracket or staged bolt")
            assert transition is not None
            recorder.add_event("cross_task_transition_started")
            robot.move(
                JointMotion(
                    JointState(plans["preinsert"]["approach_open"].expected_start_q),
                    relative_dynamics_factor=float(transition["relative_dynamics_factor"]),
                )
            )
            transition_error = float(
                np.max(
                    np.abs(
                        np.asarray(robot.state.q)
                        - plans["preinsert"]["approach_open"].expected_start_q
                    )
                )
            )
            print(f"[cross-task transition] terminal error={transition_error:.6f} rad")
            if transition_error > float(transition["terminal_joint_tolerance_rad"]):
                raise RuntimeError("cross-task transition terminal tolerance failed")
            recorder.add_event(
                "cross_task_transition_completed",
                {"terminal_error_rad": transition_error},
            )

        gripper.open(open_speed)
        width = float(gripper.state.width)
        recorder.set_achieved_gripper_state(width, False, "pickup_hand_opened", command_width=0.08)
        run_stage("approach_open", preinsert, plans["preinsert"]["approach_open"], False)
        accepted = gripper.grasp(
            float(pickup_grasp["target_width_m"]),
            float(pickup_grasp["speed_m_s"]),
            float(pickup_grasp["force_N"]),
            float(pickup_grasp["epsilon_inner_m"]),
            float(pickup_grasp["epsilon_outer_m"]),
        )
        may_hold = True
        width = float(gripper.state.width)
        grasped = bool(gripper.is_grasped)
        strict_pickup, bounded_pickup = pickup_grasp_evidence(
            pickup_grasp,
            accepted=bool(accepted),
            is_grasped=grasped,
            width_m=width,
        )
        pickup_width_fallback_active = bool(not strict_pickup and bounded_pickup)
        recorder.set_achieved_gripper_state(
            width,
            bool(strict_pickup or bounded_pickup),
            "bolt_pickup_grasped",
            command_width=float(pickup_grasp["target_width_m"]),
            operator_confirmed=False,
        )
        print(
            f"[bolt pickup grasp] accepted={accepted} "
            f"is_grasped={grasped} width={width:.6f} m "
            f"bounded_obstruction={bounded_pickup}"
        )
        recorder.log_action(
            "bolt_pickup_grasp_validation",
            {
                "strict_franka_success": strict_pickup,
                "bounded_physical_obstruction": bounded_pickup,
                "width_m": width,
                "fallback_active": pickup_width_fallback_active,
            },
        )
        if not bounded_pickup:
            raise RuntimeError(
                "pickup width does not show the commissioned shaft between the fingers"
            )
        if pickup_width_fallback_active:
            print(
                "[bolt pickup grasp] Franka's strict boolean disagrees with the "
                "bounded shaft obstruction; continuing with live width-loss monitoring"
            )
        operator_gate(
            "Visually confirm the bolt was picked up securely. Press Enter to continue; "
            "type anything to reject: ",
            "operator rejected pickup grasp",
        )
        run_stage(
            "transport_preinsert_closed",
            preinsert,
            plans["preinsert"]["transport_preinsert_closed"],
            True,
        )
        run_stage("lower_held", drop, plans["drop"]["lower_held"], True)
        operator_gate(
            "Visually confirm the bolt is upright and self-supported. Press Enter to "
            "release; type anything to reject: ",
            "operator rejected self-support check",
        )
        gripper.open(open_speed)
        may_hold = False
        pickup_width_fallback_active = False
        width = float(gripper.state.width)
        if width < float(drop["gripper"]["minimum_open_width_m"]):
            raise RuntimeError("Hand failed to open at the guide")
        recorder.set_achieved_gripper_state(width, False, "bolt_standing_released", command_width=0.08)
        operator_gate(
            "Confirm the bolt remained upright. Press Enter to reapproach; type anything "
            "to reject: ",
            "operator rejected upright bolt after release",
        )
        run_stage(
            "reposition_regrasp_open",
            drop,
            plans["drop"]["reposition_regrasp_open"],
            False,
        )
        # Franky may expose RobotState.q as a read-only NumPy view. The q7-only
        # target must be an owned writable array before replacing its endpoint.
        prewind_q = np.array(robot.state.q, dtype=np.float64, copy=True)
        prewind_start_q7 = float(prewind_q[6])
        prewind_q[6] = float(task["engagement_start_q7_rad"])
        if not -0.156888 <= prewind_q[6] <= 2.998908:
            raise RuntimeError("empty-wrist prewind target leaves the cable-safe q7 interval")
        recorder.add_event(
            "empty_wrist_prewind_started",
            {"start_q7_rad": prewind_start_q7, "target_q7_rad": float(prewind_q[6])},
        )
        print(
            f"[prewind] rotating the OPEN empty wrist q7 "
            f"{prewind_start_q7:+.6f} -> {prewind_q[6]:+.6f} rad"
        )
        robot.move(
            JointMotion(
                JointState(prewind_q),
                relative_dynamics_factor=float(
                    task["empty_wrist_prewind_relative_dynamics_factor"]
                ),
            )
        )
        prewind_final = np.asarray(robot.state.q, dtype=np.float64)
        prewind_error = float(np.max(np.abs(prewind_final - prewind_q)))
        if prewind_error > 0.005:
            raise RuntimeError("empty-wrist prewind terminal tolerance failed")
        recorder.add_event(
            "empty_wrist_prewind_completed",
            {"terminal_error_rad": prewind_error},
        )
        accepted = gripper.grasp(
            float(head_grasp["target_width_m"]),
            float(head_grasp["speed_m_s"]),
            float(head_grasp["force_N"]),
            float(head_grasp["epsilon_inner_m"]),
            float(head_grasp["epsilon_outer_m"]),
        )
        may_hold = True
        held_width = float(gripper.state.width)
        held_grasped = bool(gripper.is_grasped)
        strict = check_head_grasp_result(
            "initial",
            accepted=bool(accepted),
            is_grasped=held_grasped,
            width=held_width,
        )
        if not args.full_assembly and input(
            "Confirm THREAD_READY: upright, securely head-held, and unloaded. "
            f"Press Enter to run the guarded {float(task['total_engagement_rotation_deg']):.0f}-degree engagement chain; type anything to abort: "
        ):
            raise RuntimeError("operator rejected THREAD_READY")
        recorder.set_achieved_gripper_state(
            held_width,
            True,
            "bolt_head_regrasped",
            command_width=float(head_grasp["target_width_m"]),
            operator_confirmed=not strict and not args.full_assembly,
        )
        # Give both cameras terminal context with the achieved head grasp, then
        # persist the exact hybrid-policy handoff boundary. The following
        # threading motion remains part of the full-policy view only.
        time.sleep(0.15)
        recorder.add_event(
            "thread_ready_completed",
            {
                "head_grasp_width_m": held_width,
                "strict_franka_validation": strict,
                "automatic_threading_follows": True,
            },
        )
        for stage in active_engagement_sequence:
            if stage.get("empty_wrist_ratchet_before"):
                regrasp_stationary_head(
                    f"before_{stage['name']}_ratchet",
                    ratchet_reset=True,
                )
            elif stage.get("stationary_open_regrasp_before"):
                regrasp_stationary_head(
                    f"before_{stage['name']}_unload",
                    ratchet_reset=False,
                )
            run_direct_drive(stage)

        if not args.full_assembly:
            input(
                f"The complete {float(task['total_engagement_rotation_deg']):.0f}-degree chain finished. Press Enter to open the stationary "
                "Hand and inspect engagement: "
            )
        gripper.open(open_speed)
        may_hold = False
        release_width = float(gripper.state.width)
        if release_width < float(drop["gripper"]["minimum_open_width_m"]):
            raise RuntimeError("Hand failed to open after engagement")
        recorder.set_achieved_gripper_state(release_width, False, "engaged_bolt_released", command_width=0.08)
        time.sleep(0.3)
        # End the learnable trajectory before the operator spends arbitrary
        # time visually reviewing the result. Cameras continue recording so
        # the acceptance decision remains auditable in the immutable raw run.
        recorder.add_event(
            "training_motion_completed",
            {
                "bolt_released": True,
                "terminal_context_s": 1.0,
                "operator_review_follows": True,
            },
        )
        accepted_demo = (
            input(
                (
                    "Visually confirm the bracket is mounted, the bolt is visibly "
                    "thread-engaged/self-supported, and the Hand is open. "
                    if args.full_assembly
                    else "Visually confirm the bolt is visibly thread-engaged/self-supported "
                    "and the Hand is open. "
                )
                + "Press Enter to accept; type anything to reject: "
            )
            == ""
            if (not args.full_assembly or args.training_data)
            else False
        )
        pending_visual_review = bool(args.full_assembly and not args.training_data)
        recorder.add_event("autonomous_bolt_episode_finished", {"accepted_by_operator": accepted_demo})
        recorder.stop_episode(
            accepted_demo,
            {
                "accepted_by_operator": accepted_demo,
                "pending_visual_review": pending_visual_review,
                "guarded_sequence_completed": True,
                "termination_reason": (
                    "pending_visual_review"
                    if pending_visual_review
                    else "verified_thread_engagement"
                    if accepted_demo
                    else "operator_rejected"
                ),
                "failure_reason": (
                    None
                    if accepted_demo or pending_visual_review
                    else "operator_rejected_final_engagement"
                ),
                "bc_training_eligible": accepted_demo,
                "automatic_retreat_issued": False,
            },
        )
        recorder.close()
        if pending_visual_review:
            print(
                f"[COMPLETE] uninterrupted full-assembly recording saved pending visual "
                f"review: {episode_dir}"
            )
            print("Recording has stopped. Use Pilot to retreat safely.")
            return
        if not accepted_demo:
            print(f"[REJECTED] raw episode retained outside training split: {episode_dir}")
            return
        if args.training_data:
            print(
                f"[SUCCESS] accepted synchronized training capture saved: {episode_dir}\n"
                "It is eligible for bracket, full-policy bolt, and hybrid THREAD_READY views."
            )
    except BaseException as error:
        stop_all()
        if recorder.active:
            recorder.add_event("autonomous_bolt_episode_failed", {"error": str(error)})
            recorder.stop_episode(
                False,
                {
                    "termination_reason": "guard_or_controller_exception",
                    "failure_reason": str(error),
                    "bc_training_eligible": False,
                },
            )
        recorder.close()
        if may_hold:
            print("[SAFETY] The Hand may still hold the bolt; it was NOT automatically opened.")
        raise

    print(f"[SUCCESS] autonomous bolt engagement training demo saved: {episode_dir}")
    print("Recording has stopped. Use Pilot to retreat safely; that manual motion is outside the demo.")


if __name__ == "__main__":
    main()
