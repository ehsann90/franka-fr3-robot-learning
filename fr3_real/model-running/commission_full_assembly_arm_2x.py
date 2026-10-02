#!/usr/bin/env python3
"""Commission smooth 2x arm stages, including the final object-aware reapproach."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_drop_regrasp import build_bolt_drop_regrasp_stage  # noqa: E402
from expert_demo.bolt_preinsert import build_bolt_preinsert_stage  # noqa: E402
from expert_demo.replay_plan import build_replay_stage  # noqa: E402
from expert_demo.retimed_trajectory import (  # noqa: E402
    build_retimed_trajectory,
    write_retimed_csv,
)


DEFAULT_CONFIG = FR3_ROOT / "configs" / "full_assembly_2x_commissioning_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_stage(config: dict, name: str):
    stage = config["stages"][name]
    trajectory = project_path(stage["trajectory_dir"])
    source_kind = stage["source_kind"]
    if source_kind == "bracket":
        plan = build_replay_stage(
            trajectory, stage["source_stage"], command_rate_hz=15.0
        )
    elif source_kind == "bolt_preinsert":
        plan = build_bolt_preinsert_stage(
            str(trajectory), stage["source_stage"], command_rate_hz=15.0
        )
    elif source_kind == "bolt_drop_regrasp":
        plan = build_bolt_drop_regrasp_stage(
            str(trajectory),
            stage["source_stage"],
            command_rate_hz=15.0,
            trim_start_s=float(stage["trim_start_s"]),
            trim_end_s=float(stage["trim_end_s"]),
        )
    else:
        raise ValueError(f"unknown source_kind {source_kind!r}")
    idle_trim = config.get("idle_edge_trim", {})
    trajectory_2x = build_retimed_trajectory(
        plan,
        playback_time_scale=float(config["playback_time_scale"]),
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
    return stage, plan, trajectory_2x


def offline_blockers(config: dict, trajectory) -> list[str]:
    blockers = []
    for label, measured, configured in (
        ("velocity", trajectory.peak_abs_velocity_rad_s, config["velocity_cap_rad_s"]),
        (
            "acceleration",
            trajectory.peak_abs_acceleration_rad_s2,
            config["acceleration_cap_rad_s2"],
        ),
        ("jerk", trajectory.peak_abs_jerk_rad_s3, config["jerk_cap_rad_s3"]),
    ):
        if measured > float(configured):
            blockers.append(
                f"planned peak {label} {measured:.6f} exceeds cap {float(configured):.6f}"
            )
    if float(np.max(np.abs(trajectory.evaluate(0.0, 1)))) > 1e-10:
        blockers.append("trajectory does not start at zero velocity")
    if float(np.max(np.abs(trajectory.evaluate(trajectory.duration_s, 1)))) > 1e-10:
        blockers.append("trajectory does not end at zero velocity")
    return blockers


def print_stage(name: str, stage: dict, source_plan, trajectory, blockers: list[str]) -> None:
    print(f"\nstage:                    {name}")
    print(f"source stage:             {stage['source_kind']} / {stage['source_stage']}")
    print(f"source/current/trimmed 2x:{source_plan.source_duration_s:.3f} / "
          f"{source_plan.source_duration_s * 4.0:.3f} / {trajectory.duration_s:.3f} s")
    print(f"trimmed idle start/end:   {trajectory.source_trim_start_s:.3f} / "
          f"{trajectory.source_trim_end_s:.3f} s source time")
    print(f"retimed peak |dq|:        {trajectory.peak_abs_velocity_rad_s:.6f} rad/s")
    print(f"retimed peak |ddq|:       {trajectory.peak_abs_acceleration_rad_s2:.6f} rad/s^2")
    print(f"retimed peak |jerk|:      {trajectory.peak_abs_jerk_rad_s3:.6f} rad/s^3")
    print("endpoint velocity:        exactly zero / exactly zero")
    print(f"offline status:           {'BLOCKED' if blockers else 'PASSED'}")
    for blocker in blockers:
        print(f"  - {blocker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--offline-validation", action="store_true")
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--prepare-controller-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--execute-with-objects", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text())
    object_aware = bool(args.execute_with_objects)
    # argparse choices must come from the selected config, not module import time.
    stage_name = args.stage
    if args.offline_validation:
        requested_stages = list(config["stages"])
    else:
        if stage_name not in config["stages"]:
            raise SystemExit("--stage must name one configured stage")
        requested_stages = [stage_name]

    print("=== FULL-ASSEMBLY SMOOTH 2X ARM COMMISSIONING ===")
    print(f"mode:                     "
          f"{'OFFLINE_VALIDATION' if args.offline_validation else 'READ_ONLY_RESET_PREVIEW' if args.read_only_preview else 'PREPARE_CONTROLLER_PREVIEW' if args.prepare_controller_preview else 'EXECUTE_WITH_OBJECTS' if object_aware else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print("controller:               direct libfranka 1 kHz clamped-cubic joint trajectory")
    print("timing:                   stationary edges trimmed; active path at 2.0x source time")
    print("collision/reflex settings:UNCHANGED")
    object_setup = (
        "BRACKET MOUNTED + BOLT UPRIGHT/SELF-SUPPORTED"
        if object_aware else "NO OBJECTS"
    )
    hand_sequence = (
        "open during reapproach -> automatic head grasp"
        if object_aware else "DISABLED; Hand must remain open"
    )
    print(f"object setup:             {object_setup}")
    print(f"Hand sequence:            {hand_sequence}")

    loaded = {}
    all_blockers: list[str] = []
    for name in requested_stages:
        stage, source_plan, trajectory = load_stage(config, name)
        blockers = offline_blockers(config, trajectory)
        loaded[name] = (stage, source_plan, trajectory, blockers)
        print_stage(name, stage, source_plan, trajectory, blockers)
        all_blockers.extend(f"{name}: {blocker}" for blocker in blockers)
    if args.offline_validation:
        if all_blockers:
            raise SystemExit("[SAFETY] one or more 2x arm plans failed offline validation")
        print("\nOffline validation passed for every arm stage. No robot or Hand connection was attempted.")
        return

    assert stage_name is not None
    stage, source_plan, trajectory, blockers = loaded[stage_name]
    binary = project_path(config["binary"])
    if not binary.is_file():
        blockers.append("compiled retimed trajectory controller is missing")
    if args.execute and stage.get("execution_enabled") is not True:
        blockers.append("stage execution remains locked pending prepared empty-Hand preview")
    if object_aware:
        if stage_name != "bolt_reposition_regrasp_open":
            blockers.append("object-aware execution is limited to bolt_reposition_regrasp_open")
        if stage.get("object_aware_execution_enabled") is not True:
            blockers.append("object-aware stage execution remains locked")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] 2x arm commissioning refused")

    if object_aware:
        input(
            "Install the bracket and place the bolt upright/self-supported at the commissioned guide location. "
            "Keep the Hand open, do not move the arm, and clear the finger path. "
            "Press Enter to connect read-only (Ctrl+C cancels): "
        )
    else:
        input("Confirm there is NO BRACKET, NO BOLT, the Hand is open, and the local path is clear. Press Enter to connect read-only (Ctrl+C cancels): ")
    from franky import Frame, Gripper, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    current_q = np.asarray(robot.state.q, dtype=np.float64)
    target_q = trajectory.expected_start_q
    start_error = float(np.max(np.abs(current_q - target_q)))
    state = robot.state
    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    reset_q = np.linspace(current_q, target_q, num=201)
    reset_xyz = np.asarray([
        (
            np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
            @ joint7_to_tcp
        )[:3, 3]
        for q in reset_q
    ])
    reset_workspace = config["reset_workspace_bounds_m"]
    reset_workspace_min = np.asarray(reset_workspace["min_xyz"], dtype=float)
    reset_workspace_max = np.asarray(reset_workspace["max_xyz"], dtype=float)
    stage_workspace_min = np.asarray(stage["workspace_bounds_m"]["min_xyz"], dtype=float)
    stage_workspace_max = np.asarray(stage["workspace_bounds_m"]["max_xyz"], dtype=float)
    reset_delta = float(np.max(np.abs(target_q - current_q)))
    reset_segments = max(1, math.ceil(reset_delta / float(config["reset_maximum_joint_delta_rad"])))
    reset_blockers = []
    if grasped or width < float(config["minimum_open_hand_width_m"]):
        reset_blockers.append("Hand is not verifiably empty and open")
    if np.any(reset_xyz < reset_workspace_min) or np.any(reset_xyz > reset_workspace_max):
        reset_blockers.append("current-to-stage-start path leaves the commissioned reset workspace")
    if reset_segments > int(config["reset_maximum_segments"]):
        reset_blockers.append("current-to-stage-start reset needs too many segments")
    if object_aware and start_error > float(config["controller_start_tolerance_rad"]):
        reset_blockers.append(
            "object-aware execution forbids automatic reset; arm is not at the prepared stage start"
        )
    print(f"\nlive Hand width/grasped: {width:.6f} m / {grasped}")
    print(f"live start error:         {start_error:.6f} rad")
    print(f"reset maximum q delta:    {reset_delta:.6f} rad")
    print(f"reset segments:           {reset_segments}")
    print(f"reset xyz min/max:        {reset_xyz.min(0)} / {reset_xyz.max(0)}")
    print(f"reset workspace min/max:  {reset_workspace_min} / {reset_workspace_max}")
    print(f"stage workspace min/max:  {stage_workspace_min} / {stage_workspace_max}")
    if reset_blockers:
        print("\nReset preview blockers:")
        for blocker in reset_blockers:
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        if reset_blockers:
            raise SystemExit(3)
        return
    if reset_blockers:
        raise SystemExit("[SAFETY] prepared controller preview reset refused")

    if args.prepare_controller_preview:
        input("Keep both objects removed and the full reset path clear. Press Enter for the slow segmented arm-only reset: ")
        reset_start = current_q.copy()
        for segment_index in range(1, reset_segments + 1):
            target = reset_start + segment_index / reset_segments * (target_q - reset_start)
            robot.move(JointMotion(
                JointState(target),
                relative_dynamics_factor=float(config["reset_relative_dynamics_factor"]),
            ))
            error = float(np.max(np.abs(np.asarray(robot.state.q) - target)))
            print(f"[auto-reset {segment_index}/{reset_segments}] terminal error={error:.6f} rad")
            if error > float(config["reset_terminal_tolerance_rad"]):
                raise SystemExit("[SAFETY] reset terminal tolerance failed")
    elif start_error > float(config["controller_start_tolerance_rad"]):
        raise SystemExit("[SAFETY] execute requires the already-previewed stage start; run --prepare-controller-preview first")

    del robot
    del gripper
    gc.collect()
    time.sleep(0.5)
    with tempfile.TemporaryDirectory(prefix="fr3_retimed_2x_") as temp:
        plan_path = Path(temp) / "retimed_plan.csv"
        write_retimed_csv(plan_path, trajectory)
        if args.prepare_controller_preview:
            command = [
                str(binary), "read-only-preview", config["robot_ip"],
                str(plan_path), "-", "PREVIEW_ONLY", stage_name,
                *(str(value) for value in stage_workspace_min),
                *(str(value) for value in stage_workspace_max),
            ]
            completed = subprocess.run(command, input="\n", text=True, check=False)
        else:
            output_root = project_path(config["output_dir"])
            output_root.mkdir(parents=True, exist_ok=True)
            trial_label = "object_aware" if object_aware else "empty"
            existing = list(output_root.glob(f"*_{stage_name}_{trial_label}_2x.csv"))
            if len(existing) >= int(config["maximum_physical_trials_per_stage"]):
                raise SystemExit(
                    f"[SAFETY] the authorized {trial_label} physical-trial allowance "
                    "for this stage is exhausted"
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            telemetry = output_root / f"{stamp}_{stage_name}_{trial_label}_2x.csv"
            command = [
                str(binary), "execute", config["robot_ip"], str(plan_path),
                str(telemetry), "RUN_RETIMED_2X_STAGE", stage_name,
                *(str(value) for value in stage_workspace_min),
                *(str(value) for value in stage_workspace_max),
            ]
            completed = subprocess.run(command, input="\n\n", text=True, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if args.prepare_controller_preview:
        print("Prepared empty-Hand controller preview passed. No 2x control session was started.")
    elif object_aware:
        head_grasp = stage["object_aware_head_grasp"]
        gripper = Gripper(config["robot_ip"])
        print("[object-aware] reapproach complete; grasping the stationary bolt head")
        accepted = bool(gripper.grasp(
            float(head_grasp["target_width_m"]),
            float(head_grasp["speed_m_s"]),
            float(head_grasp["force_N"]),
            float(head_grasp["epsilon_inner_m"]),
            float(head_grasp["epsilon_outer_m"]),
        ))
        width = float(gripper.state.width)
        held = bool(gripper.is_grasped)
        width_valid = (
            float(head_grasp["accepted_width_min_m"])
            <= width
            <= float(head_grasp["accepted_width_max_m"])
        )
        print(
            f"[head grasp] accepted/width/grasped: "
            f"{accepted} / {width:.6f} m / {held}"
        )
        print(
            f"[head grasp] accepted width:        "
            f"{float(head_grasp['accepted_width_min_m']):.6f} .. "
            f"{float(head_grasp['accepted_width_max_m']):.6f} m"
        )
        if not (held and width_valid):
            raise SystemExit(
                "[SAFETY] automatic head-grasp validation failed; the Hand remains closed"
            )
        print("[SUCCESS] object-aware 2x reapproach and automatic head grasp completed.")


if __name__ == "__main__":
    main()
