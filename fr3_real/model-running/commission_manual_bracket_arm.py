#!/usr/bin/env python3
"""Commission the arm-only (no Hand action) replay of a manual bracket
episode's IK candidate, one stage at a time.

Mirrors commission_full_assembly_arm_2x.py's guarded flow: offline math-only
validation, a read-only reset preview against the live robot (Hand-open
check, reset workspace/segment check), a prepared controller preview (slow
segmented reset, then the existing fr3_retimed_joint_trajectory binary's own
READ_ONLY_PREVIEW leg -- which re-checks live start error, baseline
stillness, and the predicted path against the guarded caps and stage
workspace bounds), and only then --execute, which is refused unless the
stage's own execution_enabled flag has been deliberately set true and a
prepared preview has already put the arm at the stage start. No gripper/Hand
command is ever sent by this script regardless of --object-aware -- that
flag only changes the confirmation prompt and the trial-count label so a
real-object run isn't blocked by (or silently mixed into) the empty-Hand
validation trial allowance.
"""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_demo.retimed_trajectory import build_retimed_trajectory, write_retimed_csv  # noqa: E402
from prepare_manual_bracket_replay import STAGE_MANUAL_EVENTS, build_manual_stage_plan  # noqa: E402

DEFAULT_CONFIG = FR3_ROOT / "configs" / "bracket_manual_replay_2x_commissioning_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_stage(config: dict, name: str):
    candidate_path = project_path(config["candidate"])
    plan = build_manual_stage_plan(
        candidate_path, name, command_rate_hz=float(config["command_rate_hz"])
    )
    idle_trim = config.get("idle_edge_trim", {})
    trajectory = build_retimed_trajectory(
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
    return config["stages"][name], plan, trajectory


def offline_blockers(config: dict, trajectory) -> list[str]:
    blockers = []
    for label, measured, configured in (
        ("velocity", trajectory.peak_abs_velocity_rad_s, config["velocity_cap_rad_s"]),
        ("acceleration", trajectory.peak_abs_acceleration_rad_s2, config["acceleration_cap_rad_s2"]),
        ("jerk", trajectory.peak_abs_jerk_rad_s3, config["jerk_cap_rad_s3"]),
    ):
        if measured > float(configured):
            blockers.append(f"planned peak {label} {measured:.6f} exceeds cap {float(configured):.6f}")
    if float(np.max(np.abs(trajectory.evaluate(0.0, 1)))) > 1e-10:
        blockers.append("trajectory does not start at zero velocity")
    if float(np.max(np.abs(trajectory.evaluate(trajectory.duration_s, 1)))) > 1e-10:
        blockers.append("trajectory does not end at zero velocity")
    return blockers


def print_stage(name: str, source_plan, trajectory, blockers: list[str]) -> None:
    print(f"\nstage:                    {name}")
    print(f"source/retimed duration:  {source_plan.source_duration_s:.3f} / {trajectory.duration_s:.3f} s")
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
    parser.add_argument("--stage", choices=sorted(STAGE_MANUAL_EVENTS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--object-aware",
        action="store_true",
        help=(
            "the real bracket is physically present (at B1, or already held/placed); "
            "changes the confirmation prompt and tracks physical trials under a separate "
            "'object_aware' label instead of 'empty', so this does not consume or get "
            "blocked by the empty-workspace validation trial allowance"
        ),
    )
    args = parser.parse_args()
    trial_label = "object_aware" if args.object_aware else "empty"
    config = json.loads(args.config.expanduser().resolve().read_text())

    stage_name = args.stage
    requested_stages = list(config["stages"]) if args.offline_validation else [stage_name]
    if not args.offline_validation and stage_name not in config["stages"]:
        raise SystemExit("--stage must name one configured stage")
    # transport_place_closed happens after the bracket has already been grasped
    # (in object-aware mode), so the Hand is expected to be closed and holding
    # it within the calibrated band -- every other stage runs before the grasp
    # or after the release, so the Hand is expected open there instead.
    expect_holding_bracket = args.object_aware and stage_name == "transport_place_closed"

    print("=== MANUAL BRACKET ARM-ONLY COMMISSIONING ===")
    print(f"mode:                     "
          f"{'OFFLINE_VALIDATION' if args.offline_validation else 'READ_ONLY_RESET_PREVIEW' if args.read_only_preview else 'PREPARE_CONTROLLER_PREVIEW' if args.prepare_controller_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"candidate:                {project_path(config['candidate'])}")
    print("controller:               direct libfranka 1 kHz clamped-cubic joint trajectory")
    print(f"timing:                   {config['playback_time_scale']}x source time (this codebase's convention: larger = slower)")
    print("collision/reflex settings:UNCHANGED")
    print(f"object setup:             {'BRACKET PRESENT' if args.object_aware else 'NO OBJECTS'}")
    print(f"Hand sequence:            DISABLED; Hand must remain "
          f"{'closed, holding the bracket' if expect_holding_bracket else 'open'}")

    loaded = {}
    all_blockers: list[str] = []
    for name in requested_stages:
        stage, source_plan, trajectory = load_stage(config, name)
        blockers = offline_blockers(config, trajectory)
        loaded[name] = (stage, source_plan, trajectory, blockers)
        print_stage(name, source_plan, trajectory, blockers)
        all_blockers.extend(f"{name}: {blocker}" for blocker in blockers)
    if args.offline_validation:
        if all_blockers:
            raise SystemExit("[SAFETY] one or more stage plans failed offline validation")
        print("\nOffline validation passed for every stage. No robot or Hand connection was attempted.")
        return

    assert stage_name is not None
    stage, source_plan, trajectory, blockers = loaded[stage_name]
    binary = project_path(config["binary"])
    if not binary.is_file():
        blockers.append("compiled retimed trajectory controller is missing")
    if args.execute and stage.get("execution_enabled") is not True:
        blockers.append("stage execution remains locked pending a prepared controller preview")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] manual bracket arm commissioning refused")

    if expect_holding_bracket:
        input(
            "Confirm the arm is holding the bracket (grasped in a prior step) and the local path "
            "is clear. Press Enter to connect read-only (Ctrl+C cancels): "
        )
    elif args.object_aware:
        input(
            "Confirm the real bracket is present as expected for this stage (at B1, or already "
            "placed/held per the stage being run), the Hand is open, and the local path is clear. "
            "Press Enter to connect read-only (Ctrl+C cancels): "
        )
    else:
        input(
            "Confirm there is NO BRACKET, NO BOLT, the Hand is open, and the local path is clear. "
            "Press Enter to connect read-only (Ctrl+C cancels): "
        )
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
        (np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix) @ joint7_to_tcp)[:3, 3]
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
    if expect_holding_bracket:
        gripper_config = config["gripper"]
        held_min = float(gripper_config["held_width_min_m"])
        held_max = float(gripper_config["held_width_max_m"])
        if not grasped or not (held_min <= width <= held_max):
            reset_blockers.append(
                f"Hand does not verifiably hold the bracket within [{held_min}, {held_max}] m"
            )
    elif grasped or width < float(config["minimum_open_hand_width_m"]):
        reset_blockers.append("Hand is not verifiably empty and open")
    if np.any(reset_xyz < reset_workspace_min) or np.any(reset_xyz > reset_workspace_max):
        reset_blockers.append("current-to-stage-start path leaves the commissioned reset workspace")
    if reset_segments > int(config["reset_maximum_segments"]):
        reset_blockers.append("current-to-stage-start reset needs too many segments")
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
        reset_prompt = (
            "Confirm the reset path to this stage's start is clear of the bracket. "
            "Press Enter for the slow segmented arm-only reset: "
            if args.object_aware
            else "Keep both objects removed and the full reset path clear. Press Enter for the slow segmented arm-only reset: "
        )
        input(reset_prompt)
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
    with tempfile.TemporaryDirectory(prefix="fr3_manual_bracket_") as temp:
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
            # Trials belong to a pickup cell, not to the shared telemetry
            # directory.  Counting every same-stage file here made a completed
            # B1/B83/etc. run consume B44's allowance.
            slot = str(config["bracket_slot"])
            telemetry_suffix = f"_{slot}_{stage_name}_{trial_label}_2x.csv"
            existing = list(output_root.glob(f"*{telemetry_suffix}"))
            if len(existing) >= int(config["maximum_physical_trials_per_stage"]):
                raise SystemExit(
                    f"[SAFETY] the authorized {trial_label} physical-trial allowance "
                    f"for {slot}/{stage_name} is exhausted"
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            telemetry = output_root / f"{stamp}{telemetry_suffix}"
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
        print("Prepared controller preview passed. No live control session was started.")
    else:
        print(f"[DONE] telemetry saved under {config['output_dir']}")


if __name__ == "__main__":
    main()
