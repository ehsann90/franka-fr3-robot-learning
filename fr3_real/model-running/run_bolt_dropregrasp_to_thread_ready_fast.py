#!/usr/bin/env python3
"""Guarded, fast (production-speed) continuation from no-contact preinsert to
THREAD_READY: lower/insert -> confirm self-support -> release -> reapproach ->
empty-wrist prewind -> head regrasp -> confirm THREAD_READY.

Reuses load_inputs() from record_autonomous_bolt_engagement_demo.py (the
exact same config loading/plan building the full orchestrator uses) so the
trims, trajectory sources, and grasp parameters are identical -- only the
--full-assembly-only speed restriction and the bracket-redo-from-scratch
behavior are avoided. Stops at THREAD_READY, deliberately before the
force-controlled threading drives, which are a separate operation.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_demo.retimed_trajectory import write_retimed_csv  # noqa: E402
from record_autonomous_bolt_engagement_demo import (  # noqa: E402
    DEFAULT_RECORDING_CONFIG,
    build_configured_accelerated_trajectory,
    load_inputs,
    pickup_grasp_evidence,
)

DEFAULT_SPEED_CONFIG = FR3_ROOT / "configs" / "full_assembly_4x_production_v1.json"
WORKSPACE_MIN = [0.57, 0.30, 0.08]
WORKSPACE_MAX = [0.63, 0.35, 0.15]
OUTPUT_ROOT = "recordings/bolt_dropregrasp_fast_trials_v1"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
RESET_JOINT_MARGIN_RAD = 0.12
RESET_RELATIVE_DYNAMICS_FACTOR = 0.05
RESET_MAXIMUM_JOINT_DELTA_RAD = 0.5
RESET_TERMINAL_TOLERANCE_RAD = 0.01
# This is the previously used clear-arm envelope for the high-clearance
# bracket-replay setup.  It deliberately includes the current elevated
# hand-guided pose as well as the no-contact PREINSERT endpoint.  It is a
# TCP-only guard; the operator must still clear the held-bolt swept path.
HELD_RESET_WORKSPACE_MIN = np.asarray([0.21, -0.50, 0.005], dtype=np.float64)
HELD_RESET_WORKSPACE_MAX = np.asarray([0.70, 0.35, 0.32], dtype=np.float64)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def run_binary_stage(binary: Path, robot_ip: str, trajectory, stage_key: str, telemetry_dir: Path) -> None:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    telemetry = telemetry_dir / f"{stamp}_{stage_key}_object_aware_4x.csv"
    with tempfile.TemporaryDirectory(prefix="fr3_bolt_dropregrasp_fast_") as temp:
        plan_path = Path(temp) / "retimed_plan.csv"
        write_retimed_csv(plan_path, trajectory)
        command = [
            str(binary), "execute", robot_ip, str(plan_path), str(telemetry),
            "RUN_RETIMED_2X_STAGE", stage_key,
            *(str(value) for value in WORKSPACE_MIN),
            *(str(value) for value in WORKSPACE_MAX),
        ]
        completed = subprocess.run(command, input="\n\n", text=True, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"[DONE] {stage_key} telemetry saved: {telemetry}")


def prewind_and_head_grasp(robot, gripper, task: dict, head_grasp: dict) -> None:
    """Empty-wrist prewind -> head grasp -> mandatory THREAD_READY confirmation."""
    import numpy as np
    from franky import JointMotion, JointState  # noqa: PLC0415

    prewind_q = np.array(robot.state.q, dtype=np.float64, copy=True)
    prewind_start_q7 = float(prewind_q[6])
    prewind_q[6] = float(task["engagement_start_q7_rad"])
    if not -0.156888 <= prewind_q[6] <= 2.998908:
        raise RuntimeError("empty-wrist prewind target leaves the cable-safe q7 interval")
    print(f"[prewind] rotating the OPEN empty wrist q7 {prewind_start_q7:+.6f} -> {prewind_q[6]:+.6f} rad")
    robot.move(JointMotion(
        JointState(prewind_q),
        relative_dynamics_factor=float(task["empty_wrist_prewind_relative_dynamics_factor"]),
    ))
    prewind_error = float(np.max(np.abs(np.asarray(robot.state.q) - prewind_q)))
    if prewind_error > 0.005:
        raise RuntimeError("empty-wrist prewind terminal tolerance failed")
    print(f"[prewind] terminal error={prewind_error:.6f} rad")

    accepted = gripper.grasp(
        float(head_grasp["target_width_m"]),
        float(head_grasp["speed_m_s"]),
        float(head_grasp["force_N"]),
        float(head_grasp["epsilon_inner_m"]),
        float(head_grasp["epsilon_outer_m"]),
    )
    held_width = float(gripper.state.width)
    held_grasped = bool(gripper.is_grasped)
    strict, physical_evidence = check_head_grasp(
        head_grasp, accepted=bool(accepted), is_grasped=held_grasped, width=held_width
    )
    print(f"[head grasp] accepted={accepted} is_grasped={held_grasped} width={held_width:.6f} m "
          f"strict={strict} physical_evidence={physical_evidence}")
    if not (strict or physical_evidence):
        raise RuntimeError(
            "[SAFETY] head grasp did not pass either the strict or bounded-physical check; "
            "Hand was NOT automatically reopened -- visually verify and clear manually"
        )
    if not strict:
        print("[head grasp] Franka's strict boolean disagrees with the bounded physical evidence; "
              "continuing on operator confirmation")

    if input(
        "Confirm THREAD_READY: upright, securely head-held, and unloaded. Press Enter to accept; "
        "type anything to reject (Hand stays closed): "
    ):
        raise SystemExit("[bolt regrasp] operator rejected THREAD_READY; Hand stays closed")

    print("\n[THREAD_READY] bolt is upright, head-held, and confirmed. No threading command was issued.")
    print("No contact, rotation, release, or threading command was issued by this script.")


def check_head_grasp(head_grasp: dict, *, accepted: bool, is_grasped: bool, width: float) -> tuple[bool, bool]:
    """Mirrors check_head_grasp_result() in record_autonomous_bolt_engagement_demo.py."""
    strict_width = float(head_grasp["accepted_width_min_m"]) <= width <= float(head_grasp["accepted_width_max_m"])
    accepted_ok = accepted is not False
    strict = bool(accepted_ok and is_grasped and strict_width)
    physical_sanity = (
        float(head_grasp["operator_confirmed_width_min_m"]) <= width
        <= float(head_grasp["operator_confirmed_width_max_m"])
    )
    bounded_obstruction = (
        float(head_grasp["physical_obstruction_width_min_m"]) <= width
        <= float(head_grasp["physical_obstruction_width_max_m"])
    )
    physical_evidence = bool(physical_sanity and ((accepted_ok and is_grasped) or bounded_obstruction))
    return strict, physical_evidence


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--reset-to-start", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--lower-only",
        action="store_true",
        help=(
            "execute only lower_held and stop with the bolt still held at the "
            "recorded pre-release pose"
        ),
    )
    modes.add_argument(
        "--resume-head-grasp",
        action="store_true",
        help=(
            "skip lower_held/release/reposition (already completed) and resume from "
            "exactly after bolt_reposition_regrasp_open: prewind -> head grasp -> "
            "THREAD_READY confirm. Requires the arm already at that stage's expected "
            "stop pose with the Hand open."
        ),
    )
    parser.add_argument("--speed-config", type=Path, default=DEFAULT_SPEED_CONFIG)
    parser.add_argument("--recording-config", type=Path, default=DEFAULT_RECORDING_CONFIG)
    args = parser.parse_args()

    speed_config = json.loads(args.speed_config.expanduser().resolve().read_text())
    collection, task, preinsert, drop, engagement, commissioning, plans = load_inputs(
        args.recording_config
    )
    del engagement, commissioning
    robot_ip = collection["robot_ip"]
    binary = project_path(speed_config["binary"])
    pickup_grasp = task["pickup_shaft_grasp"]
    head_grasp = task["thread_ready_head_grasp"]
    open_speed = float(drop["gripper"]["open_speed_m_s"])
    minimum_open_width = float(drop["gripper"]["minimum_open_width_m"])

    lower_trajectory = build_configured_accelerated_trajectory(
        plans["drop"]["lower_held"], speed_config, "bolt_lower_held"
    )
    reposition_trajectory = build_configured_accelerated_trajectory(
        plans["drop"]["reposition_regrasp_open"], speed_config, "bolt_reposition_regrasp_open"
    )

    blockers = []
    for stage_key, trajectory in (
        ("bolt_lower_held", lower_trajectory),
        ("bolt_reposition_regrasp_open", reposition_trajectory),
    ):
        authorized = speed_config["stages"].get(stage_key, {}).get("production_execution_authorized")
        if authorized is not True:
            blockers.append(f"{stage_key} is not production_execution_authorized")
        for label, measured, cap in (
            ("velocity", trajectory.peak_abs_velocity_rad_s, speed_config["velocity_cap_rad_s"]),
            ("acceleration", trajectory.peak_abs_acceleration_rad_s2, speed_config["acceleration_cap_rad_s2"]),
            ("jerk", trajectory.peak_abs_jerk_rad_s3, speed_config["jerk_cap_rad_s3"]),
        ):
            if measured > float(cap):
                blockers.append(f"{stage_key} planned peak {label} exceeds cap")
    if not binary.is_file():
        blockers.append(f"missing compiled controller: {binary}")

    print("=== FAST BOLT DROP/REGRASP TO THREAD_READY ===")
    print(f"mode:                     "
          f"{'EXECUTE_LOWER_ONLY' if args.execute and args.lower_only else 'EXECUTE' if args.execute else 'RESET_TO_START' if args.reset_to_start else 'RESUME_HEAD_GRASP' if args.resume_head_grasp else 'READ_ONLY_PREVIEW'}")
    print(f"robot IP:                 {robot_ip}")
    print(f"speed config:             {args.speed_config.name}")
    print(f"lower_held duration:      {lower_trajectory.duration_s:.3f} s")
    print(f"reposition duration:      {reposition_trajectory.duration_s:.3f} s")
    print(f"head grasp target/speed/force: {head_grasp['target_width_m']:.4f} m / "
          f"{head_grasp['speed_m_s']:.3f} m/s / {head_grasp['force_N']:.1f} N")
    print(f"engagement start q7:      {float(task['engagement_start_q7_rad']):+.6f} rad")
    print(
        "threading drives:         NOT RUN BY THIS SCRIPT -- "
        + ("stops before release" if args.lower_only else "stops at THREAD_READY")
    )
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] fast drop/regrasp refused")

    input("Press Enter to connect and inspect live state (Ctrl+C cancels; no command issued yet): ")
    from franky import Frame, Gripper, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(robot_ip)
    gripper = Gripper(robot_ip)
    width = float(gripper.state.width)
    is_grasped = bool(gripper.is_grasped)
    strict, bounded = pickup_grasp_evidence(
        pickup_grasp, accepted=is_grasped, is_grasped=is_grasped, width_m=width
    )
    current_q = np.asarray(robot.state.q, dtype=np.float64)
    start_error = float(np.max(np.abs(current_q - lower_trajectory.expected_start_q)))
    print(f"\nlive Hand width/grasped: {width:.6f} m / {is_grasped}")
    print(f"strict/bounded pickup hold: {strict} / {bounded}")
    print(f"live start error:        {start_error:.6f} rad (vs bolt_lower_held start)")

    if args.reset_to_start:
        live_blockers = []
        if not (strict or bounded):
            live_blockers.append("Hand does not show evidence of holding the bolt; refusing to reset")
        reset_delta = float(np.max(np.abs(lower_trajectory.expected_start_q - current_q)))
        reset_segments = max(1, math.ceil(reset_delta / RESET_MAXIMUM_JOINT_DELTA_RAD))
        lower = JOINT_MIN + RESET_JOINT_MARGIN_RAD
        upper = JOINT_MAX - RESET_JOINT_MARGIN_RAD
        reset_path = np.linspace(current_q, lower_trajectory.expected_start_q, num=201)
        if np.any(reset_path < lower) or np.any(reset_path > upper):
            live_blockers.append("reset path violates the guarded joint range")
        state = robot.state
        measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
        joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
        joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
        reset_xyz = np.asarray(
            [
                (
                    np.asarray(
                        robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix,
                        dtype=np.float64,
                    )
                    @ joint7_to_tcp
                )[:3, 3]
                for q in reset_path
            ]
        )
        if np.any(reset_xyz < HELD_RESET_WORKSPACE_MIN) or np.any(
            reset_xyz > HELD_RESET_WORKSPACE_MAX
        ):
            live_blockers.append("predicted held-bolt reset TCP path leaves the guarded workspace")
        print(f"reset maximum q delta:   {reset_delta:.6f} rad")
        print(f"reset segments:          {reset_segments}")
        print(
            "reset TCP xyz min/max:   "
            f"{np.array2string(reset_xyz.min(axis=0), precision=5)} / "
            f"{np.array2string(reset_xyz.max(axis=0), precision=5)}"
        )
        print(
            "reset TCP workspace:     "
            f"{np.array2string(HELD_RESET_WORKSPACE_MIN, precision=3)} / "
            f"{np.array2string(HELD_RESET_WORKSPACE_MAX, precision=3)}"
        )
        if live_blockers:
            print("\nBlockers:")
            for blocker in live_blockers:
                print(f"  - {blocker}")
            raise SystemExit("[SAFETY] reset refused")
        input(
            "Confirm the held bolt, Hand, frame, and full reset path are clear. "
            "Press Enter for the slow segmented reset (Ctrl+C cancels): "
        )
        reset_start = current_q.copy()
        for segment_index in range(1, reset_segments + 1):
            target = reset_start + segment_index / reset_segments * (
                lower_trajectory.expected_start_q - reset_start
            )
            robot.move(JointMotion(JointState(target), relative_dynamics_factor=RESET_RELATIVE_DYNAMICS_FACTOR))
            error = float(np.max(np.abs(np.asarray(robot.state.q) - target)))
            print(f"[reset {segment_index}/{reset_segments}] terminal error={error:.6f} rad")
            if error > RESET_TERMINAL_TOLERANCE_RAD:
                raise SystemExit("[SAFETY] reset terminal tolerance failed")
        print("[DONE] reset complete. No Hand command was issued.")
        return

    if args.resume_head_grasp:
        resume_error = float(
            np.max(np.abs(current_q - reposition_trajectory.expected_stop_q))
        )
        print(f"live error vs reposition_regrasp_open stop: {resume_error:.6f} rad")
        resume_blockers = []
        if is_grasped or width < minimum_open_width:
            resume_blockers.append(
                "Hand does not read empty/open; this resume point expects the bolt "
                "already released and standing at the guide"
            )
        if resume_error > 0.02:
            resume_blockers.append(
                f"live arm is not at bolt_reposition_regrasp_open's stop pose "
                f"({resume_error:.6f} rad); this resume mode does not run any transport"
            )
        if resume_blockers:
            print("\nBlockers:")
            for blocker in resume_blockers:
                print(f"  - {blocker}")
            raise SystemExit("[SAFETY] resume refused")
        prewind_and_head_grasp(robot, gripper, task, head_grasp)
        return

    live_blockers = []
    if not (strict or bounded):
        live_blockers.append("Hand does not show evidence of holding the bolt")
    if start_error > 0.05:
        live_blockers.append(
            f"live arm is not close enough to bolt_lower_held's start ({start_error:.6f} rad); "
            "run --reset-to-start first"
        )
    if live_blockers:
        print("\nBlockers:")
        for blocker in live_blockers:
            print(f"  - {blocker}")
        if args.read_only_preview:
            raise SystemExit(3)
        raise SystemExit("[SAFETY] fast drop/regrasp refused")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return

    run_binary_stage(binary, robot_ip, lower_trajectory, "bolt_lower_held", project_path(OUTPUT_ROOT))

    if args.lower_only:
        print("[SUCCESS] bolt remains securely held at the recorded pre-release/drop pose")
        print("No Hand-open, release, reposition, rotation, force-control, or threading command was issued.")
        print("Use Pilot controls to retreat safely after marking or inspecting the hole location.")
        return

    if input(
        "Visually confirm the bolt is upright and self-supported. Press Enter to release; "
        "type anything to reject: "
    ):
        raise SystemExit("[bolt drop] operator rejected self-support check; Hand stays closed")

    gripper.open(open_speed)
    width = float(gripper.state.width)
    if width < minimum_open_width:
        raise RuntimeError("Hand failed to open at the guide")
    print(f"[bolt drop] released; width={width:.6f} m")

    if input("Confirm the bolt remained upright. Press Enter to reapproach; type anything to reject: "):
        raise SystemExit("[bolt drop] operator rejected upright bolt after release")

    run_binary_stage(
        binary, robot_ip, reposition_trajectory, "bolt_reposition_regrasp_open", project_path(OUTPUT_ROOT)
    )

    prewind_and_head_grasp(robot, gripper, task, head_grasp)


if __name__ == "__main__":
    main()
