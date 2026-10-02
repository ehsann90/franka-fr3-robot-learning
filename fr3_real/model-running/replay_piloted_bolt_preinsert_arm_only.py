#!/usr/bin/env python3
"""Auto-reset and replay the bolt pre-grasp or pickup-through-PREINSERT arm path without a bolt."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_preinsert import (  # noqa: E402
    BOLT_PREINSERT_STAGE_EVENTS,
    build_bolt_preinsert_stage,
)
from expert_demo.object_episode_execution import automatic_reset_blockers  # noqa: E402
from expert_demo.replay_plan import ReplayPlanError  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bolt_preinsert_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
STAGES = tuple(BOLT_PREINSERT_STAGE_EVENTS)


def static_blockers(profile: dict, plans: dict) -> list[str]:
    blockers: list[str] = []
    if profile.get("enabled") is not True:
        blockers.append("arm-only commissioning is disabled")
    if profile.get("empty_workspace_only") is not True:
        blockers.append("arm-only commissioning must require an empty workspace")
    if profile.get("gripper_commands_enabled") is not False:
        blockers.append("gripper commands must remain disabled")
    trajectory = profile.get("trajectory")
    if any(plan.trajectory_name != trajectory for plan in plans.values()):
        blockers.append("profile is not commissioned for this trajectory")
    scale = profile.get("playback_time_scale")
    if not isinstance(scale, (int, float)) or float(scale) <= 0:
        blockers.append("playback_time_scale must be positive")
        return blockers
    peak = max(float(np.max(np.abs(plan.dq_targets))) for plan in plans.values()) / float(scale)
    if peak > float(profile["maximum_scaled_target_velocity_rad_s"]):
        blockers.append("scaled target velocity exceeds configured maximum")
    step = max(plan.max_target_step_rad for plan in plans.values())
    if step > float(profile["maximum_target_step_rad"]):
        blockers.append("target position step exceeds configured maximum")
    minimum = np.asarray(profile["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(profile["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    for plan in plans.values():
        if np.any(plan.xyz_targets < minimum) or np.any(plan.xyz_targets > maximum):
            blockers.append(f"recorded {plan.name} leaves configured workspace")
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--approach-only",
        action="store_true",
        help=(
            "stop at the stationary open-Hand bolt pickup pose immediately before "
            "the grasp; do not run the transport-to-preinsert stage"
        ),
    )
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    selected_stages = ("approach_open",) if args.approach_only else STAGES
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        profile = config["replay"]["arm_only_commissioning"]
        rate = float(config["replay"]["command_rate_hz"])
        plans = {
            name: build_bolt_preinsert_stage(str(trajectory_dir), name, command_rate_hz=rate)
            for name in selected_stages
        }
        blockers = static_blockers(profile, plans)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[bolt arm replay] invalid input: {error}") from error

    scale = float(profile["playback_time_scale"])
    print("=== BOLT PREINSERT ARM-ONLY COMMISSIONING ===")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"trajectory:               {trajectory_dir}")
    print(f"stages:                   {' -> '.join(selected_stages)}")
    print(f"playback:                 {scale:.1f}x slower")
    print(f"expected arm duration:    {sum(p.source_duration_s for p in plans.values()) * scale:.3f} s")
    print("physical setup:           NO BOLT; bracket may remain installed")
    print("Hand connection/commands: DISABLED")
    print("contact/threading:        DISABLED")
    if blockers:
        print("Static blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] arm-only replay is not commissioned")

    input("Press Enter to connect and automatically validate/reset the arm (Ctrl+C cancels): ")
    from franky import (  # noqa: PLC0415
        Duration,
        Frame,
        JointMotion,
        JointState,
        JointWaypoint,
        JointWaypointMotion,
        Robot,
    )

    robot = Robot(config["robot_ip"])
    robot.relative_dynamics_factor = float(profile["stage_robot_global_dynamics_factor"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    target_q = plans["approach_open"].expected_start_q
    start_error = float(np.max(np.abs(current_q - target_q)))
    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)

    all_q = np.concatenate([plans[name].q_targets for name in selected_stages])
    all_xyz = np.concatenate([plans[name].xyz_targets for name in selected_stages])
    predicted_xyz = np.asarray([
        (
            np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
            @ joint7_to_tcp
        )[:3, 3]
        for q in all_q
    ])
    runtime_blockers = list(blockers)
    fk_error = float(np.max(np.linalg.norm(predicted_xyz - all_xyz, axis=1)))
    if fk_error > float(profile["maximum_fk_path_error_m"]):
        runtime_blockers.append("full replay FK discrepancy exceeds configured maximum")
    margin = float(profile["joint_margin_rad"])
    if np.any(all_q < JOINT_MIN + margin) or np.any(all_q > JOINT_MAX - margin):
        runtime_blockers.append("full replay violates guarded joint range")

    reset = profile["automatic_reset_to_start"]
    reset_q = np.linspace(current_q, target_q, num=101)
    reset_xyz = np.asarray([
        (
            np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
            @ joint7_to_tcp
        )[:3, 3]
        for q in reset_q
    ])
    reset_fk_error = float(np.linalg.norm(reset_xyz[-1] - plans["approach_open"].xyz_targets[0]))
    reset_blockers = automatic_reset_blockers(
        {**reset, "requires_empty_gripper": True, "requires_object_removed": True},
        current_q,
        target_q,
        reset_xyz,
        reset_fk_error,
    )
    reset_margin = float(reset["joint_margin_rad"])
    if np.any(reset_q < JOINT_MIN + reset_margin) or np.any(reset_q > JOINT_MAX - reset_margin):
        reset_blockers.append("automatic reset violates guarded joint range")

    print(f"\nlive start error:         {start_error:.6f} rad")
    print(f"maximum replay FK error: {fk_error:.6f} m")
    print(f"replay xyz min:           {np.array2string(predicted_xyz.min(axis=0), precision=5)}")
    print(f"replay xyz max:           {np.array2string(predicted_xyz.max(axis=0), precision=5)}")
    if start_error > float(profile["start_joint_tolerance_rad"]):
        print("automatic reset:          REQUIRED")
        print(f"reset max joint delta:    {float(np.max(np.abs(target_q-current_q))):.6f} rad")
        print(f"reset xyz min/max:        {np.array2string(reset_xyz.min(0), precision=5)} / {np.array2string(reset_xyz.max(0), precision=5)}")
        if reset_blockers:
            print("Automatic reset blockers:")
            for blocker in dict.fromkeys(reset_blockers):
                print(f"  - {blocker}")
            raise SystemExit("[SAFETY] automatic reset refused")
    else:
        print("automatic reset:          NOT NEEDED")
    if runtime_blockers:
        print("Execution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] arm-only replay refused")

    stop_requested = threading.Event()

    def stop_robot() -> None:
        stop_requested.set()
        robot.stop()

    def interrupt(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] stopping arm")
        stop_robot()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    if start_error > float(profile["start_joint_tolerance_rad"]):
        input("Confirm there is NO BOLT in the Hand or S1, clear the path, then press Enter for slow reset: ")
        robot.move(
            JointMotion(
                JointState(target_q),
                relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
            )
        )
        final_reset_error = float(np.max(np.abs(np.asarray(robot.state.q) - target_q)))
        print(f"[auto-reset] error={final_reset_error:.6f} rad")
        if final_reset_error > float(reset["terminal_joint_tolerance_rad"]):
            raise SystemExit("[SAFETY] reset terminal tolerance failed")

    robot.relative_dynamics_factor = float(profile["stage_robot_global_dynamics_factor"])
    input("Keep S1 empty and press Enter to replay approach_open with no Hand command: ")
    for index, name in enumerate(selected_stages):
        plan = plans[name]
        waypoint_ms = int(math.ceil(1000.0 * scale / plan.command_rate_hz))
        waypoints = [
            JointWaypoint(
                JointState(q, dq / scale),
                minimum_time=Duration(waypoint_ms),
                max_total_duration=Duration(2000),
            )
            for q, dq in zip(plan.q_targets[1:], plan.dq_targets[1:], strict=True)
        ]
        timer = threading.Timer(float(profile["state_timeout_s"]), stop_robot)
        timer.daemon = True
        timer.start()
        try:
            robot.move(
                JointWaypointMotion(
                    waypoints,
                    relative_dynamics_factor=float(profile["relative_dynamics_factor"]),
                )
            )
        finally:
            timer.cancel()
        if stop_requested.is_set():
            raise SystemExit(f"[SAFETY] {name} timed out or was stopped")
        terminal_error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_stop_q)))
        print(f"[{name}] terminal error={terminal_error:.6f} rad")
        if terminal_error > float(profile["terminal_joint_tolerance_rad"]):
            raise SystemExit(f"[SAFETY] {name} terminal tolerance failed")
        if index == 0 and len(selected_stages) > 1:
            input("At empty grasp pose. Press Enter to continue arm-only transport to NO-CONTACT PREINSERT: ")
    if args.approach_only:
        print("[SUCCESS] arm is stationary at the open-Hand bolt pickup pose immediately before grasp")
        print("No bolt, Hand, contact, or threading command was issued. Use Pilot controls to retreat when done.")
    else:
        print("[SUCCESS] bolt pickup-through-PREINSERT arm path passed with no bolt or Hand command")
        print("Use Pilot controls to retreat from PREINSERT after reviewing the result.")


if __name__ == "__main__":
    main()
