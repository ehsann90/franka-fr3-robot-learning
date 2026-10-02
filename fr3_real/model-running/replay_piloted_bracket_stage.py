#!/usr/bin/env python3
"""Preview or execute one guarded arm-only piloted bracket replay stage."""

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

from expert_demo.replay_execution import stage_config_blockers  # noqa: E402
from expert_demo.replay_plan import ReplayPlanError, build_replay_stage  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "full_arm_no_gripper",
            "approach_open",
            "transport_place_closed",
            "retreat_open",
        ),
        default="approach_open",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        plan = build_replay_stage(
            trajectory_dir,
            args.stage,
            command_rate_hz=float(config["replay"]["command_rate_hz"]),
        )
        stage = config["replay"]["stage_execution"][args.stage]
        static_blockers = stage_config_blockers(stage, plan)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[replay] invalid input: {error}") from error

    time_scale = float(stage["playback_time_scale"])
    scaled_dq = plan.dq_targets / time_scale
    expected_duration = plan.source_duration_s * time_scale
    print("=== GUARDED PILOTED BRACKET STAGE REPLAY ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"trajectory:               {trajectory_dir}")
    print(f"stage:                    {plan.name}")
    print(f"targets:                  {len(plan.q_targets)} x 7 at {plan.command_rate_hz:.1f} Hz source")
    print(f"source duration:          {plan.source_duration_s:.3f} s")
    print(f"playback time scale:      {time_scale:.1f}x slower")
    print(f"expected replay duration: {expected_duration:.3f} s")
    print(f"scaled target peak |dq|:  {float(np.max(np.abs(scaled_dq))):.6f} rad/s")
    print(f"relative dynamics factor: {stage['relative_dynamics_factor']}")
    print(f"start tolerance:          {stage['start_joint_tolerance_rad']} rad")
    print(f"workspace bounds:         {stage['workspace_bounds_m']}")
    print("required physical setup:  EMPTY WORKSPACE; NO BRACKET")
    print("camera recording:         DISABLED")
    print("gripper connection:       DISABLED")
    print("gripper commands:         DISABLED")
    if static_blockers:
        print("\nStatic execution blockers:")
        for blocker in static_blockers:
            print(f"  - {blocker}")
        if args.execute:
            raise SystemExit("[SAFETY] stage execution is not commissioned; no robot connection attempted")

    input("Press Enter to connect and validate the live arm/path (Ctrl+C cancels): ")
    from franky import Duration, Frame, JointState, JointWaypoint, JointWaypointMotion, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    start_error = float(np.max(np.abs(current_q - plan.expected_start_q)))
    runtime_blockers = list(static_blockers)
    if start_error > float(stage["start_joint_tolerance_rad"]):
        runtime_blockers.append(
            f"live start error {start_error:.6f} rad exceeds configured tolerance "
            f"{float(stage['start_joint_tolerance_rad']):.6f} rad"
        )

    margin = float(stage["joint_margin_rad"])
    lower = JOINT_MIN + margin
    upper = JOINT_MAX - margin
    if np.any(plan.q_targets < lower) or np.any(plan.q_targets > upper):
        runtime_blockers.append("recorded stage violates the configured guarded joint range")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    predicted_xyz = np.asarray(
        [
            (
                np.asarray(
                    robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix,
                    dtype=np.float64,
                )
                @ joint7_to_tcp
            )[:3, 3]
            for q in plan.q_targets
        ]
    )
    path_errors = np.linalg.norm(predicted_xyz - plan.xyz_targets, axis=1)
    maximum_fk_error = float(path_errors.max())
    if maximum_fk_error > float(stage["maximum_fk_path_error_m"]):
        runtime_blockers.append(
            f"FK path discrepancy {maximum_fk_error:.6f} m exceeds configured maximum "
            f"{float(stage['maximum_fk_path_error_m']):.6f} m"
        )
    workspace = stage["workspace_bounds_m"]
    minimum = np.asarray(workspace["min_xyz"], dtype=np.float64)
    maximum = np.asarray(workspace["max_xyz"], dtype=np.float64)
    if np.any(predicted_xyz < minimum) or np.any(predicted_xyz > maximum):
        runtime_blockers.append("predicted FK path leaves the configured stage workspace")

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"expected start q:         {np.array2string(plan.expected_start_q, precision=5)}")
    print(f"max live start error:     {start_error:.6f} rad")
    print(f"predicted path xyz min:   {np.array2string(predicted_xyz.min(axis=0), precision=5)}")
    print(f"predicted path xyz max:   {np.array2string(predicted_xyz.max(axis=0), precision=5)}")
    print(f"maximum FK path error:    {maximum_fk_error:.6f} m")
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No motion or gripper command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] stage execution refused")

    confirmation_phrase = f"REPLAY_{plan.name.upper()}"
    confirmation = input(
        f"Type {confirmation_phrase} to execute this empty-workspace arm path; anything else aborts: "
    )
    if confirmation != confirmation_phrase:
        print("[replay] aborted without motion")
        return

    waypoint_ms = int(math.ceil(1000.0 * time_scale / plan.command_rate_hz))
    waypoints = [
        JointWaypoint(
            JointState(q, dq),
            minimum_time=Duration(waypoint_ms),
            max_total_duration=Duration(2000),
        )
        for q, dq in zip(plan.q_targets[1:], scaled_dq[1:], strict=True)
    ]
    motion = JointWaypointMotion(
        waypoints,
        relative_dynamics_factor=float(stage["relative_dynamics_factor"]),
    )
    timed_out = threading.Event()

    def stop_robot() -> None:
        timed_out.set()
        robot.stop()

    def interrupt(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop()")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    timer = threading.Timer(float(stage["state_timeout_s"]), stop_robot)
    timer.daemon = True
    timer.start()
    try:
        robot.move(motion)
    except BaseException:
        robot.stop()
        raise
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise SystemExit("[SAFETY] stage timeout fired and robot.stop() was called")
    final_q = np.asarray(robot.state.q, dtype=np.float64)
    final_error = float(np.max(np.abs(final_q - plan.expected_stop_q)))
    print(f"[DONE] final q:           {np.array2string(final_q, precision=5)}")
    print(f"[DONE] terminal q error: {final_error:.6f} rad")
    if final_error > float(stage["terminal_joint_tolerance_rad"]):
        raise SystemExit("[SAFETY] replay ended outside terminal joint tolerance")
    print(f"[DONE] {plan.name} replay passed; no gripper command was issued")


if __name__ == "__main__":
    main()
