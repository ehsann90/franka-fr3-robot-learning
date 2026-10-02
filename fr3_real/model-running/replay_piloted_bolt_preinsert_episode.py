#!/usr/bin/env python3
"""Execute guarded bolt pickup through no-contact PREINSERT; never thread."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        episode = config["replay"]["object_preinsert_execution"]
        rate = float(config["replay"]["command_rate_hz"])
        plans = {
            name: build_bolt_preinsert_stage(str(trajectory_dir), name, command_rate_hz=rate)
            for name in STAGES
        }
        width_calibration = json.loads(
            (FR3_ROOT / episode["gripper"]["held_width_calibration"]).read_text()
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[bolt episode] invalid input: {error}") from error

    blockers: list[str] = []
    if episode.get("enabled") is not True:
        blockers.append("object PREINSERT execution is disabled")
    if episode.get("contact_threading_enabled") is not False:
        blockers.append("contact/threading must remain disabled")
    if any(plan.trajectory_name != episode["trajectory"] for plan in plans.values()):
        blockers.append("episode is not commissioned for this trajectory")
    if width_calibration.get("all_samples_is_grasped") is not True:
        blockers.append("held-width calibration did not verify all samples")
    if abs(
        float(width_calibration["median_width_m"])
        - float(episode["gripper"]["calibrated_median_width_m"])
    ) > 0.000001:
        blockers.append("configured held width does not match calibration")
    if episode["gripper"].get("operator_visual_confirmation_required") is not True:
        blockers.append("operator visual bolt-presence confirmation must be required")

    scale = float(episode["playback_time_scale"])
    peak = max(float(np.max(np.abs(plan.dq_targets))) for plan in plans.values()) / scale
    if peak > float(episode["maximum_scaled_target_velocity_rad_s"]):
        blockers.append("scaled target velocity exceeds configured maximum")
    maximum_step = max(plan.max_target_step_rad for plan in plans.values())
    if maximum_step > float(episode["maximum_target_step_rad"]):
        blockers.append("target joint step exceeds configured maximum")

    print("=== GUARDED BOLT PICKUP TO PREINSERT ===")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"trajectory:               {trajectory_dir}")
    print(f"variation:                {episode['screw_slot']} / {episode['screw_grasp_style']}")
    print(f"playback:                 {scale:.1f}x slower")
    print(f"expected arm duration:    {sum(plan.source_duration_s for plan in plans.values()) * scale:.3f} s")
    print(f"grasp target/speed/force: {episode['gripper']['grasp_target_width_m']:.4f} m / {episode['gripper']['grasp_speed_m_s']:.3f} m/s / {episode['gripper']['grasp_force_n']:.1f} N")
    print("bolt presence validation: is_grasped + width + mandatory operator visual check")
    print("endpoint:                 NO-CONTACT PREINSERT, HAND REMAINS CLOSED")
    print("camera recording:         DISABLED")
    print("contact/threading:        DISABLED")
    if blockers:
        print("Static blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] bolt object episode is not commissioned")

    input("Press Enter to connect and automatically validate/reset the arm and Hand (Ctrl+C cancels): ")
    from franky import (  # noqa: PLC0415
        Duration,
        Frame,
        Gripper,
        JointMotion,
        JointState,
        JointWaypoint,
        JointWaypointMotion,
        Robot,
    )

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    robot.relative_dynamics_factor = float(episode["stage_robot_global_dynamics_factor"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    target_q = plans["approach_open"].expected_start_q
    start_error = float(np.max(np.abs(current_q - target_q)))
    initial_grasped = bool(gripper.is_grasped)
    if initial_grasped:
        blockers.append("Hand reports a held object; remove/release it before automatic reset")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    all_q = np.concatenate([plans[name].q_targets for name in STAGES])
    all_xyz = np.concatenate([plans[name].xyz_targets for name in STAGES])
    predicted_xyz = np.asarray([
        (
            np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
            @ joint7_to_tcp
        )[:3, 3]
        for q in all_q
    ])
    fk_error = float(np.max(np.linalg.norm(predicted_xyz - all_xyz, axis=1)))
    if fk_error > float(episode["maximum_fk_path_error_m"]):
        blockers.append("full replay FK discrepancy exceeds configured maximum")
    margin = float(episode["joint_margin_rad"])
    if np.any(all_q < JOINT_MIN + margin) or np.any(all_q > JOINT_MAX - margin):
        blockers.append("full replay violates guarded joint range")
    minimum = np.asarray(episode["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(episode["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(predicted_xyz < minimum) or np.any(predicted_xyz > maximum):
        blockers.append("full replay leaves configured workspace")

    reset = episode["automatic_reset_to_start"]
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
    print(f"live Hand width/grasped:  {float(gripper.state.width):.6f} m / {initial_grasped}")
    print(f"maximum replay FK error: {fk_error:.6f} m")
    print(f"replay xyz min/max:       {np.array2string(predicted_xyz.min(0), precision=5)} / {np.array2string(predicted_xyz.max(0), precision=5)}")
    if start_error > float(episode["start_joint_tolerance_rad"]):
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
    if blockers:
        print("Execution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] bolt object episode refused")

    stop_requested = threading.Event()
    may_be_holding = False

    def stop_all() -> None:
        stop_requested.set()
        robot.stop()
        try:
            gripper.stop()
        except BaseException:
            pass

    def interrupt(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] stopping arm and Hand")
        stop_all()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    if start_error > float(episode["start_joint_tolerance_rad"]):
        input("Remove the bolt from the Hand and S1, clear the path, then press Enter for slow reset: ")
        robot.move(
            JointMotion(
                JointState(target_q),
                relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
            )
        )
        reset_error = float(np.max(np.abs(np.asarray(robot.state.q) - target_q)))
        print(f"[auto-reset] error={reset_error:.6f} rad")
        if reset_error > float(reset["terminal_joint_tolerance_rad"]):
            raise SystemExit("[SAFETY] reset terminal tolerance failed")

    robot.relative_dynamics_factor = float(episode["stage_robot_global_dynamics_factor"])

    def run_stage(name: str, monitor_grasp: bool) -> None:
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
        lost = threading.Event()
        monitor_stop = threading.Event()

        def monitor() -> None:
            period = 1.0 / float(episode["gripper"]["monitor_hz"])
            while not monitor_stop.wait(period):
                try:
                    if not bool(gripper.is_grasped):
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
        try:
            robot.move(
                JointWaypointMotion(
                    waypoints,
                    relative_dynamics_factor=float(episode["relative_dynamics_factor"]),
                )
            )
        finally:
            timer.cancel()
            monitor_stop.set()
            if monitor_grasp:
                thread.join(timeout=1.0)
        if stop_requested.is_set() or lost.is_set():
            raise RuntimeError(f"{name} stopped or lost the grasp")
        error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_stop_q)))
        print(f"[{name}] terminal error={error:.6f} rad")
        if error > float(episode["terminal_joint_tolerance_rad"]):
            raise RuntimeError(f"{name} terminal tolerance failed")

    try:
        print(f"\nPlace the bolt precisely at {episode['screw_slot']} in the recorded orientation.")
        input("Press Enter when S1 and the complete path are ready: ")
        run_stage("approach_open", monitor_grasp=False)
        print("[bolt] grasping while the arm is stationary")
        accepted = gripper.grasp(
            float(episode["gripper"]["grasp_target_width_m"]),
            float(episode["gripper"]["grasp_speed_m_s"]),
            float(episode["gripper"]["grasp_force_n"]),
            float(episode["gripper"]["epsilon_inner_m"]),
            float(episode["gripper"]["epsilon_outer_m"]),
        )
        may_be_holding = True
        width = float(gripper.state.width)
        grasped = bool(gripper.is_grasped)
        print(f"[bolt grasp] accepted={accepted} is_grasped={grasped} width={width:.6f} m")
        if not accepted or not grasped:
            raise RuntimeError("Hand did not report a successful bolt grasp")
        if width > float(episode["gripper"]["maximum_accepted_held_width_m"]):
            raise RuntimeError("held width is inconsistent with the calibrated bolt grasp")
        visual = input(
            "Visually confirm the bolt is securely present and correctly held. "
            "Press Enter to transport; type anything to abort while keeping the Hand closed: "
        ).strip()
        if visual:
            raise RuntimeError("operator rejected visual bolt-presence check")
        run_stage("transport_preinsert_closed", monitor_grasp=True)
    except BaseException:
        stop_all()
        if may_be_holding:
            print("[SAFETY] The Hand may still hold the bolt; it was NOT automatically opened.")
        raise
    print("[SUCCESS] bolt reached NO-CONTACT PREINSERT; Hand remains closed")
    print("No contact, insertion, rotation, release, or threading command was issued.")
    print("Use supported Pilot controls to retract and return the bolt safely.")


if __name__ == "__main__":
    main()
