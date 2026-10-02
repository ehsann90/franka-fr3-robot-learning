#!/usr/bin/env python3
"""Guarded bolt drop/regrasp episode ending at THREAD_READY; never rotate."""

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

from expert_demo.bolt_drop_regrasp import (  # noqa: E402
    BOLT_DROP_REGRASP_STAGE_EVENTS,
    build_bolt_drop_regrasp_stage,
)
from expert_demo.replay_plan import ReplayPlanError  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bolt_drop_regrasp_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
STAGES = tuple(BOLT_DROP_REGRASP_STAGE_EVENTS)


def static_blockers(episode: dict, plans: dict, width_calibration: dict) -> list[str]:
    blockers: list[str] = []
    if episode.get("enabled") is not True:
        blockers.append("object normalization execution is disabled")
    if episode.get("contact_threading_enabled") is not False:
        blockers.append("contact/threading must remain disabled")
    if episode.get("camera_recording_enabled") is not False:
        blockers.append("camera recording must remain disabled for the first object run")
    if any(plan.trajectory_name != episode.get("trajectory") for plan in plans.values()):
        blockers.append("profile is not commissioned for this capture")
    if width_calibration.get("all_samples_is_grasped") is not True:
        blockers.append("held-width calibration did not verify every sample")
    if abs(
        float(width_calibration["median_width_m"])
        - float(episode["gripper"]["calibrated_median_width_m"])
    ) > 0.000001:
        blockers.append("configured held width does not match calibration")
    if episode["gripper"].get("operator_visual_confirmation_required") is not True:
        blockers.append("mandatory operator visual confirmation is not enabled")
    scale = float(episode["playback_time_scale"])
    peak = max(float(np.max(np.abs(plan.dq_targets))) for plan in plans.values()) / scale
    if peak > float(episode["maximum_scaled_target_velocity_rad_s"]):
        blockers.append("scaled target velocity exceeds configured maximum")
    if max(plan.max_target_step_rad for plan in plans.values()) > float(
        episode["maximum_target_step_rad"]
    ):
        blockers.append("target joint step exceeds configured maximum")
    minimum = np.asarray(episode["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(episode["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if any(np.any(plan.xyz_targets < minimum) or np.any(plan.xyz_targets > maximum) for plan in plans.values()):
        blockers.append("recorded arm path leaves configured workspace")
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        episode = config["replay"]["object_normalization_execution"]
        rate = float(config["replay"]["command_rate_hz"])
        trims = config["replay"]["stage_stationary_trim_s"]
        plans = {
            name: build_bolt_drop_regrasp_stage(
                str(trajectory_dir),
                name,
                command_rate_hz=rate,
                trim_start_s=float(trims[name]["start_s"]),
                trim_end_s=float(trims[name]["end_s"]),
            )
            for name in STAGES
        }
        width_calibration = json.loads(
            (FR3_ROOT / episode["gripper"]["held_width_calibration"]).read_text()
        )
        blockers = static_blockers(episode, plans, width_calibration)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[drop/regrasp episode] invalid input: {error}") from error

    scale = float(episode["playback_time_scale"])
    gripper_config = episode["gripper"]
    print("=== GUARDED BOLT DROP + HEAD REGRASP ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"trajectory:               {trajectory_dir}")
    print(f"playback:                 {scale:.1f}x slower")
    print(f"expected arm duration:    {sum(p.source_duration_s for p in plans.values()) * scale:.3f} s")
    print("required start:           BOLT HELD at validated NO-CONTACT PREINSERT")
    print("sequence:                 lower held -> verify support -> open -> reposition -> regrasp head")
    print("endpoint:                 THREAD_READY; HAND CLOSED")
    print("camera recording:         DISABLED")
    print("contact/rotation/threading: DISABLED")
    if blockers:
        print("Static blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] object normalization episode is not commissioned")

    input("Press Enter to connect and validate live arm and Hand state (Ctrl+C cancels): ")
    from franky import Duration, Frame, Gripper, JointState, JointWaypoint, JointWaypointMotion, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    target_q = plans["lower_held"].expected_start_q
    start_error = float(np.max(np.abs(current_q - target_q)))
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    if start_error > float(episode["start_joint_tolerance_rad"]):
        blockers.append(
            f"live start error {start_error:.6f} rad exceeds "
            f"{float(episode['start_joint_tolerance_rad']):.6f} rad"
        )
    if not grasped:
        blockers.append("Hand does not report a held bolt at PREINSERT")
    if width > float(gripper_config["maximum_accepted_held_width_m"]):
        blockers.append("live held width is inconsistent with the calibrated bolt grasp")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    all_q = np.concatenate([plans[name].q_targets for name in STAGES])
    recorded_xyz = np.concatenate([plans[name].xyz_targets for name in STAGES])
    predicted_xyz = np.asarray([
        (
            np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
            @ joint7_to_tcp
        )[:3, 3]
        for q in all_q
    ])
    fk_error = float(np.max(np.linalg.norm(predicted_xyz - recorded_xyz, axis=1)))
    if fk_error > float(episode["maximum_fk_path_error_m"]):
        blockers.append("replay FK discrepancy exceeds configured maximum")
    margin = float(episode["joint_margin_rad"])
    if np.any(all_q < JOINT_MIN + margin) or np.any(all_q > JOINT_MAX - margin):
        blockers.append("replay violates guarded joint range")
    minimum = np.asarray(episode["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(episode["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(predicted_xyz < minimum) or np.any(predicted_xyz > maximum):
        blockers.append("predicted replay leaves configured workspace")

    print(f"\nlive start error:         {start_error:.6f} rad")
    print(f"live Hand width/grasped:  {width:.6f} m / {grasped}")
    print(f"maximum replay FK error: {fk_error:.6f} m")
    print(f"replay xyz min/max:       {predicted_xyz.min(0)} / {predicted_xyz.max(0)}")
    print(f"scaled source peak |dq|: {max(p.max_source_abs_dq_rad_s for p in plans.values()) / scale:.6f} rad/s")
    if blockers:
        print("Execution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if blockers:
        raise SystemExit("[SAFETY] live drop/regrasp execution refused")

    visual = input(
        "Visually confirm the bolt is securely held, aligned, and clear of the guide. "
        "Press Enter to run; type anything to abort: "
    ).strip()
    if visual:
        raise SystemExit("[operator] live drop/regrasp aborted")

    robot.relative_dynamics_factor = float(episode["stage_robot_global_dynamics_factor"])
    stop_requested = threading.Event()
    may_be_holding = True

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
        monitor_stop = threading.Event()
        lost = threading.Event()

        def monitor() -> None:
            while not monitor_stop.wait(1.0 / float(gripper_config["monitor_hz"])):
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
            raise RuntimeError(f"{name} stopped or lost the bolt grasp")
        error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_stop_q)))
        print(f"[{name}] terminal error={error:.6f} rad")
        if error > float(episode["terminal_joint_tolerance_rad"]):
            raise RuntimeError(f"{name} terminal tolerance failed")

    try:
        run_stage("lower_held", monitor_grasp=True)
        confirm_support = input(
            "Visually confirm the guide safely supports the upright bolt. "
            "Press Enter to release; type anything to abort with the Hand CLOSED: "
        ).strip()
        if confirm_support:
            raise RuntimeError("operator rejected bolt support check")
        print("[bolt] opening Hand while arm is stationary")
        gripper.open(float(gripper_config["open_speed_m_s"]))
        may_be_holding = False
        open_width = float(gripper.state.width)
        print(f"[bolt release] Hand width={open_width:.6f} m")
        if open_width < float(gripper_config["minimum_open_width_m"]):
            raise RuntimeError("Hand failed to reach the required release width")
        released = input(
            "Visually confirm the bolt remains upright and self-supported. "
            "Press Enter to reposition; type anything to abort: "
        ).strip()
        if released:
            raise RuntimeError("operator rejected upright-bolt check after release")
        run_stage("reposition_regrasp_open", monitor_grasp=False)
        print("[bolt] regrasping head while arm is stationary")
        accepted = gripper.grasp(
            float(gripper_config["grasp_target_width_m"]),
            float(gripper_config["grasp_speed_m_s"]),
            float(gripper_config["grasp_force_n"]),
            float(gripper_config["epsilon_inner_m"]),
            float(gripper_config["epsilon_outer_m"]),
        )
        may_be_holding = True
        held_width = float(gripper.state.width)
        grasped = bool(gripper.is_grasped)
        print(f"[head regrasp] accepted={accepted} is_grasped={grasped} width={held_width:.6f} m")
        if not accepted or not grasped:
            raise RuntimeError("Hand did not verify the bolt-head regrasp")
        if held_width > float(gripper_config["maximum_accepted_held_width_m"]):
            raise RuntimeError("regrasp width is inconsistent with the calibrated bolt grasp")
        final_visual = input(
            "Visually confirm the bolt is upright and securely held by its head. "
            "Press Enter to accept THREAD_READY; type anything to reject while keeping it held: "
        ).strip()
        if final_visual:
            raise RuntimeError("operator rejected THREAD_READY visual check")
    except BaseException:
        stop_all()
        if may_be_holding:
            print("[SAFETY] The Hand may hold the bolt; it was NOT automatically opened.")
        raise
    print("[SUCCESS] THREAD_READY: bolt upright and regrasped by the head")
    print("No rotation, force-controlled contact, threading, release, or camera command was issued.")


if __name__ == "__main__":
    main()
