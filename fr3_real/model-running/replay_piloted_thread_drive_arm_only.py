#!/usr/bin/env python3
"""Preview or execute an automatic reset plus empty-Hand Pilot thread-drive replay."""

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

from expert_demo.piloted_thread_replay import build_piloted_thread_drive  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_piloted_thread_replay_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.expanduser().resolve().read_text())
    source = project_path(config["source_episode"])
    ratchet_limits = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
    q7_cable_min = float(ratchet_limits["summary"]["q7_prewind_rad"])
    q7_cable_max = float(ratchet_limits["summary"]["q7_forward_rad"])
    plan = build_piloted_thread_drive(
        source,
        cycle=int(config["source_cycle"]),
        command_rate_hz=float(config["command_rate_hz"]),
        maximum_q7_rad=float(config["maximum_replay_q7_rad"]),
    )
    scale = float(config["playback_time_scale"])
    scaled_dq = plan.dq_targets / scale
    expected_duration = plan.source_duration_s * scale
    static_blockers: list[str] = []
    if config.get("hand_connection_enabled") is not False:
        static_blockers.append("Hand connection must remain disabled for arm-only commissioning")
    if config.get("hand_commands_enabled") is not False:
        static_blockers.append("Hand commands must remain disabled for arm-only commissioning")
    if config.get("contact_threading_enabled") is not False:
        static_blockers.append("contact/threading must remain disabled for arm-only commissioning")
    if float(np.max(np.abs(scaled_dq))) > float(config["maximum_scaled_target_velocity_rad_s"]):
        static_blockers.append("scaled recorded velocity exceeds configured maximum")
    if plan.maximum_target_step_rad > float(config["maximum_target_step_rad"]):
        static_blockers.append("resampled joint target step exceeds configured maximum")
    margin = float(config["joint_margin_rad"])
    if np.any(plan.q_targets[:, :6] < (JOINT_MIN + margin)[:6]) or np.any(
        plan.q_targets[:, :6] > (JOINT_MAX - margin)[:6]
    ):
        static_blockers.append("recorded drive leaves the guarded q1-q6 range")
    if np.any(plan.q_targets[:, 6] < q7_cable_min) or np.any(
        plan.q_targets[:, 6] > q7_cable_max
    ):
        static_blockers.append("recorded drive leaves the operator-calibrated q7 cable range")
    minimum = np.asarray(config["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(config["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(plan.xyz_targets < minimum) or np.any(plan.xyz_targets > maximum):
        static_blockers.append("recorded drive leaves the configured local workspace")

    print("=== PILOTED THREAD DRIVE 1 ARM-ONLY REPLAY ===")
    print(
        "mode:                     "
        + ("DRY_RUN" if args.dry_run else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE")
    )
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"source episode:           {source}")
    print(f"source cycle:             {plan.cycle}")
    print(f"targets:                  {len(plan.q_targets)} x 7 at {plan.command_rate_hz:.1f} Hz")
    print(f"source duration:          {plan.source_duration_s:.3f} s")
    print(f"playback:                 {scale:.1f}x slower")
    print(f"expected replay duration: {expected_duration:.3f} s")
    print(f"q7 replay interval:       {plan.q_targets[0,6]:+.6f} .. {plan.q_targets[-1,6]:+.6f} rad")
    print(f"q7 cable-safe interval:   {q7_cable_min:+.6f} .. {q7_cable_max:+.6f} rad")
    print(f"scaled peak |dq|:         {float(np.max(np.abs(scaled_dq))):.6f} rad/s")
    print(f"required physical setup:  {config['required_setup']}")
    print("automatic reset:          INCLUDED — saved 7-joint drive-start target")
    print("Hand connection/commands: DISABLED")
    print("contact/threading/camera: DISABLED")
    print(f"execution enabled:        {config['execution_enabled']}")
    if static_blockers:
        print("\nStatic blockers:")
        for blocker in static_blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if args.execute and config.get("execution_enabled") is not True:
        raise SystemExit("[SAFETY] arm-only execution remains locked pending read-only preview")
    if static_blockers:
        raise SystemExit("[SAFETY] arm-only replay configuration is invalid")

    input("Confirm there is NO BOLT and press Enter to connect read-only (Ctrl+C cancels): ")
    from franky import Duration, Frame, JointMotion, JointState, JointWaypoint, JointWaypointMotion, Robot

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)

    def fk_xyz(q_values: np.ndarray) -> np.ndarray:
        return np.asarray([
            (
                np.asarray(
                    robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix,
                    dtype=np.float64,
                )
                @ joint7_to_tcp
            )[:3, 3]
            for q in q_values
        ])

    predicted_xyz = fk_xyz(plan.q_targets)
    fk_error = float(np.max(np.linalg.norm(predicted_xyz - plan.xyz_targets, axis=1)))
    runtime_blockers = list(static_blockers)
    if fk_error > float(config["maximum_fk_path_error_m"]):
        runtime_blockers.append("replay FK discrepancy exceeds configured maximum")
    if np.any(predicted_xyz < minimum) or np.any(predicted_xyz > maximum):
        runtime_blockers.append("predicted replay path leaves the local workspace")

    reset = config["automatic_reset_to_start"]
    reset_q = np.linspace(current_q, plan.expected_start_q, num=201)
    reset_xyz = fk_xyz(reset_q)
    reset_error = float(np.linalg.norm(reset_xyz[-1] - plan.xyz_targets[0]))
    reset_delta = float(np.max(np.abs(plan.expected_start_q - current_q)))
    reset_blockers: list[str] = []
    if reset.get("enabled") is not True:
        reset_blockers.append("automatic reset is disabled")
    if reset_delta > float(reset["maximum_joint_delta_rad"]):
        reset_blockers.append(
            f"reset joint delta {reset_delta:.6f} rad exceeds configured maximum"
        )
    reset_margin = float(reset["joint_margin_rad"])
    hard_margin = float(reset["minimum_mechanical_margin_for_inward_only_start_rad"])
    hard_lower = JOINT_MIN[:6] + hard_margin
    hard_upper = JOINT_MAX[:6] - hard_margin
    if np.any(reset_q[:, :6] < hard_lower) or np.any(reset_q[:, :6] > hard_upper):
        reset_blockers.append("automatic reset violates the hard q1-q6 mechanical margin")
    conservative_lower = JOINT_MIN[:6] + reset_margin
    conservative_upper = JOINT_MAX[:6] - reset_margin
    target_inside_conservative = np.all(
        (plan.expected_start_q[:6] >= conservative_lower)
        & (plan.expected_start_q[:6] <= conservative_upper)
    )
    if not target_inside_conservative:
        reset_blockers.append("automatic reset target violates conservative q1-q6 ranges")
    for joint in range(6):
        start = current_q[joint]
        target = plan.expected_start_q[joint]
        if start < conservative_lower[joint] and target <= start:
            reset_blockers.append(
                f"q{joint + 1} starts below its conservative range and does not move inward"
            )
        if start > conservative_upper[joint] and target >= start:
            reset_blockers.append(
                f"q{joint + 1} starts above its conservative range and does not move inward"
            )
    if np.any(reset_q[:, 6] < q7_cable_min) or np.any(reset_q[:, 6] > q7_cable_max):
        reset_blockers.append("automatic reset leaves the q7 cable-safe interval")
    reset_min = np.asarray(reset["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    reset_max = np.asarray(reset["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(reset_xyz < reset_min) or np.any(reset_xyz > reset_max):
        reset_blockers.append("predicted automatic-reset path leaves its workspace")
    if reset_error > float(reset["maximum_fk_target_error_m"]):
        reset_blockers.append("automatic reset target FK differs from recorded drive start")
    runtime_blockers.extend(reset_blockers)

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"saved drive-start q:      {np.array2string(plan.expected_start_q, precision=5)}")
    print(f"automatic reset max dq:  {reset_delta:.6f} rad")
    print(f"reset path xyz min:       {np.array2string(reset_xyz.min(0), precision=5)}")
    print(f"reset path xyz max:       {np.array2string(reset_xyz.max(0), precision=5)}")
    print(f"reset target FK error:    {reset_error:.6f} m")
    print(f"replay path xyz min:      {np.array2string(predicted_xyz.min(0), precision=5)}")
    print(f"replay path xyz max:      {np.array2string(predicted_xyz.max(0), precision=5)}")
    print(f"maximum replay FK error: {fk_error:.6f} m")
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] automatic reset/replay refused")

    stop_requested = threading.Event()

    def stop_robot() -> None:
        stop_requested.set()
        robot.stop()

    def interrupt(signum, frame) -> None:
        del signum, frame
        print("\n[KILLSWITCH] stopping arm")
        stop_robot()

    signal.signal(signal.SIGINT, interrupt)
    input("Keep the local path clear. Press Enter for the slow automatic reset to DRIVE 1 start: ")
    robot.move(
        JointMotion(
            JointState(plan.expected_start_q),
            relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
        )
    )
    reset_terminal_error = float(
        np.max(np.abs(np.asarray(robot.state.q, dtype=np.float64) - plan.expected_start_q))
    )
    print(f"[auto-reset] terminal error={reset_terminal_error:.6f} rad")
    if reset_terminal_error > float(reset["terminal_joint_tolerance_rad"]):
        raise SystemExit("[SAFETY] automatic reset terminal tolerance failed")

    input("Confirm again: NO BOLT, empty Hand, clear local path. Press Enter to replay DRIVE 1: ")
    waypoint_ms = int(math.ceil(1000.0 * scale / plan.command_rate_hz))
    waypoints = [
        JointWaypoint(
            JointState(q, dq),
            minimum_time=Duration(waypoint_ms),
            max_total_duration=Duration(3000),
        )
        for q, dq in zip(plan.q_targets[1:], scaled_dq[1:], strict=True)
    ]
    timer = threading.Timer(float(config["state_timeout_s"]), stop_robot)
    timer.daemon = True
    timer.start()
    try:
        robot.move(
            JointWaypointMotion(
                waypoints,
                relative_dynamics_factor=float(config["relative_dynamics_factor"]),
            )
        )
    finally:
        timer.cancel()
    if stop_requested.is_set():
        raise SystemExit("[SAFETY] replay timed out or was stopped")
    final_q = np.asarray(robot.state.q, dtype=np.float64)
    final_error = float(np.max(np.abs(final_q - plan.expected_stop_q)))
    print(f"[DONE] final q:           {np.array2string(final_q, precision=5)}")
    print(f"[DONE] terminal error:    {final_error:.6f} rad")
    if final_error > float(config["terminal_joint_tolerance_rad"]):
        raise SystemExit("[SAFETY] replay ended outside terminal joint tolerance")
    print("[SUCCESS] DRIVE 1 arm-only replay passed; no Hand/contact/camera command was issued")


if __name__ == "__main__":
    main()
