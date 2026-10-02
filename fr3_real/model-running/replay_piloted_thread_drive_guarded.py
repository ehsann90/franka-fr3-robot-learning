#!/usr/bin/env python3
"""Preview or run guarded bolt-held replay of successful Pilot thread Drive 1."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.piloted_thread_guard import PilotedThreadReplayGuard  # noqa: E402
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
    parser.add_argument(
        "--resume-held",
        action="store_true",
        help="resume at Drive 1 start with the upright bolt already securely head-held",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.resume_held and not args.execute:
        raise SystemExit("--resume-held is valid only with --execute")

    config = json.loads(args.config.expanduser().resolve().read_text())
    guarded = config["guarded_bolt_held_replay"]
    limits = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
    dynamic = json.loads(project_path("calibration/bolt_wrist_dynamic_model_v1.json").read_text())
    head = json.loads(project_path(guarded["thread_ready_head_width_calibration"]).read_text())
    source = project_path(config["source_episode"])
    output_root = project_path(guarded["output_dir"])
    plan = build_piloted_thread_drive(
        source,
        cycle=int(config["source_cycle"]),
        command_rate_hz=float(config["command_rate_hz"]),
        maximum_q7_rad=float(config["maximum_replay_q7_rad"]),
    )
    scale = float(config["playback_time_scale"])
    scaled_dq = plan.dq_targets / scale
    q7_min = float(limits["summary"]["q7_prewind_rad"])
    q7_max = float(limits["summary"]["q7_forward_rad"])
    static_blockers: list[str] = []
    if np.any(plan.q_targets[:, 6] < q7_min) or np.any(plan.q_targets[:, 6] > q7_max):
        static_blockers.append("replay leaves the operator-calibrated q7 cable range")
    if float(np.max(np.abs(scaled_dq))) > float(config["maximum_scaled_target_velocity_rad_s"]):
        static_blockers.append("scaled replay velocity exceeds configured maximum")
    if plan.maximum_target_step_rad > float(config["maximum_target_step_rad"]):
        static_blockers.append("resampled target step exceeds configured maximum")
    if not np.allclose(plan.dq_targets[[0, -1]], 0.0):
        static_blockers.append("replay boundary velocities are not explicitly stationary")

    print("=== GUARDED PILOTED BOLT THREAD DRIVE 1 ===")
    print(
        "mode:                     "
        + ("DRY_RUN" if args.dry_run else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE")
    )
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"source:                   {source}")
    print(f"trajectory:               full Drive 1, q7 uniformly scaled inside cable limit")
    print(f"targets/source duration:  {len(plan.q_targets)} / {plan.source_duration_s:.3f} s")
    print(f"playback/duration:        {scale:.1f}x slower / {plan.source_duration_s*scale:.3f} s")
    print(f"q7 replay interval:       {plan.q_targets[0,6]:+.6f} .. {plan.q_targets[:,6].max():+.6f} rad")
    print(f"guarded q7 stop:          {float(np.asarray(dynamic['knots'])[-1,0]):+.6f} rad (torque-model boundary)")
    print(f"automatic reset:          INCLUDED — exact saved seven-joint start")
    print(f"hard axial cutoff:        +/-{float(guarded['approved_hard_absolute_axial_delta_N']):.3f} N")
    print(f"hard residual cutoff:     {float(guarded['approved_hard_negative_residual_Tz_Nm']):+.6f} Nm")
    print(f"q1-q6 tracking cutoff:    {float(guarded['maximum_q1_q6_path_tracking_error_rad']):.3f} rad")
    print("Hand sequence:             reset empty/open -> operator places upright bolt -> automatic head grasp")
    print("ratchet/camera/continuation: DISABLED")
    print(f"guarded replay status:    {guarded['status']}")
    print(f"execution enabled:        {guarded['execution_enabled']}")
    source_guard = guarded["source_guard_analysis"]
    print("\nIMPORTANT source-envelope comparison:")
    print(f"  successful source axial delta: {source_guard['source_axial_delta_range_N'][0]:+.3f} .. {source_guard['source_axial_delta_range_N'][1]:+.3f} N")
    print(f"  successful source minimum residual: {source_guard['source_minimum_residual_K_Tz_Nm']:+.3f} Nm")
    print("  The source exceeded both current guards; a live guarded replay may stop early.")
    if static_blockers:
        print("\nStatic blockers:")
        for blocker in static_blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if args.execute and guarded.get("execution_enabled") is not True:
        raise SystemExit("[SAFETY] guarded bolt-held execution remains locked pending preview")
    if args.execute:
        existing = list(output_root.glob("*_guarded_drive1")) if output_root.exists() else []
        if len(existing) >= int(guarded["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] guarded Drive 1 physical-trial allowance is exhausted")
    if static_blockers:
        raise SystemExit("[SAFETY] guarded replay configuration is invalid")

    input(
        (
            "Confirm the bolt is upright, securely head-held, and the arm has not moved; press Enter to connect (Ctrl+C cancels): "
            if args.resume_held
            else "Confirm there is NO BOLT and press Enter to connect read-only (Ctrl+C cancels): "
        )
    )
    from franky import Duration, Frame, Gripper, JointMotion, JointState, JointWaypoint, JointWaypointMotion, Robot

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
                ) @ joint7_to_tcp
            )[:3, 3]
            for q in q_values
        ])

    predicted_xyz = fk_xyz(plan.q_targets)
    fk_error = float(np.max(np.linalg.norm(predicted_xyz - plan.xyz_targets, axis=1)))
    runtime_blockers = list(static_blockers)
    if fk_error > float(config["maximum_fk_path_error_m"]):
        runtime_blockers.append("replay FK discrepancy exceeds configured maximum")
    path_min = np.asarray(config["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    path_max = np.asarray(config["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(predicted_xyz < path_min) or np.any(predicted_xyz > path_max):
        runtime_blockers.append("predicted replay leaves the local workspace")

    reset = config["automatic_reset_to_start"]
    reset_q = np.linspace(current_q, plan.expected_start_q, num=201)
    reset_xyz = fk_xyz(reset_q)
    reset_delta = float(np.max(np.abs(plan.expected_start_q - current_q)))
    reset_error = float(np.linalg.norm(reset_xyz[-1] - plan.xyz_targets[0]))
    maximum_reset_segment_delta = float(reset["maximum_joint_delta_rad"])
    reset_segment_count = max(1, int(math.ceil(reset_delta / maximum_reset_segment_delta)))
    reset_segment_targets = np.linspace(current_q, plan.expected_start_q, num=reset_segment_count + 1)[1:]
    reset_segment_deltas = np.max(
        np.abs(np.diff(np.vstack([current_q, reset_segment_targets]), axis=0)), axis=1
    )
    if float(np.max(reset_segment_deltas)) > maximum_reset_segment_delta + 1e-9:
        runtime_blockers.append("one segmented automatic-reset step exceeds its maximum joint delta")
    if reset_segment_count > 3:
        runtime_blockers.append("automatic reset would require more than three segments")
    hard_margin = float(reset["minimum_mechanical_margin_for_inward_only_start_rad"])
    if np.any(reset_q[:, :6] < (JOINT_MIN + hard_margin)[:6]) or np.any(
        reset_q[:, :6] > (JOINT_MAX - hard_margin)[:6]
    ):
        runtime_blockers.append("automatic reset violates hard q1-q6 margins")
    if np.any(reset_q[:, 6] < q7_min) or np.any(reset_q[:, 6] > q7_max):
        runtime_blockers.append("automatic reset leaves the q7 cable range")
    reset_min = np.asarray(reset["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    reset_max = np.asarray(reset["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(reset_xyz < reset_min) or np.any(reset_xyz > reset_max):
        runtime_blockers.append("automatic reset leaves its workspace")
    if reset_error > float(reset["maximum_fk_target_error_m"]):
        runtime_blockers.append("automatic reset target differs from recorded start")

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"saved drive-start q:      {np.array2string(plan.expected_start_q, precision=5)}")
    print(f"automatic reset max dq:  {reset_delta:.6f} rad")
    print(f"automatic reset segments:{reset_segment_count} x max {float(np.max(reset_segment_deltas)):.6f} rad")
    print(f"reset xyz min/max:        {np.array2string(reset_xyz.min(0), precision=5)} / {np.array2string(reset_xyz.max(0), precision=5)}")
    print(f"reset target FK error:    {reset_error:.6f} m")
    print(f"replay xyz min/max:       {np.array2string(predicted_xyz.min(0), precision=5)} / {np.array2string(predicted_xyz.max(0), precision=5)}")
    print(f"maximum replay FK error: {fk_error:.6f} m")
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] guarded reset/replay refused")

    gripper = Gripper(config["robot_ip"])
    accepted = True
    if args.resume_held:
        live_start_error = float(
            np.max(np.abs(np.asarray(robot.state.q, dtype=np.float64) - plan.expected_start_q))
        )
        print(f"[resume] live Drive 1 start error={live_start_error:.6f} rad")
        if live_start_error > float(config["start_joint_tolerance_rad"]):
            raise SystemExit("[SAFETY] held-resume arm pose no longer matches Drive 1 start")
    else:
        input("Press Enter for the slow empty-Hand automatic reset to Drive 1 start: ")
        for index, target in enumerate(reset_segment_targets, start=1):
            robot.move(JointMotion(JointState(target), relative_dynamics_factor=float(reset["relative_dynamics_factor"])))
            segment_error = float(np.max(np.abs(np.asarray(robot.state.q) - target)))
            print(f"[auto-reset {index}/{reset_segment_count}] terminal error={segment_error:.6f} rad")
            if segment_error > float(reset["terminal_joint_tolerance_rad"]):
                raise SystemExit("[SAFETY] automatic reset segment terminal tolerance failed")
        reset_terminal_error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_start_q)))
        if reset_terminal_error > float(reset["terminal_joint_tolerance_rad"]):
            raise SystemExit("[SAFETY] automatic reset terminal tolerance failed")
        gripper.open(float(guarded["grasp_speed_m_s"]))
        input("Place the bolt upright in the guide between the open fingers. Press Enter to grasp its head: ")
        accepted = bool(gripper.grasp(float(guarded["grasp_width_m"]), float(guarded["grasp_speed_m_s"]), float(guarded["grasp_force_N"])))
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    margin = float(guarded["gripper_width_comparison_margin_m"])
    accepted_min = float(head["minimum_width_m"]) - margin
    accepted_max = float(head["maximum_width_m"]) + margin
    alternate_min, alternate_max = map(float, guarded["accepted_alternate_head_width_m"])
    width_valid = accepted_min <= width <= accepted_max or alternate_min <= width <= alternate_max
    print(f"[head grasp] accepted/width/grasped: {accepted} / {width:.6f} m / {grasped}")
    print(f"[head grasp] valid width bands: {accepted_min:.6f}..{accepted_max:.6f} or {alternate_min:.6f}..{alternate_max:.6f} m")
    if not accepted or not grasped or not width_valid:
        raise SystemExit("[SAFETY] automatic head grasp validation failed; Hand remains closed")
    visual = input(
        "Visually confirm the bolt is upright and securely held by its head. Press Enter to continue; type anything to abort: "
    )
    if visual:
        print("[abort] Hand remains closed; no replay command was issued")
        return
    del gripper

    baseline_rows = []
    deadline = time.perf_counter() + float(guarded["baseline_duration_s"])
    while time.perf_counter() < deadline:
        sample = robot.state
        baseline_rows.append([sample.K_F_ext_hat_K[2], np.max(np.abs(sample.dq))])
        time.sleep(1.0 / float(guarded["baseline_sample_hz"]))
    baseline_rows = np.asarray(baseline_rows, dtype=np.float64)
    baseline_fz = float(np.mean(baseline_rows[:, 0]))
    if float(np.std(baseline_rows[:, 0])) > float(guarded["maximum_baseline_Fz_std_N"]):
        raise SystemExit("[SAFETY] axial baseline is not stationary; Hand remains closed")
    if float(np.max(baseline_rows[:, 1])) > float(guarded["maximum_baseline_abs_dq_rad_s"]):
        raise SystemExit("[SAFETY] arm moved during baseline; Hand remains closed")

    knots = np.asarray(dynamic["knots"], dtype=np.float64)
    dynamic_q7_max = float(knots[-1, 0])
    guard = PilotedThreadReplayGuard(
        path_q=plan.q_targets,
        model_q7=knots[:, 0],
        model_K_Tz=knots[:, 1],
        baseline_Fz_N=baseline_fz,
        hard_absolute_axial_delta_N=float(guarded["approved_hard_absolute_axial_delta_N"]),
        hard_negative_residual_Tz_Nm=float(guarded["approved_hard_negative_residual_Tz_Nm"]),
        q7_min_rad=q7_min,
        q7_max_rad=dynamic_q7_max,
        maximum_q1_q6_path_error_rad=float(guarded["maximum_q1_q6_path_tracking_error_rad"]),
    )
    capacity = 100000
    trace = np.full((capacity, 24), np.nan, dtype=np.float64)
    trace_count = 0

    def callback(callback_state, *unused) -> None:
        nonlocal trace_count
        del unused
        q_live = np.asarray(callback_state.q, dtype=np.float64)
        wrench = np.asarray(callback_state.K_F_ext_hat_K, dtype=np.float64)
        residual, axial, path_error = guard.update(q_live, wrench)
        if trace_count < capacity:
            trace[trace_count] = [float(callback_state.time.to_sec()), *q_live, *np.asarray(callback_state.dq), *wrench, residual, axial, path_error]
            trace_count += 1
        else:
            guard.stop_reason = guard.stop_reason or "telemetry_capacity"

    monitor_done = threading.Event()

    def monitor() -> None:
        while not monitor_done.wait(0.002):
            if guard.stop_reason is not None:
                robot.stop(); return

    def interrupt(signum, frame) -> None:
        del signum, frame
        guard.stop_reason = "operator_abort"
        robot.stop()

    signal.signal(signal.SIGINT, interrupt)
    input("Keep one hand near emergency stop. Press Enter for guarded bolt-held Drive 1: ")
    waypoint_ms = int(math.ceil(1000.0 * scale / plan.command_rate_hz))
    waypoints = [JointWaypoint(JointState(q, dq), minimum_time=Duration(waypoint_ms), max_total_duration=Duration(3000)) for q, dq in zip(plan.q_targets[1:], scaled_dq[1:], strict=True)]
    motion = JointWaypointMotion(waypoints, relative_dynamics_factor=float(config["relative_dynamics_factor"]))
    motion.register_callback(callback)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    motion_error = None
    try:
        robot.move(motion)
    except BaseException as error:
        motion_error = repr(error)
        if guard.stop_reason is None:
            raise
    finally:
        monitor_done.set(); thread.join(timeout=2.0)
    output = output_root / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_guarded_drive1")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "control_trace.npz", trace=trace[:trace_count])
    result = {"stop_reason": guard.stop_reason or "motion_completed", "motion_error": motion_error, "minimum_residual_Tz_Nm": guard.minimum_residual_Tz_Nm, "axial_delta_range_N": [guard.minimum_axial_delta_N, guard.maximum_axial_delta_N], "maximum_q1_q6_path_error_rad": guard.maximum_q1_q6_path_error_observed_rad, "hand_remains_closed": True}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); print(f"[saved] {output}")
    print("The Hand remains closed. No ratchet, release, continuation, or camera command was issued.")


if __name__ == "__main__":
    main()
