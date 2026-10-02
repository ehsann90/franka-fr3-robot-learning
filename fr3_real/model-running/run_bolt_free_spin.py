#!/usr/bin/env python3
"""Guarded first +15-degree bolt-held rotation with no engagement or axial preload."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_free_spin import BoltFreeSpinGuard  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_free_spin_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        model = json.loads(project_path(config["dynamic_model_calibration"]).read_text())
        limits = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
        head_width = json.loads(
            project_path(config["thread_ready_head_width_calibration"]).read_text()
        )
        reference = np.load(project_path(config["thread_ready_reference_capture"]) / "trajectory_arrays.npz")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load bolt free-spin configuration: {error}") from error
    if model.get("thresholds_derived") is not False:
        raise SystemExit("dynamic model calibration has unexpected threshold state")
    if (
        head_width.get("calibration_type") != "bolt_held_width"
        or head_width.get("screw_grasp_style") != "THREAD_READY_HEAD"
        or head_width.get("all_samples_is_grasped") is not True
        or head_width.get("robot_ip") != config["robot_ip"]
    ):
        raise SystemExit("THREAD_READY_HEAD width calibration is invalid or incompatible")
    if config.get("axial_preload_enabled") is not False or config.get("threading_enabled") is not False:
        raise SystemExit("axial preload and threading must remain disabled")

    print("=== GUARDED BOLT-HELD FREE-ROTATION TEST ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"required setup:           {config['required_setup']}")
    print(f"rotation:                 +{config['rotation_angle_deg']:.1f} deg q7")
    print(f"relative dynamics factor: {config['relative_dynamics_factor']}")
    print(f"hard residual K_Tz stop:  {config['approved_hard_negative_residual_Tz_Nm']:+.6f} Nm")
    print(f"hard axial delta stop:    +{config['approved_hard_positive_axial_delta_N']:.3f} N")
    print("axial preload/contact:    DISABLED")
    print("threading:                DISABLED; BOLT MUST NOT BE ENGAGED")
    print(f"execution enabled:        {config['execution_enabled']}")
    if args.execute and config.get("execution_enabled") is not True:
        raise SystemExit("[SAFETY] bolt free-spin execution is not commissioned; no connection attempted")

    input("Press Enter to connect and validate the live arm and Hand (Ctrl+C cancels): ")
    from franky import Frame, Gripper, JointMotion, JointState, Robot

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    q = np.asarray(state.q, dtype=np.float64)
    T = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    ee_t_k = np.asarray(state.EE_T_K.matrix, dtype=np.float64)
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    blockers: list[str] = []
    width_margin = float(config["gripper_width_comparison_margin_m"])
    accepted_width_min = float(head_width["minimum_width_m"]) - width_margin
    accepted_width_max = float(head_width["maximum_width_m"]) + width_margin
    if not grasped or not accepted_width_min <= width <= accepted_width_max:
        blockers.append("Hand state is inconsistent with a securely held bolt")
    if not np.allclose(ee_t_k, np.eye(4), atol=1e-8):
        blockers.append("EE_T_K is not identity")
    margin = float(config["joint_mechanical_margin_rad"])
    if np.any(q < JOINT_MIN + margin) or np.any(q > JOINT_MAX - margin):
        blockers.append("live joints violate guarded mechanical ranges")
    target_q = q.copy()
    target_q[6] += float(config["rotation_angle_rad"])
    cable_min = float(limits["summary"]["q7_prewind_rad"])
    cable_max = float(limits["summary"]["q7_forward_rad"])
    if q[6] < cable_min or target_q[6] > cable_max:
        blockers.append("live/start target q7 leaves the calibrated cable-safe interval")
    knots = np.asarray(model["knots"], dtype=np.float64)
    if q[6] < knots[0, 0] - 0.02 or target_q[6] > knots[-1, 0]:
        blockers.append("q7 test interval is outside the calibrated dynamic torque model")

    reference_T = np.asarray(reference["T_base_ee"], dtype=np.float64).mean(axis=0)
    translation_error = float(np.linalg.norm(T[:3, 3] - reference_T[:3, 3]))
    axis_dot = float(np.clip(np.dot(T[:3, 2], reference_T[:3, 2]), -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(axis_dot)))
    if translation_error > float(config["maximum_start_translation_error_m"]):
        blockers.append("live position is too far from THREAD_READY")
    if tilt_deg > float(config["maximum_start_tool_axis_tilt_deg"]):
        blockers.append("live tool axis is too tilted from the calibrated screw axis")
    minimum = np.asarray(config["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(config["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(T[:3, 3] < minimum) or np.any(T[:3, 3] > maximum):
        blockers.append("live TCP position is outside the local workspace")

    baseline_rows = []
    deadline = time.perf_counter() + float(config["baseline_duration_s"])
    while time.perf_counter() < deadline:
        sample = robot.state
        baseline_rows.append([
            float(sample.K_F_ext_hat_K[2]),
            float(sample.K_F_ext_hat_K[5]),
            float(np.max(np.abs(np.asarray(sample.dq, dtype=np.float64)))),
        ])
        time.sleep(1.0 / float(config["baseline_sample_hz"]))
    baseline_rows = np.asarray(baseline_rows)
    baseline_Fz = float(np.mean(baseline_rows[:, 0]))
    baseline_Fz_std = float(np.std(baseline_rows[:, 0]))
    maximum_dq = float(np.max(baseline_rows[:, 2]))
    if baseline_Fz_std > float(config["maximum_baseline_Fz_std_N"]):
        blockers.append("live baseline Fz is not stationary enough")
    if maximum_dq > float(config["maximum_baseline_abs_dq_rad_s"]):
        blockers.append("arm moved during live baseline")

    measured_tcp = T
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(q, target_q, num=101)
    path_xyz = np.asarray([
        (np.asarray(robot.model.pose(Frame.Joint7, waypoint, state.F_T_EE, state.EE_T_K).matrix, dtype=np.float64) @ joint7_to_tcp)[:3, 3]
        for waypoint in q_path
    ])
    if np.any(path_xyz < minimum) or np.any(path_xyz > maximum):
        blockers.append("predicted q7-only path leaves local workspace")

    print(f"\nlive/target q7:           {q[6]:+.6f} / {target_q[6]:+.6f} rad")
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    print(
        f"accepted head width:      {accepted_width_min:.6f} .. "
        f"{accepted_width_max:.6f} m"
    )
    print(f"THREAD_READY error:       {translation_error * 1000:.3f} mm / tool-axis tilt {tilt_deg:.3f} deg")
    print(f"live baseline F_Kz:       {baseline_Fz:+.6f} N (std={baseline_Fz_std:.6f} N)")
    print(f"maximum baseline |dq|:    {maximum_dq:.6f} rad/s")
    print(f"predicted path xyz min:   {np.array2string(path_xyz.min(axis=0), precision=5)}")
    print(f"predicted path xyz max:   {np.array2string(path_xyz.max(axis=0), precision=5)}")
    if blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if blockers:
        raise SystemExit("[SAFETY] bolt-held free-rotation execution refused")

    response = input(
        "Visually confirm the bolt is upright, head-held, freely supported, NOT engaged, and "
        "unloaded. Type RUN_BOLT_FREE_SPIN_15_DEG; anything else aborts: "
    )
    if response != "RUN_BOLT_FREE_SPIN_15_DEG":
        print("[free-spin] aborted without motion")
        return

    guard = BoltFreeSpinGuard(
        model_q7=knots[:, 0],
        model_K_Tz=knots[:, 1],
        baseline_Fz_N=baseline_Fz,
        hard_negative_residual_Tz_Nm=float(config["approved_hard_negative_residual_Tz_Nm"]),
        hard_positive_axial_delta_N=float(config["approved_hard_positive_axial_delta_N"]),
        q7_min_rad=cable_min,
        q7_max_rad=cable_max,
        start_xyz_m=T[:3, 3].copy(),
        maximum_tcp_translation_m=float(config["maximum_tcp_translation_m"]),
    )
    capacity = int(config["callback_capacity"])
    trace = np.full((capacity, 23), np.nan, dtype=np.float64)
    trace_count = 0

    def callback(callback_state, *unused) -> None:
        nonlocal trace_count
        del unused
        wrench = np.asarray(callback_state.K_F_ext_hat_K, dtype=np.float64)
        pose = np.asarray(callback_state.O_T_EE.matrix, dtype=np.float64)
        q_live = np.asarray(callback_state.q, dtype=np.float64)
        residual, axial = guard.update(q_live[6], wrench, pose[:3, 3])
        if trace_count < capacity:
            trace[trace_count] = [
                float(callback_state.time.to_sec()), *q_live,
                *np.asarray(callback_state.dq, dtype=np.float64),
                *wrench, residual, axial,
            ]
            trace_count += 1

    monitor_done = threading.Event()

    def monitor_stop() -> None:
        while not monitor_done.wait(0.002):
            if guard.stop_reason is not None:
                robot.stop()
                return

    def interrupt(signum, frame) -> None:
        del signum, frame
        guard.stop_reason = "operator_abort"
        robot.stop()

    signal.signal(signal.SIGINT, interrupt)
    robot.relative_dynamics_factor = float(config["relative_dynamics_factor"])
    motion = JointMotion(JointState(target_q))
    motion.register_callback(callback)
    monitor = threading.Thread(target=monitor_stop, daemon=True)
    monitor.start()
    motion_error = None
    try:
        robot.move(motion)
    except BaseException as error:
        motion_error = repr(error)
        if guard.stop_reason is None:
            raise
    finally:
        monitor_done.set()
        monitor.join(timeout=2.0)

    final = robot.state
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": guard.stop_reason or "motion_completed",
        "motion_error": motion_error,
        "minimum_residual_K_Tz_Nm": guard.minimum_residual_Tz_Nm,
        "maximum_axial_delta_N": guard.maximum_axial_delta_N,
        "maximum_tcp_translation_m": guard.maximum_tcp_translation_observed_m,
        "final_q7_rad": float(final.q[6]),
        "target_q7_rad": float(target_q[6]),
        "axial_preload_used": False,
        "threading_enabled": False,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = project_path(config["output_dir"]) / f"{stamp}_bolt_free_spin"
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "control_trace.npz", trace=trace[:trace_count])
    print(f"[result] stop reason:     {result['stop_reason']}")
    print(f"[result] minimum residual:{result['minimum_residual_K_Tz_Nm']:+.6f} Nm")
    print(f"[result] maximum axial:   {result['maximum_axial_delta_N']:+.6f} N")
    print(f"[result] TCP translation: {result['maximum_tcp_translation_m'] * 1000:.3f} mm")
    print(f"[saved] {output}")
    print("No axial preload, contact search, release, ratchet, or threading command was issued.")


if __name__ == "__main__":
    main()
