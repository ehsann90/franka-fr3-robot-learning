#!/usr/bin/env python3
"""Preview or run the first stationary, no-rotation bolt axial-preload test."""

from __future__ import annotations

import argparse
import importlib.metadata
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

from expert_demo.bolt_axial_preload import (  # noqa: E402
    AxialPreloadGuard,
    decompose_translation_error,
    rotation_angle_rad,
    tool_axis_tilt_rad,
)


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_axial_preload_v1.json"
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

    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        head_width = json.loads(
            project_path(config["thread_ready_head_width_calibration"]).read_text()
        )
        reference = np.load(
            project_path(config["thread_ready_reference_capture"]) / "trajectory_arrays.npz"
        )
        manual_calibration = json.loads(
            (
                project_path(config["axial_force_calibration_capture"])
                / "axial_load_summary.json"
            ).read_text()
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load axial-preload configuration: {error}") from error

    target_force = float(config["approved_axial_force_target_Kz_N"])
    calibrated_force = float(manual_calibration["axial_force_loaded_minus_unloaded_N"])
    if not np.isclose(target_force, calibrated_force, atol=1e-6):
        raise SystemExit("approved target does not match the accepted manual calibration")
    if (
        head_width.get("calibration_type") != "bolt_held_width"
        or head_width.get("screw_grasp_style") != "THREAD_READY_HEAD"
        or head_width.get("all_samples_is_grasped") is not True
    ):
        raise SystemExit("THREAD_READY_HEAD width calibration is invalid")
    if any(config.get(key) is not False for key in ("rotation_enabled", "threading_enabled", "ratchet_enabled")):
        raise SystemExit("rotation, threading, and ratchet must remain disabled")

    controller = config["controller"]
    if controller.get("status") == "RETIRED_ZERO_JACOBIAN_DO_NOT_EXECUTE":
        print("=== RETIRED FRANKY AXIAL-PRELOAD PATH ===")
        print("status:                  DISABLED — DO NOT EXECUTE")
        print("reason:                  Franky 1.1.4 loads private libfranka 0.18.0 and returned an all-zero Jacobian")
        print("verified replacement:    direct repository libfranka 0.21.2 model/Jacobian path")
        print("next controller:         direct low-level libfranka; gains remain NOT_CALIBRATED")
        print("No robot or Hand connection was attempted.")
        return
    hard_force = float(config["approved_hard_absolute_axial_delta_N"])
    wrong_direction = config["wrong_direction_detector"]
    duration_s = float(controller["motion_duration_s"])
    timeout_s = float(controller["state_timeout_s"])
    franky_version = importlib.metadata.version("franky-control")
    output_root = project_path(config["output_dir"])
    existing_trials = sorted(
        path for path in output_root.glob("*_bolt_axial_preload") if path.is_dir()
    ) if output_root.exists() else []
    maximum_trials = int(config["maximum_physical_trial_count"])

    print("=== GUARDED STATIONARY BOLT AXIAL PRELOAD ===")
    print(
        f"mode:                    "
        f"{'DRY_RUN' if args.dry_run else 'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}"
    )
    print(f"robot IP:                {config['robot_ip']}")
    print(f"installed Franky:        {franky_version}")
    print(f"required setup:          {config['required_setup']}")
    print(f"approved target:         +{target_force:.6f} N along +Kz")
    print(f"hard absolute cutoff:    {hard_force:.6f} N from live baseline")
    print(
        f"wrong-direction stop:    -{float(wrong_direction['threshold_N']):.3f} N "
        f"for {float(wrong_direction['persistence_s']):.3f} s"
    )
    print("commanded motion:        measured current-pose hold; NO ROTATION")
    print(f"impedance implementation:{controller['implementation']}")
    print("force constraint:        live-computed base Z only; base X/Y remain position controlled")
    print(f"impedance stiffness:     {controller['translational_stiffness_N_m']:.1f} N/m / {controller['rotational_stiffness_Nm_rad']:.1f} Nm/rad")
    print(f"command duration/timeout:{duration_s:.1f} s / {timeout_s:.1f} s")
    print("force ramp:              NOT AVAILABLE in installed Franky 1.1.4 motion")
    print("rotation/threading:      DISABLED")
    print("Franka protections:      UNCHANGED")
    print(f"execution enabled:       {config['execution_enabled']}")
    print(f"physical trials:         {len(existing_trials)} / {maximum_trials}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if args.execute and config.get("execution_enabled") is not True:
        raise SystemExit(
            "[SAFETY] axial-preload execution remains disabled pending read-only preview"
        )
    if args.execute and len(existing_trials) >= maximum_trials:
        raise SystemExit(
            "[SAFETY] approved axial-preload trial count is exhausted; no connection attempted"
        )

    input("Press Enter to connect and establish a live unloaded baseline (Ctrl+C cancels): ")
    from franky import Affine, ExponentialImpedanceMotion, Frame, Gripper, Robot

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    start_T = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    commanded_T = np.asarray(state.O_T_EE_c.matrix, dtype=np.float64)
    ee_t_k = np.asarray(state.EE_T_K.matrix, dtype=np.float64)
    q = np.asarray(state.q, dtype=np.float64)
    axis_base = start_T[:3, 2].copy()
    axis_norm = float(np.linalg.norm(axis_base))
    if axis_norm > 0:
        axis_base /= axis_norm
    base_z_alignment = float(axis_base[2])
    runtime_blockers: list[str] = []

    if franky_version != "1.1.4":
        runtime_blockers.append("installed Franky version differs from the inspected 1.1.4 source")
    if not np.allclose(ee_t_k, np.eye(4), atol=1e-8):
        runtime_blockers.append("EE_T_K is not identity; EE +Z cannot be treated as K +Z")
    minimum_alignment = float(controller["minimum_abs_base_z_alignment"])
    if abs(base_z_alignment) < minimum_alignment:
        runtime_blockers.append("live +Kz is not sufficiently aligned with a single base-Z constraint")
    if abs(base_z_alignment) < 1e-9:
        commanded_base_z_force = np.nan
    else:
        commanded_base_z_force = target_force / base_z_alignment
    if not np.isfinite(commanded_base_z_force) or abs(commanded_base_z_force) >= hard_force:
        runtime_blockers.append("computed base-Z force command is invalid or exceeds the hard cutoff")

    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    width_margin = float(config["gripper_width_comparison_margin_m"])
    accepted_width_min = float(head_width["minimum_width_m"]) - width_margin
    accepted_width_max = float(head_width["maximum_width_m"]) + width_margin
    if not grasped or not accepted_width_min <= width <= accepted_width_max:
        runtime_blockers.append("Hand state is inconsistent with the calibrated head grasp")

    margin = float(config["joint_mechanical_margin_rad"])
    if np.any(q < JOINT_MIN + margin) or np.any(q > JOINT_MAX - margin):
        runtime_blockers.append("live joints violate guarded mechanical ranges")

    reference_T = np.asarray(reference["T_base_ee"], dtype=np.float64).mean(axis=0)
    translation_error = float(np.linalg.norm(start_T[:3, 3] - reference_T[:3, 3]))
    signed_axial_start_error, lateral_start_error = decompose_translation_error(
        start_T[:3, 3], reference_T[:3, 3], axis_base
    )
    tool_axis_tilt = tool_axis_tilt_rad(reference_T[:3, :3], start_T[:3, :3])
    if abs(signed_axial_start_error) > float(config["maximum_start_axial_error_m"]):
        runtime_blockers.append("live axial position is too far from calibrated THREAD_READY")
    if lateral_start_error > float(config["maximum_start_lateral_error_m"]):
        runtime_blockers.append("live lateral position is too far from calibrated THREAD_READY")
    if np.degrees(tool_axis_tilt) > float(config["maximum_start_tool_axis_tilt_deg"]):
        runtime_blockers.append("live tool axis is too tilted from calibrated THREAD_READY")

    minimum = np.asarray(config["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(config["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(start_T[:3, 3] < minimum) or np.any(start_T[:3, 3] > maximum):
        runtime_blockers.append("live TCP position is outside the local workspace")

    baseline_rows = []
    deadline = time.perf_counter() + float(config["baseline_duration_s"])
    while time.perf_counter() < deadline:
        sample = robot.state
        baseline_rows.append(
            [
                float(sample.K_F_ext_hat_K[2]),
                float(np.max(np.abs(np.asarray(sample.dq, dtype=np.float64)))),
            ]
        )
        time.sleep(1.0 / float(config["baseline_sample_hz"]))
    baseline_rows = np.asarray(baseline_rows)
    baseline_Fz = float(np.mean(baseline_rows[:, 0]))
    baseline_Fz_std = float(np.std(baseline_rows[:, 0]))
    maximum_dq = float(np.max(baseline_rows[:, 1]))
    if baseline_Fz_std > float(config["maximum_baseline_Fz_std_N"]):
        runtime_blockers.append("live baseline F_Kz is not stationary enough")
    if maximum_dq > float(config["maximum_baseline_abs_dq_rad_s"]):
        runtime_blockers.append("arm moved during the live baseline")

    transverse_force = abs(commanded_base_z_force) * np.sqrt(
        max(0.0, 1.0 - base_z_alignment * base_z_alignment)
    )
    commanded_pose_translation_error = float(
        np.linalg.norm(commanded_T[:3, 3] - start_T[:3, 3])
    )
    commanded_pose_rotation_error = rotation_angle_rad(
        commanded_T[:3, :3], start_T[:3, :3]
    )
    desired_force_base = axis_base * target_force
    predicted_wrench_linear_first = np.concatenate(
        [desired_force_base, np.zeros(3, dtype=np.float64)]
    )
    predicted_wrench_angular_first = np.concatenate(
        [np.zeros(3, dtype=np.float64), desired_force_base]
    )
    jacobian = np.asarray(robot.model.zero_jacobian(Frame.EndEffector, state), dtype=np.float64)
    coriolis = np.asarray(robot.model.coriolis(state), dtype=np.float64)
    predicted_tau_linear_first = jacobian.T @ predicted_wrench_linear_first + coriolis
    predicted_tau_angular_first = jacobian.T @ predicted_wrench_angular_first + coriolis
    print(f"\nlive xyz:               {start_T[:3, 3]}")
    print(f"live +Kz in base:        {axis_base}")
    print(f"Kz/base-Z alignment:     {base_z_alignment:+.6f}")
    print(f"computed base-Z command: {commanded_base_z_force:+.6f} N")
    print(f"axial projection:        {commanded_base_z_force * base_z_alignment:+.6f} N")
    print(f"off-axis component:      {transverse_force:.6f} N")
    print(
        f"measured/commanded pose: {commanded_pose_translation_error * 1000:.3f} mm / "
        f"{np.degrees(commanded_pose_rotation_error):.3f} deg (diagnostic only)"
    )
    print(f"Jacobian row norms:      {np.array2string(np.linalg.norm(jacobian, axis=1), precision=5)}")
    print(
        "predicted tau [F,T]:    "
        f"{np.array2string(predicted_tau_linear_first, precision=5)} Nm; "
        f"norm={np.linalg.norm(predicted_tau_linear_first - coriolis):.6f}"
    )
    print(
        "diagnostic tau [T,F]:   "
        f"{np.array2string(predicted_tau_angular_first, precision=5)} Nm; "
        f"norm={np.linalg.norm(predicted_tau_angular_first - coriolis):.6f}"
    )
    print(f"Hand width/grasped:      {width:.6f} m / {grasped}")
    print(f"accepted head width:     {accepted_width_min:.6f} .. {accepted_width_max:.6f} m")
    print(
        f"THREAD_READY error:      total {translation_error * 1000:.3f} mm; "
        f"axial {signed_axial_start_error * 1000:+.3f} mm; "
        f"lateral {lateral_start_error * 1000:.3f} mm / "
        f"tool-axis tilt {np.degrees(tool_axis_tilt):.3f} deg (yaw ignored)"
    )
    print(f"live baseline F_Kz:      {baseline_Fz:+.6f} N (std={baseline_Fz_std:.6f} N)")
    print(f"maximum baseline |dq|:   {maximum_dq:.6f} rad/s")
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] stationary axial-preload execution refused")
    if config.get("read_only_preview_commissioning", {}).get("status") != "PASSED":
        raise SystemExit("[SAFETY] read-only preview is not marked PASSED")

    response = input(
        "Visually confirm the bolt is upright, securely head-held, self-supported, "
        "NOT engaged, and unloaded. Type RUN_STATIONARY_AXIAL_PRELOAD; anything else aborts: "
    )
    if response != "RUN_STATIONARY_AXIAL_PRELOAD":
        print("[preload] aborted without motion")
        return

    guard = AxialPreloadGuard(
        baseline_Fz_N=baseline_Fz,
        axis_base=axis_base,
        start_xyz_m=start_T[:3, 3].copy(),
        hard_absolute_axial_delta_N=hard_force,
        wrong_direction_threshold_N=float(wrong_direction["threshold_N"]),
        wrong_direction_persistence_s=float(wrong_direction["persistence_s"]),
        maximum_axial_travel_m=float(config["maximum_axial_travel_m"]),
        maximum_lateral_translation_m=float(config["maximum_lateral_translation_m"]),
        maximum_duration_s=timeout_s,
    )
    capacity = int(config["callback_capacity"])
    trace = np.full((capacity, 34), np.nan, dtype=np.float64)
    trace_count = 0
    callback_start = None

    def callback(callback_state, unused_step, unused_rel, unused_abs, command) -> None:
        nonlocal trace_count, callback_start
        del unused_step, unused_rel, unused_abs
        robot_time = float(callback_state.time.to_sec())
        if callback_start is None:
            callback_start = robot_time
        elapsed = robot_time - callback_start
        wrench_K = np.asarray(callback_state.K_F_ext_hat_K, dtype=np.float64)
        wrench_O = np.asarray(callback_state.O_F_ext_hat_K, dtype=np.float64)
        pose = np.asarray(callback_state.O_T_EE.matrix, dtype=np.float64)
        q_live = np.asarray(callback_state.q, dtype=np.float64)
        dq_live = np.asarray(callback_state.dq, dtype=np.float64)
        axial_delta = guard.update(elapsed, wrench_K, pose[:3, 3])
        displacement = pose[:3, 3] - start_T[:3, 3]
        axial_travel = float(np.dot(displacement, axis_base))
        lateral_travel = float(
            np.linalg.norm(displacement - axial_travel * axis_base)
        )
        if trace_count < capacity:
            trace[trace_count] = [
                elapsed,
                *q_live,
                float(np.max(np.abs(dq_live))),
                *wrench_K,
                *wrench_O,
                *pose[:3, 3],
                *np.asarray(command.tau_J, dtype=np.float64),
                axial_delta,
                axial_travel,
                lateral_travel,
            ]
            trace_count += 1

    monitor_done = threading.Event()

    def monitor_stop() -> None:
        start = time.perf_counter()
        planned_deadline = start + duration_s
        hard_deadline = start + timeout_s
        while not monitor_done.wait(0.002):
            now = time.perf_counter()
            if guard.stop_reason is None and now >= planned_deadline:
                guard.stop_reason = "planned_duration_complete"
            if guard.stop_reason is None and now >= hard_deadline:
                guard.stop_reason = "timeout"
            if guard.stop_reason is not None:
                robot.stop()
                return

    def interrupt(signum, frame) -> None:
        del signum, frame
        guard.stop_reason = "operator_abort"
        robot.stop()

    signal.signal(signal.SIGINT, interrupt)
    force_constraints = [None, None, commanded_base_z_force, None, None, None]
    motion = ExponentialImpedanceMotion(
        Affine(start_T),
        translational_stiffness=float(controller["translational_stiffness_N_m"]),
        rotational_stiffness=float(controller["rotational_stiffness_Nm_rad"]),
        force_constraints=force_constraints,
        exponential_decay=float(controller["exponential_decay"]),
    )
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

    recorded = trace[:trace_count]
    final_state = robot.state
    final_T = np.asarray(final_state.O_T_EE.matrix, dtype=np.float64)
    if len(recorded):
        tail_start = max(0.0, recorded[-1, 0] - 0.5)
        tail = recorded[recorded[:, 0] >= tail_start]
        steady_axial_delta = float(np.mean(tail[:, -3]))
    else:
        steady_axial_delta = np.nan
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": guard.stop_reason or "motion_completed",
        "motion_error": motion_error,
        "approved_target_Kz_N": target_force,
        "commanded_base_Z_N": commanded_base_z_force,
        "base_z_alignment_with_Kz": base_z_alignment,
        "baseline_F_Kz_N": baseline_Fz,
        "steady_final_0p5s_axial_delta_N": steady_axial_delta,
        "minimum_axial_delta_N": guard.minimum_axial_delta_N,
        "maximum_axial_delta_N": guard.maximum_axial_delta_N,
        "maximum_absolute_axial_delta_N": guard.maximum_absolute_axial_delta_N,
        "maximum_axial_travel_m": guard.maximum_axial_travel_observed_m,
        "maximum_lateral_translation_m": guard.maximum_lateral_translation_observed_m,
        "start_T_base_ee": start_T.tolist(),
        "final_T_base_ee": final_T.tolist(),
        "final_hand_is_grasped": bool(gripper.is_grasped),
        "rotation_enabled": False,
        "threading_enabled": False,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"{stamp}_bolt_axial_preload"
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "control_trace.npz", trace=recorded)
    print(f"[result] stop reason:     {result['stop_reason']}")
    print(f"[result] steady axial:    {steady_axial_delta:+.6f} N")
    print(f"[result] axial range:     {guard.minimum_axial_delta_N:+.6f} .. {guard.maximum_axial_delta_N:+.6f} N")
    print(f"[result] axial travel:    {guard.maximum_axial_travel_observed_m * 1000:.3f} mm")
    print(f"[result] lateral travel:  {guard.maximum_lateral_translation_observed_m * 1000:.3f} mm")
    print(f"[saved] {output}")
    print("No rotation, threading, ratchet, release, or camera command was issued.")
    print("Use Pilot to retract upward and unload the bolt before any next step.")


if __name__ == "__main__":
    main()
