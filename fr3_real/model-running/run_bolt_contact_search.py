#!/usr/bin/env python3
"""Guarded +Kz bolt CONTACT_SEARCH; never rotate or maintain thread preload."""

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

from expert_demo.contact_search import ContactSearchGuard, rotation_angle_rad  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_contact_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.expanduser().resolve().read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read contact-search configuration: {error}") from error
    contact = config.get("contact_detection", {})
    motion = config.get("automatic_contact_motion", {})
    blockers = []
    if contact.get("status") != "PROVISIONAL_OPERATOR_APPROVED":
        blockers.append("contact detector is not operator-approved")
    if contact.get("read_only_validation_passed") is not True:
        blockers.append("contact detector has not passed read-only validation")
    if motion.get("enabled") is not True:
        blockers.append("automatic contact motion is disabled")
    if config.get("rotation_threading", {}).get("enabled") is not False:
        blockers.append("rotation/threading must remain disabled")
    required = (
        "axial_force_hard_limit_relative_to_baseline_N",
        "maximum_approach_distance_m",
        "approach_speed_m_s",
        "state_timeout_s",
        "relative_dynamics_factor",
        "maximum_start_translation_error_m",
        "maximum_start_rotation_error_deg",
        "maximum_baseline_abs_dq_rad_s",
        "joint_margin_rad",
        "workspace_bounds_m",
        "thread_ready_reference_capture",
        "gripper_held_width_max_m",
        "output_dir",
    )
    for key in required:
        if motion.get(key) in (None, "NOT_CALIBRATED"):
            blockers.append(f"automatic_contact_motion.{key} is NOT_CALIBRATED")
    if blockers:
        raise SystemExit("invalid contact-search configuration:\n  - " + "\n  - ".join(blockers))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    detector = config["contact_detection"]
    motion_config = config["automatic_contact_motion"]
    threshold_delta = float(detector["threshold_relative_to_live_baseline_N"])
    persistence_s = float(detector["persistence_duration_s"])
    hard_delta = float(motion_config["axial_force_hard_limit_relative_to_baseline_N"])
    travel_m = float(motion_config["maximum_approach_distance_m"])
    speed_m_s = float(motion_config["approach_speed_m_s"])
    timeout_s = float(motion_config["state_timeout_s"])

    print("=== GUARDED BOLT CONTACT_SEARCH ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                {config['robot_ip']}")
    print("required start:          bolt upright, held, and unloaded at THREAD_READY")
    print("commanded direction:     +EE-z after runtime EE_T_K=identity verification")
    print(f"approach speed/travel:   {speed_m_s * 1000:.3f} mm/s / {travel_m * 1000:.3f} mm")
    print(f"contact detector:        +{threshold_delta:.3f} N for {persistence_s:.3f} s")
    print(f"hard axial cutoff:       +{hard_delta:.3f} N relative to live baseline")
    print(f"state timeout:           {timeout_s:.3f} s")
    print("axial force target:      NOT_CALIBRATED / NOT USED")
    print("rotation/threading:      DISABLED")
    print("Franka protections:      UNCHANGED")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return

    input("Press Enter to connect and establish a live unloaded baseline (Ctrl+C cancels): ")
    from franky import Affine, CartesianMotion, Gripper, ReferenceType, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    start_T = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    ee_t_k = np.asarray(state.EE_T_K.matrix, dtype=np.float64)
    q = np.asarray(state.q, dtype=np.float64)
    runtime_blockers: list[str] = []
    if not np.allclose(ee_t_k, np.eye(4), atol=1e-8):
        runtime_blockers.append("EE_T_K is not identity; +EE-z cannot be treated as +Kz")
    if not bool(gripper.is_grasped):
        runtime_blockers.append("Hand does not report a held bolt")
    width = float(gripper.state.width)
    if width > float(motion_config["gripper_held_width_max_m"]):
        runtime_blockers.append("Hand width is inconsistent with the calibrated bolt grasp")
    margin = float(motion_config["joint_margin_rad"])
    if np.any(q < JOINT_MIN + margin) or np.any(q > JOINT_MAX - margin):
        runtime_blockers.append("current joints violate guarded range")

    reference_arrays = np.load(
        project_path(motion_config["thread_ready_reference_capture"]) / "trajectory_arrays.npz"
    )
    reference_T = np.asarray(reference_arrays["T_base_ee"], dtype=np.float64).mean(axis=0)
    translation_error = float(np.linalg.norm(start_T[:3, 3] - reference_T[:3, 3]))
    rotation_error = rotation_angle_rad(reference_T[:3, :3], start_T[:3, :3])
    if translation_error > float(motion_config["maximum_start_translation_error_m"]):
        runtime_blockers.append("live position is too far from calibrated THREAD_READY")
    if np.degrees(rotation_error) > float(motion_config["maximum_start_rotation_error_deg"]):
        runtime_blockers.append("live orientation is too far from calibrated THREAD_READY")

    target_T = start_T.copy()
    target_T[:3, 3] += start_T[:3, 2] * travel_m
    minimum = np.asarray(motion_config["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    maximum = np.asarray(motion_config["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(start_T[:3, 3] < minimum) or np.any(start_T[:3, 3] > maximum):
        runtime_blockers.append("live start pose is outside the contact-search workspace")
    if np.any(target_T[:3, 3] < minimum) or np.any(target_T[:3, 3] > maximum):
        runtime_blockers.append("bounded target pose is outside the contact-search workspace")

    baseline_records = []
    baseline_deadline = time.perf_counter() + 3.0
    while time.perf_counter() < baseline_deadline:
        sample = robot.state
        baseline_records.append(
            (
                float(sample.K_F_ext_hat_K[2]),
                float(np.max(np.abs(np.asarray(sample.dq, dtype=np.float64)))),
            )
        )
        time.sleep(0.02)
    baseline_forces = np.asarray([record[0] for record in baseline_records])
    maximum_baseline_dq = max(record[1] for record in baseline_records)
    baseline = float(np.mean(baseline_forces))
    baseline_std = float(np.std(baseline_forces))
    if maximum_baseline_dq > float(motion_config["maximum_baseline_abs_dq_rad_s"]):
        runtime_blockers.append("arm was not stationary during the live baseline")

    print(f"\nlive xyz:                {start_T[:3, 3]}")
    print(f"target xyz:              {target_T[:3, 3]}")
    print(f"THREAD_READY error:      {translation_error * 1000:.3f} mm / {np.degrees(rotation_error):.3f} deg")
    print(f"Hand width/grasped:      {width:.6f} m / {bool(gripper.is_grasped)}")
    print(f"EE_T_K identity error:   {float(np.max(np.abs(ee_t_k - np.eye(4)))):.3e}")
    print(f"live baseline F_Kz:      {baseline:+.6f} N (std={baseline_std:.6f} N)")
    print(f"maximum baseline |dq|:   {maximum_baseline_dq:.6f} rad/s")
    if runtime_blockers:
        print("Execution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] CONTACT_SEARCH execution refused")
    if motion_config.get("physical_execution_commissioned") is not True:
        raise SystemExit(
            "[SAFETY] physical execution is not commissioned; pass a read-only preview first"
        )

    input(
        "Confirm the bolt is upright/unloaded and the path is clear. "
        "Press Enter to execute the bounded no-rotation CONTACT_SEARCH: "
    )
    axis_base = start_T[:3, 2].copy()
    guard = ContactSearchGuard(
        baseline_force_n=baseline,
        contact_delta_n=threshold_delta,
        contact_persistence_s=persistence_s,
        hard_delta_n=hard_delta,
        maximum_travel_m=travel_m,
        maximum_duration_s=timeout_s,
        start_xyz_m=start_T[:3, 3],
        axis_base=axis_base,
    )
    capacity = int(np.ceil((timeout_s + 2.0) * 1100))
    trace = np.full((capacity, 14), np.nan, dtype=np.float64)
    trace_count = 0
    callback_start = None

    def callback(callback_state, *unused) -> None:
        nonlocal trace_count, callback_start
        del unused
        robot_time = float(callback_state.time.to_sec())
        if callback_start is None:
            callback_start = robot_time
        elapsed = robot_time - callback_start
        wrench = np.asarray(callback_state.K_F_ext_hat_K, dtype=np.float64)
        pose = np.asarray(callback_state.O_T_EE.matrix, dtype=np.float64)
        guard.update(elapsed, float(wrench[2]), pose[:3, 3])
        if trace_count < capacity:
            trace[trace_count, 0] = elapsed
            trace[trace_count, 1:7] = wrench
            trace[trace_count, 7:10] = pose[:3, 3]
            trace[trace_count, 10:14] = [
                float(wrench[2]) - baseline,
                guard.maximum_travel_observed_m,
                float(np.max(np.abs(np.asarray(callback_state.dq, dtype=np.float64)))),
                1.0 if guard.stop_reason is not None else 0.0,
            ]
            trace_count += 1

    stop_thread_done = threading.Event()

    def stop_monitor() -> None:
        deadline = time.perf_counter() + timeout_s
        while not stop_thread_done.wait(0.002):
            if guard.stop_reason is None and time.perf_counter() >= deadline:
                guard.stop_reason = "timeout"
            if guard.stop_reason is not None:
                robot.stop()
                return

    def interrupt(signum, frame) -> None:
        del signum, frame
        guard.stop_reason = "operator_abort"
        robot.stop()

    signal.signal(signal.SIGINT, interrupt)
    robot.translation_velocity_limit.set(speed_m_s)
    relative_target = Affine([0.0, 0.0, travel_m])
    motion = CartesianMotion(
        relative_target,
        ReferenceType.Relative,
        relative_dynamics_factor=float(motion_config["relative_dynamics_factor"]),
    )
    motion.register_callback(callback)
    monitor = threading.Thread(target=stop_monitor, daemon=True)
    monitor.start()
    motion_error = None
    try:
        robot.move(motion)
    except BaseException as error:
        motion_error = repr(error)
        if guard.stop_reason is None:
            raise
    finally:
        stop_thread_done.set()
        monitor.join(timeout=2.0)

    final_state = robot.state
    final_T = np.asarray(final_state.O_T_EE.matrix, dtype=np.float64)
    final_force_delta = float(final_state.K_F_ext_hat_K[2]) - baseline
    final_travel = float(np.dot(final_T[:3, 3] - start_T[:3, 3], axis_base))
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": guard.stop_reason or "motion_completed_without_contact",
        "baseline_force_Kz_N": baseline,
        "baseline_std_N": baseline_std,
        "contact_threshold_delta_N": threshold_delta,
        "hard_force_delta_N": hard_delta,
        "peak_force_delta_N": guard.peak_force_delta_n,
        "maximum_travel_observed_m": guard.maximum_travel_observed_m,
        "final_force_delta_N": final_force_delta,
        "final_travel_m": final_travel,
        "start_T_base_ee": start_T.tolist(),
        "final_T_base_ee": final_T.tolist(),
        "motion_error": motion_error,
        "rotation_threading_enabled": False,
        "axial_force_target_used": False,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = project_path(motion_config["output_dir"]) / f"{stamp}_bolt_contact_search"
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "control_trace.npz", trace=trace[:trace_count])
    print(f"[result] stop reason:     {result['stop_reason']}")
    print(f"[result] peak delta:      {result['peak_force_delta_N']:+.6f} N")
    print(f"[result] travel:          {result['maximum_travel_observed_m'] * 1000:.3f} mm")
    print(f"[saved] {output}")
    print("No rotation, force maintenance, threading, gripper, or camera command was issued.")
    print("Use Pilot to retract upward and unload the bolt before any next step.")


if __name__ == "__main__":
    main()
