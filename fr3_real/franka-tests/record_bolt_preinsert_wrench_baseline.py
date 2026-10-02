#!/usr/bin/env python3
"""Record a stationary, read-only FR3 wrench baseline at bolt PREINSERT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402
from expert_demo.wrench_baseline import summarize_wrench_baseline  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bolt_preinsert_v1.json"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--condition",
        choices=("preinsert_free", "thread_ready_supported"),
        default="preinsert_free",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    try:
        raw_config = config_path.read_bytes()
        config = json.loads(raw_config)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read configuration {config_path}: {error}") from error
    baseline = config.get("sensor_baseline", {})
    if baseline.get("enabled") is not True:
        raise SystemExit("sensor_baseline.enabled must be true")
    if config.get("contact_threading", {}).get("enabled") is not False:
        raise SystemExit("contact/threading must remain disabled during baseline capture")
    sample_hz = float(baseline["robot_sample_hz"])
    duration_s = float(args.duration if args.duration is not None else baseline["duration_s"])
    if not 2.0 <= duration_s <= 60.0:
        raise SystemExit("baseline duration must be between 2 and 60 seconds")
    output_root = project_path(baseline["output_dir"])

    print("=== BOLT STATIONARY WRENCH BASELINE ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:                {config['robot_ip']}")
    print(f"sample rate/duration:    {sample_hz:.1f} Hz / {duration_s:.1f} s")
    print(f"condition:               {args.condition}")
    print(f"output root:             {output_root}")
    print("arm commands:            DISABLED")
    print("Hand connection/command: DISABLED (existing grasp is not changed)")
    print("contact/threading:       DISABLED / NOT_CALIBRATED")
    print("signals:                 q, dq, joint torque, O/K external wrench, EE pose")
    print("K-frame Z:               CANDIDATE screw axis; not experimentally verified")
    print("threshold calculation:   DISABLED")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return

    if args.condition == "preinsert_free":
        print("\nHold the securely grasped bolt stationary at validated NO-CONTACT PREINSERT.")
        print("Keep visible clearance from the hole.")
    else:
        print("\nKeep the upright bolt securely regrasped and stationary at THREAD_READY.")
        print("The guide may support the bolt, but apply no deliberate preload or rotation.")
    print("Do not touch the robot, bolt, bracket, or fixture during capture.")
    input("Press Enter to connect read-only and begin the stationary baseline (Ctrl+C cancels): ")

    from franky import Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    recorder = PilotedTrajectoryRecorder(robot, None, robot_hz=sample_hz, gripper_hz=1.0)
    preflight = recorder.preflight()
    xyz = preflight["translation_xyz_m"]
    print(f"[baseline] preflight xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})")
    print(f"[baseline] recording {duration_s:.1f} stationary seconds ...")
    recorder.start(start_gripper_thread=False)
    recorder.add_event("stationary_baseline_started")
    try:
        time.sleep(duration_s)
    except KeyboardInterrupt:
        print("\n[baseline] interrupted; stopping without discarding captured raw state")
    finally:
        recorder.add_event("stationary_baseline_stopped")
        recorder.stop()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    capture_name = f"{stamp}_bolt_{args.condition}_wrench_baseline"
    capture_dir = output_root / capture_name
    metadata = {
        "trajectory_name": capture_name,
        "capture_type": "stationary_bolt_wrench_baseline",
        "condition": args.condition,
        "task_name": "bolt",
        "robot_ip": config["robot_ip"],
        "screw_slot": "S1",
        "screw_grasp_style": "CENTER",
        "bolt_held": True,
        "preinsert_no_contact": args.condition == "preinsert_free",
        "thread_ready_supported": args.condition == "thread_ready_supported",
        "operator_confirmed_stationary_setup": True,
        "notes": args.notes,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "arm_commands_enabled": False,
        "gripper_connection_enabled": False,
        "gripper_commands_enabled": False,
        "contact_threading_enabled": False,
        "wrench_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "wrench_units": ["N", "N", "N", "Nm", "Nm", "Nm"],
        "wrench_sign_note": (
            "libfranka convention: robot-on-environment is positive; "
            "environment-on-robot is negative"
        ),
        "k_frame_z_status": "candidate screw axis only; experimental verification required",
        "frame_transforms_requested": ["T_base_ee", "F_T_EE", "EE_T_K"],
    }
    accepted = len(recorder.errors) == 0 and len(recorder.samples) >= 2
    recorder.write(capture_dir, metadata, accepted=accepted)
    try:
        summary = summarize_wrench_baseline(recorder.samples)
    except ValueError as error:
        raise SystemExit(f"[baseline] raw capture saved but summary failed: {error}") from error
    (capture_dir / "wrench_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[saved] samples/errors:  {len(recorder.samples)} / {len(recorder.errors)}")
    print(f"[saved] achieved rate:   {summary['achieved_sample_hz']:.2f} Hz")
    print(f"[saved] maximum |dq|:    {summary['maximum_measured_abs_dq_rad_s']:.6f} rad/s")
    for signal in ("K_F_ext_hat_K", "O_F_ext_hat_K"):
        components = summary["wrench_statistics"][signal]["components"]
        means = [float(component["mean"]) for component in components]
        stds = [float(component["std"]) for component in components]
        print(f"[saved] {signal} mean: {means}")
        print(f"[saved] {signal} std:  {stds}")
    print(f"[saved] {capture_dir}")
    print("[IMPORTANT] No contact threshold was calculated. K-frame Z is still unverified.")
    if not accepted:
        raise SystemExit("[baseline] capture retained but rejected because sampling errors occurred")


if __name__ == "__main__":
    main()
