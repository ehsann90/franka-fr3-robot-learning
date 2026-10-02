#!/usr/bin/env python3
"""Read-only capture of a gentle operator-applied load along the bolt axis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.axial_load_calibration import summarize_axial_load_phases  # noqa: E402
from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402


DEFAULT_OUTPUT = FR3_ROOT / "recordings" / "bolt_axial_load_calibrations_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-duration", type=float, default=3.0)
    parser.add_argument("--load-setup-delay", type=float, default=2.0)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if not 2.0 <= args.window_duration <= 10.0:
        raise SystemExit("--window-duration must be between 2 and 10 seconds")
    if not 1.0 <= args.load_setup_delay <= 10.0:
        raise SystemExit("--load-setup-delay must be between 1 and 10 seconds")
    output_root = args.output_dir.expanduser().resolve()

    print("=== BOLT READ-ONLY AXIAL LOAD CALIBRATION ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:                {args.robot_ip}")
    print("required start:          bolt upright and securely held at THREAD_READY")
    print("verified insertion axis: +Kz")
    print("measured axial signals:  K_F_ext_hat_K[2], K_F_ext_hat_K[5]")
    print("arm/Hand commands:       DISABLED")
    print("rotation/threading:      DISABLED")
    print("threshold calculation:   DISABLED")
    print(f"phase window:            {args.window_duration:.1f} s")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return

    print("\nUse only supported Franka Desk/Pilot hand-guiding controls.")
    print("The calibration records your manually applied load; it never commands force or motion.")
    print("Use the gentlest clearly repeatable straight-down load. Do not rotate the bolt.")
    print("Stop immediately if the print deforms, the bolt tilts, or anything looks wrong.")
    input("Press Enter with the bolt stationary and without deliberate preload (Ctrl+C cancels): ")

    from franky import Robot  # noqa: PLC0415

    robot = Robot(args.robot_ip)
    recorder = PilotedTrajectoryRecorder(robot, None, robot_hz=50.0, gripper_hz=1.0)
    preflight = recorder.preflight()
    xyz = preflight["translation_xyz_m"]
    print(f"[capture] preflight xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})")
    recorder.start(start_gripper_thread=False)
    try:
        recorder.add_event("unloaded_baseline_started")
        print(f"[phase 1/3] hands off: recording unloaded baseline for {args.window_duration:.1f} s")
        time.sleep(args.window_duration)
        recorder.add_event("unloaded_baseline_complete")

        input(
            "Prepare to apply a gentle straight-down (+Kz) manual load. "
            "Press Enter when ready; do not rotate: "
        )
        print(f"[phase 2/3] apply the gentle load now; measurement starts in {args.load_setup_delay:.1f} s")
        time.sleep(args.load_setup_delay)
        recorder.add_event("axial_load_window_started")
        print(f"[phase 2/3] hold the gentle load steady for {args.window_duration:.1f} s")
        time.sleep(args.window_duration)
        recorder.add_event("axial_load_window_complete")

        input(
            "Release all deliberate load without rotating; keep the bolt supported and upright. "
            "Press Enter when hands-off again: "
        )
        recorder.add_event("unloaded_recovery_started")
        print(f"[phase 3/3] hands off: recording recovery for {args.window_duration:.1f} s")
        time.sleep(args.window_duration)
        recorder.add_event("unloaded_recovery_complete")
    except KeyboardInterrupt:
        print("\n[capture] interrupted; stopping read-only capture")
        raise
    finally:
        recorder.stop()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_bolt_axial_load_calibration"
    capture_dir = output_root / name
    accepted = len(recorder.errors) == 0
    recorder.write(
        capture_dir,
        {
            "trajectory_name": name,
            "capture_type": "operator_applied_axial_load_read_only",
            "task_name": "bolt",
            "robot_ip": args.robot_ip,
            "condition": "thread_ready_supported",
            "screw_axis_K": [0.0, 0.0, 1.0],
            "axis_verification": {
                "world_down_alignment_with_positive_Kz": 0.9989,
                "recorded_lowering_alignment_with_positive_Kz": 0.9918,
            },
            "operator_applied_load": True,
            "arm_commands_enabled": False,
            "gripper_connection_enabled": False,
            "gripper_commands_enabled": False,
            "rotation_threading_enabled": False,
            "thresholds_derived": False,
            "notes": args.notes,
        },
        accepted=accepted,
    )
    try:
        summary = summarize_axial_load_phases(recorder.samples, recorder.events)
    except ValueError as error:
        raise SystemExit(f"[capture] raw data saved but phase summary failed: {error}") from error
    (capture_dir / "axial_load_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[saved] samples/errors:  {len(recorder.samples)} / {len(recorder.errors)}")
    print(f"[saved] axial delta Fz:  {summary['axial_force_loaded_minus_unloaded_N']:+.6f} N")
    print(f"[saved] axial delta Tz:  {summary['axial_torque_loaded_minus_unloaded_Nm']:+.6f} Nm")
    print(f"[saved] full K delta:    {summary['loaded_minus_unloaded_K_wrench']}")
    print(f"[saved] {capture_dir}")
    print("[IMPORTANT] No force, torque, contact, or safety threshold was selected.")


if __name__ == "__main__":
    main()
