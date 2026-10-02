#!/usr/bin/env python3
"""Read-only capture of operator-selected bolt ratchet q7 cable-safe endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.ratchet_limits import summarize_ratchet_limits  # noqa: E402
from expert_demo.snapshots import capture_robot_snapshot  # noqa: E402


DEFAULT_OUTPUT = FR3_ROOT / "calibration" / "bolt_ratchet_limits_v1.json"


def write_new_atomic(path: Path, document: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing calibration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()

    print("=== BOLT RATCHET CABLE-SAFE LIMIT CALIBRATION ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:                {args.robot_ip}")
    print(f"output:                  {output}")
    print("required setup:          NO BOLT; clear local wrist/Hand path")
    print("motion/Hand commands:    DISABLED")
    print("limits:                  selected physically by operator in Pilot")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if output.exists():
        raise SystemExit(f"calibration already exists; choose a new --output: {output}")

    print("\nRemove the bolt. Keep the bracket/fixture clear and use supported Pilot controls only.")
    input("Press Enter to connect the read-only arm client (Ctrl+C cancels): ")
    from franky import Robot  # noqa: PLC0415

    robot = Robot(args.robot_ip)
    input(
        "In Pilot, place the empty Hand at the local THREAD_READY geometry and rotate q7 to "
        "your conservative PREWIND cable-safe endpoint. Press Enter while stationary: "
    )
    prewind = capture_robot_snapshot(
        robot, None, snapshot_kind="bolt_ratchet_prewind_cable_safe_limit"
    )
    input(
        "In Pilot, rotate only through the tightening direction to your conservative FORWARD "
        "cable-safe endpoint. Stop before strain or the robot limit; press Enter stationary: "
    )
    forward = capture_robot_snapshot(
        robot, None, snapshot_kind="bolt_ratchet_forward_cable_safe_limit"
    )
    try:
        summary = summarize_ratchet_limits(prewind, forward)
    except ValueError as error:
        raise SystemExit(f"[rejected] endpoint calibration is invalid: {error}") from error
    print(f"q7 prewind/forward:       {summary['q7_prewind_rad']:+.6f} / {summary['q7_forward_rad']:+.6f} rad")
    print(f"available forward range: {summary['q7_available_forward_range_deg']:.3f} deg")
    print(f"other-joint change max:  {summary['maximum_other_joint_change_rad']:.6f} rad")
    response = input(
        "Visually confirm both endpoints are conservative and cable-safe. Press Enter to save; "
        "type anything to reject: "
    ).strip()
    if response:
        raise SystemExit("[rejected] no calibration was written")
    document = {
        "schema_version": 1,
        "calibration_kind": "operator_selected_bolt_ratchet_cable_safe_limits",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "robot_ip": args.robot_ip,
        "tightening_rotation_sign_Kz": "positive",
        "prewind_snapshot": prewind,
        "forward_snapshot": forward,
        "summary": summary,
        "automatic_motion_performed": False,
        "gripper_connection_or_command_performed": False
    }
    write_new_atomic(output, document)
    print(f"[saved] {output}")
    print("No robot or Hand command was issued. Use Pilot to return to a neutral safe pose.")


if __name__ == "__main__":
    main()
