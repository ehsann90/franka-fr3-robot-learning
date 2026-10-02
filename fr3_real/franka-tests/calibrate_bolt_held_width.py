#!/usr/bin/env python3
"""Measure and persist Franka Hand width while a bolt is manually held."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = FR3_ROOT / "calibration" / "bolt_grasp_width_v1.json"


def write_new_calibration(path: Path, document: dict) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; this utility never overwrites a grasp calibration"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(document, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def summarize_widths(widths: list[float]) -> dict:
    values = np.asarray(widths, dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("exactly five finite width readings are required")
    return {
        "sample_count": 5,
        "width_samples_m": values.tolist(),
        "median_width_m": float(np.median(values)),
        "minimum_width_m": float(values.min()),
        "maximum_width_m": float(values.max()),
        "peak_to_peak_width_m": float(np.ptp(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--grasp", default="CENTER")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"[calibration] refusing to overwrite existing file: {output}")

    print("=== BOLT HELD-WIDTH CALIBRATION ===")
    print(f"robot IP:          {args.robot_ip}")
    print(f"grasp style:       {args.grasp.strip().upper()}")
    print(f"output:            {output}")
    print("arm commands:      DISABLED")
    print("Hand commands:     DISABLED")
    print("operation:         five read-only Hand state samples")
    print("\nBefore connecting, use supported Desk/Pilot controls to establish the named grasp.")
    if args.grasp.strip().upper() == "THREAD_READY_HEAD":
        print("  1. keep the bolt upright and self-supported in the guide")
        print("  2. securely hold the bolt by its head at THREAD_READY")
        print("  3. apply no deliberate axial preload or rotation")
        print("  4. hold the arm and bolt stationary")
    else:
        print("  1. place the bolt at S1")
        print(f"  2. pilot to the {args.grasp.strip().upper()} grasp")
        print("  3. close securely around the bolt")
        print("  4. hold the arm and bolt stationary")
    input("Press Enter when the bolt is securely held (Ctrl+C cancels): ")

    from franky import Gripper  # noqa: PLC0415

    gripper = Gripper(args.robot_ip)
    widths: list[float] = []
    grasped: list[bool] = []
    temperatures: list[int] = []
    max_widths: list[float] = []
    for index in range(5):
        state = gripper.state
        width = float(state.width)
        held = bool(state.is_grasped)
        widths.append(width)
        grasped.append(held)
        temperatures.append(int(state.temperature))
        max_widths.append(float(state.max_width))
        print(f"[read {index + 1}/5] width={width:.6f} m is_grasped={held}")
        if index < 4:
            time.sleep(0.1)
    if not all(grasped):
        raise SystemExit(
            "[calibration] rejected: Hand did not report is_grasped=True for every sample"
        )
    summary = summarize_widths(widths)
    if summary["median_width_m"] <= 0.0 or summary["median_width_m"] >= min(max_widths):
        raise SystemExit("[calibration] rejected: measured width is outside the Hand range")

    document = {
        "schema_version": 1,
        "calibration_type": "bolt_held_width",
        "robot_ip": args.robot_ip,
        "screw_slot": "S1",
        "screw_grasp_style": args.grasp.strip().upper(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
        "all_samples_is_grasped": True,
        "temperature_c_range": [min(temperatures), max(temperatures)],
        "hand_max_width_m": float(np.median(max_widths)),
        "grasp_command_parameters": "NOT_CALIBRATED",
        "notes": "Measured after operator securely grasped the printed bolt using Desk/Pilot",
    }
    write_new_calibration(output, document)
    print(f"[saved] median held width={summary['median_width_m']:.6f} m")
    print(f"[saved] peak-to-peak noise={summary['peak_to_peak_width_m']:.6f} m")
    print(f"[saved] {output}")
    print("[saved] no arm or Hand command was issued")


if __name__ == "__main__":
    main()
