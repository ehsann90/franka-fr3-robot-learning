#!/usr/bin/env python3
"""Launch the direct-libfranka stationary bolt axial-ramp commissioning test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_axial_preload_v1.json"


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
    direct = config["direct_low_level_controller"]
    binary = project_path(direct["binary"])
    output_root = project_path(direct["output_dir"])
    head_calibration = json.loads(
        project_path(config["thread_ready_head_width_calibration"]).read_text()
    )
    target = float(config["approved_axial_force_target_Kz_N"])
    ramp = float(direct["axial_force_ramp_duration_s"])

    print("=== DIRECT LIBFRANKA BOLT PRELOAD LAUNCHER ===")
    print(
        "mode:                     "
        + ("DRY_RUN" if args.dry_run else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE")
    )
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"binary:                   {binary}")
    print(f"approved measured target: +{target:.6f} N")
    print(
        f"command schedule:         0 -> +{float(direct['maximum_commanded_axial_force_N']):.3f} N "
        f"at {float(direct['maximum_commanded_axial_force_slew_N_s']):.3f} N/s"
    )
    print(f"ceiling hold:             {float(direct['axial_force_ceiling_hold_duration_s']):.1f} s")
    print("force feedback:           DISABLED; Kp=0; Ki=0")
    print("rotation/threading:       DISABLED")
    print("Franka protections:       UNCHANGED")
    print(f"controller status:        {direct['status']}")
    print(f"execution enabled:        {direct['execution_enabled']}")

    blockers: list[str] = []
    if not binary.is_file():
        blockers.append("direct-libfranka controller binary is missing; rebuild it")
    if target != 1.20036 or ramp != 3.0:
        blockers.append("configured force/ramp differs from the explicitly approved test")
    if float(direct["axial_force_kp"]) != 0.0 or float(direct["axial_force_ki_per_s"]) != 0.0:
        blockers.append("feedback gains must remain zero for ceiling characterization")
    if float(direct["maximum_commanded_axial_force_N"]) != 1.5:
        blockers.append("force-command ceiling differs from the explicitly approved P-only test")
    if float(direct["maximum_commanded_axial_force_slew_N_s"]) != 0.5:
        blockers.append("force-command slew differs from the explicitly approved ceiling test")
    if float(direct["axial_force_ceiling_hold_duration_s"]) != 1.0:
        blockers.append("ceiling hold differs from the explicitly approved one-second duration")
    if float(direct["axial_stiffness_N_m"]) != 0.0 or float(direct["axial_damping_Ns_m"]) != 0.0:
        blockers.append("screw-axis stiffness and damping must remain zero")
    if binary.is_file():
        linked = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=True
        ).stdout
        if "/home/salaar/Job/libfranka/build/libfranka.so.0.21" not in linked:
            blockers.append("binary is not linked to the verified repository libfranka 0.21 path")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links the rejected Franky libfranka 0.18 path")
    if args.execute and direct.get("execution_enabled") is not True:
        blockers.append("execution remains locked pending successful read-only preview review")

    if blockers:
        print("\nExecution blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if blockers:
        raise SystemExit("[SAFETY] direct preload launcher refused")

    input(
        "Press Enter to read the Franka Hand state before the direct arm preview "
        "(Ctrl+C cancels): "
    )
    from franky import Gripper

    gripper = Gripper(config["robot_ip"])
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    margin = float(config["gripper_width_comparison_margin_m"])
    accepted_min = float(head_calibration["minimum_width_m"]) - margin
    accepted_max = float(head_calibration["maximum_width_m"]) + margin
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    print(f"accepted head width:      {accepted_min:.6f} .. {accepted_max:.6f} m")
    if not grasped or not accepted_min <= width <= accepted_max:
        raise SystemExit("[SAFETY] Hand state is inconsistent with the calibrated head grasp")
    del gripper

    command = [str(binary), "read-only-preview", config["robot_ip"]]
    if args.execute:
        output_root.mkdir(parents=True, exist_ok=True)
        existing = list(output_root.glob("*_direct_axial_preload.csv"))
        if len(existing) >= int(direct["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] direct preload physical-trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = output_root / f"{stamp}_direct_axial_preload.csv"
        command = [str(binary), "execute", config["robot_ip"], str(output)]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
