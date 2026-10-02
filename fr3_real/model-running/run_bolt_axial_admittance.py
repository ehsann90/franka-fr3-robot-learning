#!/usr/bin/env python3
"""Launch the guarded stationary no-rotation bolt axial-admittance test."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_axial_admittance_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def accepted_head_width(config: dict, calibration: dict, width: float) -> bool:
    margin = float(config["gripper_width_comparison_margin_m"])
    primary = (
        float(calibration["minimum_width_m"]) - margin,
        float(calibration["maximum_width_m"]) + margin,
    )
    alternate = tuple(map(float, config["accepted_alternate_head_width_m"]))
    return primary[0] <= width <= primary[1] or alternate[0] <= width <= alternate[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.expanduser().resolve().read_text())
    binary = project_path(config["binary"])
    head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    mode_name = "DRY_RUN" if args.dry_run else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE"
    blockers: list[str] = []
    exact = {
        "approved_measured_target_N": 1.200360,
        "maximum_axial_speed_m_s": 0.000500,
        "target_ramp_duration_s": 3.0,
        "maximum_axial_travel_m": 0.003,
        "maximum_lateral_travel_m": 0.0003,
        "hard_absolute_axial_delta_N": 2.0,
        "wrong_direction_delta_N": -0.5,
        "wrong_direction_persistence_s": 0.1,
    }
    for key, value in exact.items():
        if abs(float(config[key]) - value) > 1e-12:
            blockers.append(f"{key} differs from the explicitly approved value")
    derived_gain = float(config["maximum_axial_speed_m_s"]) / float(config["approved_measured_target_N"])
    if abs(float(config["force_to_velocity_gain_m_s_per_N"]) - derived_gain) > 1e-12:
        blockers.append("force-to-velocity gain is not exactly speed cap / force target")
    if float(config["commanded_rotation_rad_s"]) != 0.0:
        blockers.append("rotation must remain exactly zero")
    if not binary.is_file():
        blockers.append("direct axial-admittance binary is missing")
    if binary.is_file():
        linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
        if config["linked_libfranka"] not in linked:
            blockers.append("binary is not linked to repository libfranka 0.21")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links Franky's private libfranka 0.18")
    if args.execute and config.get("execution_enabled") is not True:
        blockers.append("execution remains locked pending read-only preview")

    print("=== GUARDED STATIONARY BOLT AXIAL ADMITTANCE LAUNCHER ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"scope:                    {config['scope']}")
    print("force target/ramp:        +1.200360 N / 3.000 s")
    print("gain/speed cap:           0.416542 mm/(s*N) / 0.500 mm/s")
    print("maximum axial/lateral:    3.000 / 0.300 mm")
    print("hard axial stop:          +/-2.000 N")
    print("wrong-direction stop:     -0.500 N for 0.100 s")
    print("rotation/Hand/threading:  DISABLED")
    print(f"controller status:        {config['controller_status']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    if blockers:
        print("\nLauncher blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if blockers:
        raise SystemExit("[SAFETY] axial-admittance launcher refused")

    input("Keep the upright bolt securely head-held, unengaged, and unloaded. Press Enter to validate the Hand (Ctrl+C cancels): ")
    from franky import Gripper

    gripper = Gripper(config["robot_ip"])
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    if not grasped or not accepted_head_width(config, head, width):
        raise SystemExit("[SAFETY] Hand state is inconsistent with the calibrated head grasp")
    visual = input("Confirm the bolt is upright, self-supported, NOT engaged, and unloaded. Press Enter to continue; type anything to abort: ")
    if visual:
        print("[abort] Hand remains closed; no arm control session was started")
        return
    del gripper
    gc.collect()
    time.sleep(0.5)

    direct_mode = "read-only-preview"
    confirmation = "PREVIEW_ONLY"
    output = "-"
    if args.execute:
        output_root = project_path(config["output_dir"])
        output_root.mkdir(parents=True, exist_ok=True)
        existing = list(output_root.glob("*_axial_admittance.csv"))
        if len(existing) >= int(config["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] authorized axial-admittance trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = str(output_root / f"{stamp}_axial_admittance.csv")
        direct_mode = "execute"
        confirmation = "AXIAL_ADMITTANCE_AUTHORIZED"
    completed = subprocess.run([
        str(binary), direct_mode, config["robot_ip"], output, confirmation
    ], check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
