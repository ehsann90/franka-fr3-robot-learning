#!/usr/bin/env python3
"""Launch the guarded combined axial-admittance plus five-degree thread test."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_combined_thread_5_v1.json"


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


def write_dynamic_model(path: Path, document: dict) -> None:
    knots = np.asarray(document["knots"], dtype=np.float64)
    np.savetxt(path, knots[:, :2], delimiter=",",
               header="q7_rad,median_K_Tz_Nm", comments="")


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
    dynamic = json.loads(project_path(config["dynamic_model_calibration"]).read_text())
    head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    mode_name = "DRY_RUN" if args.dry_run else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE"
    blockers: list[str] = []
    exact = {
        "approved_measured_target_N": 1.200360,
        "force_acquisition_duration_s": 3.0,
        "maximum_axial_speed_m_s": 0.000500,
        "maximum_axial_travel_m": 0.003,
        "maximum_lateral_travel_m": 0.0003,
        "hard_absolute_axial_delta_N": 2.0,
        "wrong_direction_delta_N": -0.5,
        "wrong_direction_persistence_s": 0.1,
        "hard_negative_residual_Tz_Nm": -0.5383051567216975,
    }
    for key, value in exact.items():
        if abs(float(config[key]) - value) > 1e-12:
            blockers.append(f"{key} differs from the explicitly approved value")
    derived_gain = float(config["maximum_axial_speed_m_s"]) / float(config["approved_measured_target_N"])
    if abs(float(config["force_to_velocity_gain_m_s_per_N"]) - derived_gain) > 1e-12:
        blockers.append("force-to-velocity gain is not speed cap / approved force target")
    angle_deg = float(config["rotation_angle_deg"])
    if angle_deg not in (5.0, 40.0, 45.0, 50.0, 85.0):
        blockers.append("rotation angle is not an explicitly compiled commissioning stage")
    speed = float(config["peak_angular_speed_rad_s"])
    if speed not in (0.03, 0.06, 0.09, 0.12):
        blockers.append("peak angular speed is not a separately compiled bounded value")
    expected_binary = {
        (5.0, 0.03): "fr3_bolt_combined_thread_5",
        (40.0, 0.03): "fr3_bolt_combined_thread_40",
        (45.0, 0.03): "fr3_bolt_combined_thread_45",
        (50.0, 0.03): "fr3_bolt_combined_thread_50",
        (85.0, 0.03): "fr3_bolt_combined_thread_85",
        (5.0, 0.06): "fr3_bolt_combined_thread_5_2x",
        (40.0, 0.06): "fr3_bolt_combined_thread_40_2x",
        (45.0, 0.06): "fr3_bolt_combined_thread_45_2x",
        (50.0, 0.06): "fr3_bolt_combined_thread_50_2x",
        (85.0, 0.06): "fr3_bolt_combined_thread_85_2x",
        (5.0, 0.09): "fr3_bolt_combined_thread_5_3x",
        (40.0, 0.09): "fr3_bolt_combined_thread_40_3x",
        (45.0, 0.09): "fr3_bolt_combined_thread_45_3x",
        (50.0, 0.09): "fr3_bolt_combined_thread_50_3x",
        (85.0, 0.09): "fr3_bolt_combined_thread_85_3x",
        (5.0, 0.12): "fr3_bolt_combined_thread_5_4x",
        (40.0, 0.12): "fr3_bolt_combined_thread_40_4x",
        (45.0, 0.12): "fr3_bolt_combined_thread_45_4x",
        (50.0, 0.12): "fr3_bolt_combined_thread_50_4x",
        (85.0, 0.12): "fr3_bolt_combined_thread_85_4x",
    }.get((angle_deg, speed))
    if expected_binary is None or binary.name != expected_binary:
        blockers.append("configuration does not match the compiled combined controller")
    expected_timeout = 3.0 + np.pi * np.deg2rad(angle_deg) / (2.0 * speed) + 5.0
    if abs(float(config["derived_state_timeout_s"]) - expected_timeout) > 1e-5:
        blockers.append("state timeout is not acquisition + motion + settling")
    if not binary.is_file():
        blockers.append("combined direct controller binary is missing")
    if binary.is_file():
        linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
        if config["linked_libfranka"] not in linked:
            blockers.append("binary is not linked to repository libfranka 0.21")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links Franky's private libfranka 0.18")
    if args.execute and config.get("execution_enabled") is not True:
        blockers.append("execution remains locked pending read-only preview")

    print(f"=== GUARDED COMBINED +{angle_deg:.0f}-DEGREE THREAD TEST LAUNCHER ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"sequence:                 3 s axial acquisition -> +{angle_deg:.0f} deg tightening")
    print("force target/gain:        +1.200360 N / 0.416542 mm/(s*N)")
    print(f"axial/yaw speed caps:     0.500 mm/s / {speed:.3f} rad/s")
    print("maximum axial/lateral:    3.000 / 0.300 mm")
    print("hard axial/residual stop: +/-2.000 N / -0.538305 Nm")
    print("ratchet/seated/release:   DISABLED")
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
        raise SystemExit("[SAFETY] combined-thread launcher refused")

    input(f"Keep the bolt securely head-held and unloaded ({config['required_setup']}). Press Enter to validate the Hand (Ctrl+C cancels): ")
    from franky import Gripper

    gripper = Gripper(config["robot_ip"])
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    if not grasped or not accepted_head_width(config, head, width):
        raise SystemExit("[SAFETY] Hand state is inconsistent with the calibrated head grasp")
    visual = input("Confirm the bolt matches the required setup and is unloaded. Press Enter to continue; type anything to abort: ")
    if visual:
        print("[abort] Hand remains closed; no arm control session was started")
        return
    del gripper
    gc.collect()
    time.sleep(0.5)

    output = "-"
    direct_mode = "read-only-preview"
    confirmation = "PREVIEW_ONLY"
    if args.execute:
        output_root = project_path(config["output_dir"])
        output_root.mkdir(parents=True, exist_ok=True)
        suffix = f"combined_thread_{int(angle_deg)}"
        existing = list(output_root.glob(f"*_{suffix}.csv"))
        if len(existing) >= int(config["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] authorized combined trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = str(output_root / f"{stamp}_{suffix}.csv")
        direct_mode = "execute"
        confirmation = "COMBINED_THREAD_AUTHORIZED"
    with tempfile.TemporaryDirectory(prefix="fr3_combined_thread_") as temp:
        model_path = Path(temp) / "dynamic_model.csv"
        write_dynamic_model(model_path, dynamic)
        completed = subprocess.run([
            str(binary), direct_mode, config["robot_ip"], output,
            confirmation, str(model_path),
        ], check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
