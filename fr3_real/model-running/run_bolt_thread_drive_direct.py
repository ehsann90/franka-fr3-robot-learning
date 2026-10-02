#!/usr/bin/env python3
"""Launch the direct-libfranka first guarded bolt tightening drive."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_thread_drive_v1.json"


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
    direct = config["direct_controller"]
    binary = project_path(direct["binary"])
    output_root = project_path(direct["output_dir"])
    dynamic = json.loads(project_path(config["dynamic_model_calibration"]).read_text())
    head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    model_q7 = ",".join(format(float(row[0]), ".17g") for row in dynamic["knots"])
    model_tz = ",".join(format(float(row[1]), ".17g") for row in dynamic["knots"])

    print("=== GUARDED BOLT THREAD DRIVE LAUNCHER ===")
    print(
        "mode:                     "
        + ("DRY_RUN" if args.dry_run else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE")
    )
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"binary:                   {binary}")
    print(f"scope:                    {config['scope']}")
    print(f"combined authorization:   RECORDED — {config['combined_motion_authorization']['date']}")
    print(f"controller status:        {config['controller_status']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    print("automatic ratchet/Hand:   DISABLED")
    print("camera recording:         DISABLED")

    blockers: list[str] = []
    if not binary.is_file():
        blockers.append("direct thread-drive binary is missing; rebuild required")
    if len(dynamic.get("knots", [])) < 2:
        blockers.append("dynamic torque model is missing or invalid")
    if float(config["drive_angle_deg"]) != 89.0:
        blockers.append("drive angle differs from the authorized 89 degrees")
    if float(config["planned_axial_command_ceiling_N"]) != 1.5:
        blockers.append("axial command ceiling differs from the authorization")
    if float(config["planned_axial_command_slew_N_s"]) != 0.5:
        blockers.append("axial command slew differs from the authorization")
    if float(config["approved_hard_absolute_axial_delta_N"]) != 2.0:
        blockers.append("hard axial cutoff differs from the authorization")
    if abs(float(config["approved_hard_negative_residual_Tz_Nm"]) + 0.5383051567216975) > 1e-12:
        blockers.append("residual-torque cutoff differs from the authorization")
    if binary.is_file():
        linked = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=True
        ).stdout
        if direct["linked_libfranka"] not in linked:
            blockers.append("binary is not linked to the verified repository libfranka 0.21 path")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links the rejected Franky libfranka 0.18 path")
    if args.execute and config.get("execution_enabled") is not True:
        blockers.append("execution remains locked until the direct-controller preview passes")

    if blockers:
        print("\nLauncher blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if blockers:
        raise SystemExit("[SAFETY] guarded thread-drive launcher refused")

    print("Reading the Franka Hand state without commanding it...")
    from franky import Gripper

    gripper = Gripper(config["robot_ip"])
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    del gripper
    margin = float(config["gripper_width_comparison_margin_m"])
    accepted_min = float(head["minimum_width_m"]) - margin
    accepted_max = float(head["maximum_width_m"]) + margin
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    print(f"accepted head width:      {accepted_min:.6f} .. {accepted_max:.6f} m")
    if not grasped or not accepted_min <= width <= accepted_max:
        raise SystemExit("[SAFETY] Hand state is inconsistent with the calibrated head grasp")

    output = "-"
    direct_mode = "read-only-preview"
    if args.execute:
        output_root.mkdir(parents=True, exist_ok=True)
        existing = list(output_root.glob("*_bolt_thread_drive.csv"))
        if len(existing) >= int(direct["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] the authorized physical-trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = str(output_root / f"{stamp}_bolt_thread_drive.csv")
        direct_mode = "execute"
    command = [
        str(binary), direct_mode, config["robot_ip"], output, model_q7, model_tz
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
