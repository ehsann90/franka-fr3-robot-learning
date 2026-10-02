#!/usr/bin/env python3
"""Launch the guarded empty-Hand Cartesian-velocity yaw commissioning test."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_cartesian_yaw_empty_v1.json"


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
    binary = project_path(config["binary"])
    output_root = project_path(config["output_dir"])
    mode_name = (
        "DRY_RUN" if args.dry_run else
        "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE"
    )
    print("=== EMPTY-HAND CARTESIAN YAW LAUNCHER ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"scope:                    {config['scope']}")
    print(f"test angle/peak speed:    +{config['test_angle_deg']:.1f} deg / {config['peak_angular_speed_rad_s']:.3f} rad/s")
    print("linear velocity:          [0, 0, 0] m/s")
    print(f"max TCP translation:      {config['maximum_tcp_translation_m']*1000.0:.3f} mm")
    print("bolt/Hand/contact:        DISABLED; BOLT MUST BE REMOVED")
    print(f"controller status:        {config['controller_status']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    blockers: list[str] = []
    if not binary.is_file():
        blockers.append("Cartesian yaw binary is missing")
    speed = float(config["peak_angular_speed_rad_s"])
    expected_binary = {
        (5.0, 0.03): "fr3_cartesian_yaw_empty",
        (45.0, 0.03): "fr3_cartesian_yaw_empty_45",
        (5.0, 0.06): "fr3_cartesian_yaw_empty_2x",
        (45.0, 0.06): "fr3_cartesian_yaw_empty_45_2x",
    }.get((float(config["test_angle_deg"]), speed))
    if expected_binary is None or binary.name != expected_binary:
        blockers.append("test angle does not match an explicitly compiled commissioning binary")
    if speed not in (0.03, 0.06):
        blockers.append("peak speed is not one of the separately compiled bounded implementations")
    if list(map(float, config["commanded_linear_velocity_m_s"])) != [0.0, 0.0, 0.0]:
        blockers.append("configured linear velocity is not exactly zero")
    if binary.is_file():
        linked = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=True
        ).stdout
        if config["linked_libfranka"] not in linked:
            blockers.append("binary is not linked to the verified repository libfranka 0.21 path")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links Franky's private libfranka 0.18")
    if args.execute and config.get("execution_enabled") is not True:
        blockers.append("execution remains locked pending read-only preview")
    if blockers:
        print("\nLauncher blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    if blockers:
        raise SystemExit("[SAFETY] empty-Hand Cartesian yaw launcher refused")

    output = "-"
    direct_mode = "read-only-preview"
    confirmation = "PREVIEW_ONLY"
    if args.execute:
        output_root.mkdir(parents=True, exist_ok=True)
        existing = list(output_root.glob("*_empty_cartesian_yaw.csv"))
        if len(existing) >= int(config["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] authorized empty-Hand trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = str(output_root / f"{stamp}_empty_cartesian_yaw.csv")
        direct_mode = "execute"
        confirmation = "EMPTY_CARTESIAN_YAW_AUTHORIZED"
    command = [
        str(binary), direct_mode, config["robot_ip"], output, confirmation
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
