#!/usr/bin/env python3
"""Guarded, fast (production-speed) transport_preinsert_closed replay for a
bolt that is already grasped, without ever calling gripper.open() or
gripper.grasp().

This exists because the full orchestrator's bolt-pickup flow always opens
the Hand before approaching, and the fast per-stage arm runner
(commission_full_assembly_arm_2x.py) requires the Hand to read empty/open --
neither is safe to run against a bolt that is already correctly held. This
script only ever reads Gripper state; it validates the current hold against
the already-commissioned pickup_shaft_grasp physical-obstruction band from
configs/autonomous_demo_collection_v1.json (the same bounded-width fallback
check record_autonomous_bolt_engagement_demo.py uses), then runs the
existing fr3_retimed_joint_trajectory controller for the already-authorized
bolt_transport_preinsert_closed stage from full_assembly_4x_production_v1.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_demo.bolt_preinsert import build_bolt_preinsert_stage  # noqa: E402
from expert_demo.retimed_trajectory import write_retimed_csv  # noqa: E402
from record_autonomous_bolt_engagement_demo import (  # noqa: E402
    build_configured_accelerated_trajectory,
    pickup_grasp_evidence,
)

STAGE_NAME = "bolt_transport_preinsert_closed"
SOURCE_STAGE = "transport_preinsert_closed"
DEFAULT_TRAJECTORY_DIR = (
    FR3_ROOT / "recordings" / "piloted_bolt_preinsert_trajectories_v1"
    / "20260812_130029_bolt_preinsert_pilot_0001_S1_CENTER"
)
DEFAULT_SPEED_CONFIG = FR3_ROOT / "configs" / "full_assembly_4x_production_v1.json"
DEFAULT_COLLECTION_CONFIG = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
DEFAULT_WORKSPACE_MIN = [0.43, -0.49, 0.04]
DEFAULT_WORKSPACE_MAX = [0.64, 0.35, 0.25]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--speed-config", type=Path, default=DEFAULT_SPEED_CONFIG)
    parser.add_argument("--collection-config", type=Path, default=DEFAULT_COLLECTION_CONFIG)
    args = parser.parse_args()

    speed_config = json.loads(args.speed_config.expanduser().resolve().read_text())
    collection = json.loads(args.collection_config.expanduser().resolve().read_text())
    pickup_grasp = collection["tasks"]["bolt_insert_engage"]["pickup_shaft_grasp"]
    robot_ip = collection["robot_ip"]

    plan = build_bolt_preinsert_stage(
        str(args.trajectory_dir.expanduser().resolve()), SOURCE_STAGE, command_rate_hz=15.0
    )
    trajectory = build_configured_accelerated_trajectory(plan, speed_config, STAGE_NAME)

    print("=== FAST BOLT TRANSPORT-TO-PREINSERT (BOLT ALREADY GRASPED) ===")
    print(f"mode:                     {'EXECUTE' if args.execute else 'READ_ONLY_PREVIEW'}")
    print(f"robot IP:                 {robot_ip}")
    print(f"trajectory:               {args.trajectory_dir}")
    print(f"speed config:             {args.speed_config.name}")
    print(f"retimed duration:         {trajectory.duration_s:.3f} s")
    print(f"retimed peak |dq|:        {trajectory.peak_abs_velocity_rad_s:.6f} rad/s")
    print(f"retimed peak |ddq|:       {trajectory.peak_abs_acceleration_rad_s2:.6f} rad/s^2")
    print(f"retimed peak |jerk|:      {trajectory.peak_abs_jerk_rad_s3:.6f} rad/s^3")
    print(f"physical hold width band: [{pickup_grasp['physical_hold_width_min_m']}, "
          f"{pickup_grasp['physical_hold_width_max_m']}] m")
    print("gripper commands:         NEVER ISSUED BY THIS SCRIPT (open or grasp)")

    blockers = []
    stage_authorized = speed_config["stages"].get(STAGE_NAME, {}).get(
        "production_execution_authorized"
    )
    if stage_authorized is not True:
        blockers.append(f"{STAGE_NAME} is not production_execution_authorized in {args.speed_config.name}")
    for label, measured, cap in (
        ("velocity", trajectory.peak_abs_velocity_rad_s, speed_config["velocity_cap_rad_s"]),
        ("acceleration", trajectory.peak_abs_acceleration_rad_s2, speed_config["acceleration_cap_rad_s2"]),
        ("jerk", trajectory.peak_abs_jerk_rad_s3, speed_config["jerk_cap_rad_s3"]),
    ):
        if measured > float(cap):
            blockers.append(f"planned peak {label} exceeds cap")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] fast bolt transport refused")

    input("Press Enter to connect and inspect live state (Ctrl+C cancels; no command issued): ")
    from franky import Gripper, Robot  # noqa: PLC0415

    robot = Robot(robot_ip)
    gripper = Gripper(robot_ip)
    width = float(gripper.state.width)
    is_grasped = bool(gripper.is_grasped)
    strict, bounded = pickup_grasp_evidence(
        pickup_grasp, accepted=is_grasped, is_grasped=is_grasped, width_m=width
    )
    current_q = np.asarray(robot.state.q, dtype=np.float64)
    target_q = trajectory.expected_start_q
    start_error = float(np.max(np.abs(current_q - target_q)))

    print(f"\nlive Hand width/grasped: {width:.6f} m / {is_grasped}")
    print(f"strict franka success:   {strict}")
    print(f"bounded physical hold:   {bounded}")
    print(f"live start error:        {start_error:.6f} rad")

    live_blockers = []
    if not (strict or bounded):
        live_blockers.append(
            "Hand does not show evidence of holding the bolt (neither strict nor bounded-width check passed)"
        )
    if start_error > 0.02:
        live_blockers.append(
            f"live arm is not close enough to this stage's start ({start_error:.6f} rad); "
            "this script does not perform any reset, run the fast bolt_approach_open stage "
            "or an appropriate guarded reset first"
        )
    if live_blockers:
        print("\nBlockers:")
        for blocker in live_blockers:
            print(f"  - {blocker}")
        if args.read_only_preview:
            raise SystemExit(3)
        raise SystemExit("[SAFETY] fast bolt transport refused")

    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return

    binary = project_path(speed_config["binary"])
    if not binary.is_file():
        raise SystemExit(f"missing compiled controller: {binary}")
    with tempfile.TemporaryDirectory(prefix="fr3_bolt_transport_fast_") as temp:
        plan_path = Path(temp) / "retimed_plan.csv"
        write_retimed_csv(plan_path, trajectory)
        output_root = project_path("recordings/bolt_transport_preinsert_fast_trials_v1")
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        telemetry = output_root / f"{stamp}_{STAGE_NAME}_object_aware_4x.csv"
        command = [
            str(binary), "execute", robot_ip, str(plan_path), str(telemetry),
            "RUN_RETIMED_2X_STAGE", STAGE_NAME,
            *(str(value) for value in DEFAULT_WORKSPACE_MIN),
            *(str(value) for value in DEFAULT_WORKSPACE_MAX),
        ]
        completed = subprocess.run(command, input="\n\n", text=True, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"[DONE] telemetry saved: {telemetry}")
    print("[DONE] no Hand command was issued by this script")


if __name__ == "__main__":
    main()
