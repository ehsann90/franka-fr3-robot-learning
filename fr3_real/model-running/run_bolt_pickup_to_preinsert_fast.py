#!/usr/bin/env python3
"""Guarded, fast (production-speed) bolt pickup through no-contact preinsert:
approach -> grasp -> mandatory visual confirmation -> transport.

record_autonomous_bolt_engagement_demo.py already contains this exact
sequence internally, but only runs it at production speed when combined with
--full-assembly (which would also redo the bracket stage against the old
bracket trajectory -- not appropriate here, since the bracket was placed via
a separate, newer candidate). This script extracts just the bolt-only
sequence at the already-authorized full_assembly_4x_production_v1.json
speed, reusing the exact same plan builder, retiming helper, compiled
controller, grasp-evidence validation (pickup_grasp_evidence, the same
bounded-physical-width fallback used by the full orchestrator), and
mandatory operator visual-confirmation gate as the original.
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

import math

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_demo.bolt_preinsert import build_bolt_preinsert_stage  # noqa: E402
from expert_demo.retimed_trajectory import write_retimed_csv  # noqa: E402
from record_autonomous_bolt_engagement_demo import (  # noqa: E402
    build_configured_accelerated_trajectory,
    pickup_grasp_evidence,
)

DEFAULT_TRAJECTORY_DIR = (
    FR3_ROOT / "recordings" / "piloted_bolt_preinsert_trajectories_v1"
    / "20260812_130029_bolt_preinsert_pilot_0001_S1_CENTER"
)
DEFAULT_SPEED_CONFIG = FR3_ROOT / "configs" / "full_assembly_4x_production_v1.json"
DEFAULT_COLLECTION_CONFIG = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
WORKSPACE_MIN = [0.43, -0.49, 0.04]
WORKSPACE_MAX = [0.64, 0.35, 0.25]
OUTPUT_ROOT = "recordings/bolt_pickup_preinsert_fast_trials_v1"
# Same mechanical ranges the other guarded reset scripts in this project use.
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
RESET_JOINT_MARGIN_RAD = 0.12
RESET_RELATIVE_DYNAMICS_FACTOR = 0.05
RESET_MAXIMUM_JOINT_DELTA_RAD = 0.5
RESET_TERMINAL_TOLERANCE_RAD = 0.01


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def build_stage(trajectory_dir: Path, source_stage: str, speed_config: dict, stage_key: str):
    plan = build_bolt_preinsert_stage(str(trajectory_dir), source_stage, command_rate_hz=15.0)
    trajectory = build_configured_accelerated_trajectory(plan, speed_config, stage_key)
    return plan, trajectory


def offline_blockers(stage_key: str, speed_config: dict, trajectory) -> list[str]:
    blockers = []
    authorized = speed_config["stages"].get(stage_key, {}).get("production_execution_authorized")
    if authorized is not True:
        blockers.append(f"{stage_key} is not production_execution_authorized")
    for label, measured, cap in (
        ("velocity", trajectory.peak_abs_velocity_rad_s, speed_config["velocity_cap_rad_s"]),
        ("acceleration", trajectory.peak_abs_acceleration_rad_s2, speed_config["acceleration_cap_rad_s2"]),
        ("jerk", trajectory.peak_abs_jerk_rad_s3, speed_config["jerk_cap_rad_s3"]),
    ):
        if measured > float(cap):
            blockers.append(f"{stage_key} planned peak {label} exceeds cap")
    return blockers


def run_binary_stage(binary: Path, robot_ip: str, trajectory, stage_key: str, telemetry_dir: Path) -> None:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    telemetry = telemetry_dir / f"{stamp}_{stage_key}_object_aware_4x.csv"
    with tempfile.TemporaryDirectory(prefix="fr3_bolt_pickup_fast_") as temp:
        plan_path = Path(temp) / "retimed_plan.csv"
        write_retimed_csv(plan_path, trajectory)
        command = [
            str(binary), "execute", robot_ip, str(plan_path), str(telemetry),
            "RUN_RETIMED_2X_STAGE", stage_key,
            *(str(value) for value in WORKSPACE_MIN),
            *(str(value) for value in WORKSPACE_MAX),
        ]
        completed = subprocess.run(command, input="\n\n", text=True, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"[DONE] {stage_key} telemetry saved: {telemetry}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--reset-to-start", action="store_true",
                        help="guarded slow segmented move to bolt_approach_open's (idle-trimmed) "
                             "start pose; refuses unless the Hand reads empty/open")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--speed-config", type=Path, default=DEFAULT_SPEED_CONFIG)
    parser.add_argument("--collection-config", type=Path, default=DEFAULT_COLLECTION_CONFIG)
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    speed_config = json.loads(args.speed_config.expanduser().resolve().read_text())
    collection = json.loads(args.collection_config.expanduser().resolve().read_text())
    pickup_grasp = collection["tasks"]["bolt_insert_engage"]["pickup_shaft_grasp"]
    robot_ip = collection["robot_ip"]
    binary = project_path(speed_config["binary"])

    approach_plan, approach_trajectory = build_stage(
        trajectory_dir, "approach_open", speed_config, "bolt_approach_open"
    )
    transport_plan, transport_trajectory = build_stage(
        trajectory_dir, "transport_preinsert_closed", speed_config, "bolt_transport_preinsert_closed"
    )
    blockers = offline_blockers("bolt_approach_open", speed_config, approach_trajectory)
    blockers += offline_blockers("bolt_transport_preinsert_closed", speed_config, transport_trajectory)
    if not binary.is_file():
        blockers.append(f"missing compiled controller: {binary}")

    print("=== FAST BOLT PICKUP TO PREINSERT (approach -> grasp -> confirm -> transport) ===")
    print(f"mode:                     "
          f"{'EXECUTE' if args.execute else 'RESET_TO_START' if args.reset_to_start else 'READ_ONLY_PREVIEW'}")
    print(f"robot IP:                 {robot_ip}")
    print(f"trajectory:               {trajectory_dir}")
    print(f"speed config:             {args.speed_config.name}")
    print(f"approach duration:        {approach_trajectory.duration_s:.3f} s "
          f"(peak |dq|={approach_trajectory.peak_abs_velocity_rad_s:.4f} rad/s)")
    print(f"transport duration:       {transport_trajectory.duration_s:.3f} s "
          f"(peak |dq|={transport_trajectory.peak_abs_velocity_rad_s:.4f} rad/s)")
    print(f"grasp target/speed/force: {pickup_grasp['target_width_m']:.4f} m / "
          f"{pickup_grasp['speed_m_s']:.3f} m/s / {pickup_grasp['force_N']:.1f} N")
    print(f"physical hold width band: [{pickup_grasp['physical_hold_width_min_m']}, "
          f"{pickup_grasp['physical_hold_width_max_m']}] m")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] fast bolt pickup refused")

    input("Press Enter to connect and inspect live state (Ctrl+C cancels; no command issued yet): ")
    from franky import Gripper, Robot  # noqa: PLC0415

    robot = Robot(robot_ip)
    gripper = Gripper(robot_ip)
    width = float(gripper.state.width)
    is_grasped = bool(gripper.is_grasped)
    current_q = np.asarray(robot.state.q, dtype=np.float64)
    start_error = float(np.max(np.abs(current_q - approach_trajectory.expected_start_q)))

    print(f"\nlive Hand width/grasped: {width:.6f} m / {is_grasped}")
    print(f"live start error:        {start_error:.6f} rad (vs bolt_approach_open start)")

    live_blockers = []
    if is_grasped or width < 0.07:
        live_blockers.append("Hand does not read empty/open; clear it before a fresh pickup attempt")
    if args.reset_to_start:
        reset_delta = float(np.max(np.abs(approach_trajectory.expected_start_q - current_q)))
        reset_segments = max(1, math.ceil(reset_delta / RESET_MAXIMUM_JOINT_DELTA_RAD))
        lower = JOINT_MIN + RESET_JOINT_MARGIN_RAD
        upper = JOINT_MAX - RESET_JOINT_MARGIN_RAD
        reset_path = np.linspace(current_q, approach_trajectory.expected_start_q, num=201)
        if np.any(reset_path < lower) or np.any(reset_path > upper):
            live_blockers.append("reset path violates the guarded joint range")
        print(f"reset maximum q delta:   {reset_delta:.6f} rad")
        print(f"reset segments:          {reset_segments}")
        if live_blockers:
            print("\nBlockers:")
            for blocker in live_blockers:
                print(f"  - {blocker}")
            raise SystemExit("[SAFETY] reset refused")
        from franky import JointMotion, JointState  # noqa: PLC0415
        input("Keep the path to bolt_approach_open's start clear. Press Enter for the slow segmented reset: ")
        reset_start = current_q.copy()
        for segment_index in range(1, reset_segments + 1):
            target = reset_start + segment_index / reset_segments * (
                approach_trajectory.expected_start_q - reset_start
            )
            robot.move(JointMotion(JointState(target), relative_dynamics_factor=RESET_RELATIVE_DYNAMICS_FACTOR))
            error = float(np.max(np.abs(np.asarray(robot.state.q) - target)))
            print(f"[reset {segment_index}/{reset_segments}] terminal error={error:.6f} rad")
            if error > RESET_TERMINAL_TOLERANCE_RAD:
                raise SystemExit("[SAFETY] reset terminal tolerance failed")
        print("[DONE] reset complete. No Hand command was issued.")
        return
    if start_error > 0.05:
        live_blockers.append(
            f"live arm is not close enough to bolt_approach_open's start ({start_error:.6f} rad); "
            "run --reset-to-start first"
        )
    if live_blockers:
        print("\nBlockers:")
        for blocker in live_blockers:
            print(f"  - {blocker}")
        if args.read_only_preview:
            raise SystemExit(3)
        raise SystemExit("[SAFETY] fast bolt pickup refused")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return

    print("\nPlace the bolt precisely at S1 in the recorded orientation if not already done.")
    if input("Press Enter when S1 and the complete path are ready; type anything to abort: "):
        raise SystemExit("[bolt pickup] aborted before any motion")

    run_binary_stage(binary, robot_ip, approach_trajectory, "bolt_approach_open", project_path(OUTPUT_ROOT))

    print("[bolt] grasping while the arm is stationary")
    accepted = gripper.grasp(
        float(pickup_grasp["target_width_m"]),
        float(pickup_grasp["speed_m_s"]),
        float(pickup_grasp["force_N"]),
        float(pickup_grasp["epsilon_inner_m"]),
        float(pickup_grasp["epsilon_outer_m"]),
    )
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    strict, bounded = pickup_grasp_evidence(
        pickup_grasp, accepted=bool(accepted), is_grasped=grasped, width_m=width
    )
    print(f"[bolt pickup grasp] accepted={accepted} is_grasped={grasped} width={width:.6f} m "
          f"strict={strict} bounded={bounded}")
    if not (strict or bounded):
        raise SystemExit(
            "[SAFETY] Hand did not report a successful pickup by either the strict or "
            "bounded-width check; the Hand was NOT automatically reopened -- visually verify "
            "and clear it manually before retrying"
        )
    if not strict:
        print("[bolt pickup grasp] Franka's strict boolean disagrees with the bounded shaft "
              "obstruction; continuing on the bounded-width evidence")

    if input(
        "Visually confirm the bolt was picked up securely. Press Enter to transport; "
        "type anything to reject (Hand stays closed): "
    ):
        raise SystemExit(
            "[bolt pickup] operator rejected pickup grasp; Hand was NOT automatically reopened"
        )

    run_binary_stage(
        binary, robot_ip, transport_trajectory, "bolt_transport_preinsert_closed", project_path(OUTPUT_ROOT)
    )
    print("\n[SUCCESS] bolt reached NO-CONTACT PREINSERT at production speed; Hand remains closed")
    print("No contact, insertion, rotation, release, or threading command was issued.")


if __name__ == "__main__":
    main()
