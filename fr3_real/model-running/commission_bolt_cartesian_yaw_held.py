#!/usr/bin/env python3
"""Stage the guarded bolt-held, deliberately unengaged Cartesian-yaw test."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.piloted_thread_replay import build_piloted_thread_drive  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_cartesian_yaw_held_5_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def write_dynamic_model(path: Path, document: dict) -> None:
    knots = np.asarray(document["knots"], dtype=np.float64)
    np.savetxt(
        path,
        knots[:, :2],
        delimiter=",",
        header="q7_rad,median_K_Tz_Nm",
        comments="",
    )


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
    modes.add_argument("--prepare-controller-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume-held",
        action="store_true",
        help="preserve the prepared Drive-1 pose and already closed Hand",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.resume_held and not (args.prepare_controller_preview or args.execute):
        raise SystemExit("--resume-held requires --prepare-controller-preview or --execute")

    config = json.loads(args.config.expanduser().resolve().read_text())
    binary = project_path(config["binary"])
    dynamic_path = project_path(config["dynamic_model_calibration"])
    dynamic = json.loads(dynamic_path.read_text())
    ratchet = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
    head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    drive = build_piloted_thread_drive(
        project_path(config["source_episode"]),
        cycle=int(config["source_cycle"]),
        command_rate_hz=float(config["command_rate_hz"]),
        maximum_q7_rad=float(config["maximum_source_q7_rad"]),
    )
    start_pose = drive.pose_targets[0]
    mode_name = (
        "DRY_RUN" if args.dry_run else
        "READ_ONLY_RESET_PREVIEW" if args.read_only_preview else
        "PREPARE_CONTROLLER_PREVIEW" if args.prepare_controller_preview else "EXECUTE"
    )
    blockers: list[str] = []
    if not binary.is_file():
        blockers.append("bolt-held Cartesian-yaw binary is missing")
    speed = float(config["peak_angular_speed_rad_s"])
    expected_binary = {
        (5.0, 0.03): "fr3_cartesian_yaw_held_5",
        (45.0, 0.03): "fr3_cartesian_yaw_held_45",
        (5.0, 0.06): "fr3_cartesian_yaw_held_5_2x",
        (45.0, 0.06): "fr3_cartesian_yaw_held_45_2x",
    }.get((float(config["test_angle_deg"]), speed))
    if expected_binary is None or binary.name != expected_binary:
        blockers.append("configuration does not match an explicitly compiled held-yaw binary")
    if list(map(float, config["commanded_linear_velocity_m_s"])) != [0.0, 0.0, 0.0]:
        blockers.append("configured linear velocity is not exactly zero")
    if config["axial_force_command_enabled"] is not False:
        blockers.append("axial-force command must remain disabled")
    if float(config["hard_absolute_axial_delta_N"]) != 2.0:
        blockers.append("axial cutoff differs from the approved value")
    if abs(float(config["hard_negative_residual_Tz_Nm"]) + 0.5383051567216975) > 1e-12:
        blockers.append("residual-torque cutoff differs from the approved value")
    if binary.is_file():
        linked = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, check=True).stdout
        if config["linked_libfranka"] not in linked:
            blockers.append("binary is not linked to the verified repository libfranka 0.21 path")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links Franky's private libfranka 0.18")
    if args.prepare_controller_preview and config.get("prepare_controller_preview_enabled") is not True:
        blockers.append("physical preparation is locked pending read-only reset preview")
    if args.execute and config.get("execution_enabled") is not True:
        blockers.append("held-yaw execution is locked pending prepared controller preview")

    print("=== GUARDED BOLT-HELD CARTESIAN YAW LAUNCHER ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"scope:                    {config['scope']}")
    print(f"command:                  +{config['test_angle_deg']:.1f} deg about live tool axis")
    print(f"peak angular speed:       {speed:.3f} rad/s")
    print("linear velocity:          EXACTLY ZERO")
    print("axial force command:      DISABLED")
    print("hard axial/residual stop: +/-2.000 N / -0.538305 Nm")
    print("max TCP translation:      0.300 mm")
    print("automatic reset:          INCLUDED — exact saved Drive 1 start")
    print("automatic ratchet/thread: DISABLED")
    print(f"controller status:        {config['controller_status']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    if blockers:
        print("\nStatic blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    fatal = [
        item for item in blockers
        if not (args.read_only_preview and item.startswith("physical preparation is locked"))
    ]
    if fatal:
        raise SystemExit("[SAFETY] bolt-held Cartesian-yaw launcher refused")

    prompt = (
        "\nKeep the upright bolt securely head-held and do not move the arm. Press Enter to verify the prepared start (Ctrl+C cancels): "
        if args.resume_held else
        "\nKeep the bolt removed. Press Enter to inspect the current-to-start reset path read-only (Ctrl+C cancels): "
    )
    input(prompt)
    from franky import Frame, Robot

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)

    def fk_xyz(q_values: np.ndarray) -> np.ndarray:
        return np.asarray([
            (
                np.asarray(
                    robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix,
                    dtype=np.float64,
                ) @ joint7_to_tcp
            )[:3, 3]
            for q in q_values
        ])

    reset = config["automatic_reset_to_start"]
    target_q = drive.expected_start_q
    reset_delta = float(np.max(np.abs(target_q - current_q)))
    segment_count = max(1, int(math.ceil(
        reset_delta / float(reset["maximum_joint_delta_per_segment_rad"])
    )))
    segment_targets = np.linspace(current_q, target_q, num=segment_count + 1)[1:]
    reset_q = np.linspace(current_q, target_q, num=201)
    reset_xyz = fk_xyz(reset_q)
    reset_fk_error = float(np.linalg.norm(reset_xyz[-1] - start_pose[:3, 3]))
    q7_min = float(ratchet["summary"]["q7_prewind_rad"])
    q7_max = float(ratchet["summary"]["q7_forward_rad"])
    runtime: list[str] = []
    margin = float(reset["minimum_mechanical_margin_rad"])
    bounds_min = np.asarray(reset["workspace_bounds_m"]["min_xyz"])
    bounds_max = np.asarray(reset["workspace_bounds_m"]["max_xyz"])
    if segment_count > int(reset["maximum_segments"]):
        runtime.append("automatic reset needs too many stationary segments")
    if np.any(reset_q < JOINT_MIN + margin) or np.any(reset_q > JOINT_MAX - margin):
        runtime.append("automatic reset violates guarded mechanical joint ranges")
    if np.any(reset_q[:, 6] < q7_min) or np.any(reset_q[:, 6] > q7_max):
        runtime.append("automatic reset leaves the cable-safe q7 range")
    if np.any(reset_xyz < bounds_min) or np.any(reset_xyz > bounds_max):
        runtime.append("automatic reset leaves its Cartesian workspace")
    if reset_fk_error > float(reset["maximum_fk_target_error_m"]):
        runtime.append("automatic reset endpoint does not match saved Drive 1 start")
    if args.resume_held:
        live_error = float(np.max(np.abs(current_q[:6] - target_q[:6])))
        if live_error > float(reset["terminal_joint_tolerance_rad"]):
            runtime.append("prepared q1-q6 no longer match the saved Drive 1 geometry")
        dynamic_q7 = np.asarray(dynamic["knots"], dtype=np.float64)[:, 0]
        conservative_q7_target = current_q[6] + math.radians(float(config["test_angle_deg"]))
        if current_q[6] < dynamic_q7[0] - 0.02 or current_q[6] > dynamic_q7[-1]:
            runtime.append("live q7 is outside the calibrated dynamic torque model")
        if conservative_q7_target > dynamic_q7[-1]:
            runtime.append("insufficient calibrated dynamic-model reserve for the requested yaw")
        if conservative_q7_target > q7_max:
            runtime.append("insufficient cable-safe reserve for the requested yaw")
    print(f"live q:                   {np.array2string(current_q, precision=5)}")
    print(f"saved Drive 1 start q:    {np.array2string(target_q, precision=5)}")
    print(f"automatic reset max dq:  {reset_delta:.6f} rad")
    print(f"automatic reset segments:{segment_count}")
    print(f"reset xyz min/max:        {np.array2string(reset_xyz.min(0), precision=5)} / {np.array2string(reset_xyz.max(0), precision=5)}")
    print(f"reset target FK error:    {reset_fk_error:.6f} m")
    if runtime:
        print("\nExecution blockers:")
        for blocker in runtime:
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if runtime:
        raise SystemExit("[SAFETY] physical preparation refused")

    from franky import Gripper, JointMotion, JointState

    gripper = Gripper(config["robot_ip"])
    grasp_accepted = True
    if args.resume_held:
        print("[resume] preserving the prepared arm pose and closed Hand")
    else:
        input("Confirm there is NO BOLT and the full reset path is clear. Press Enter for the slow segmented reset: ")
        for index, target in enumerate(segment_targets, start=1):
            robot.move(JointMotion(
                JointState(target),
                relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
            ))
            error = float(np.max(np.abs(np.asarray(robot.state.q) - target)))
            print(f"[auto-reset {index}/{segment_count}] terminal error={error:.6f} rad")
            if error > float(reset["terminal_joint_tolerance_rad"]):
                raise SystemExit("[SAFETY] reset terminal tolerance failed")
        gripper.open(float(config["grasp_speed_m_s"]))
        input("Place the upright, UNENGAGED bolt in the guide between the open fingers. Press Enter to grasp its head: ")
        grasp_accepted = bool(gripper.grasp(
            float(config["grasp_width_m"]),
            float(config["grasp_speed_m_s"]),
            float(config["grasp_force_N"]),
        ))
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    print(f"[head grasp] accepted/width/grasped: {grasp_accepted} / {width:.6f} m / {grasped}")
    if not grasp_accepted or not grasped or not accepted_head_width(config, head, width):
        raise SystemExit("[SAFETY] head-grasp validation failed; Hand remains closed")
    visual = input("Confirm the bolt is upright, self-supported, head-held, NOT thread-engaged, and unloaded. Press Enter to continue; type anything to abort: ")
    if visual:
        print("[abort] Hand remains closed; no controller session was started")
        return
    del gripper
    del robot
    gc.collect()
    time.sleep(0.5)

    output = Path("-")
    direct_mode = "read-only-preview"
    confirmation = "PREVIEW_ONLY"
    output_root = project_path(config["output_dir"])
    if args.execute:
        output_root.mkdir(parents=True, exist_ok=True)
        existing = list(output_root.glob("*_held_cartesian_yaw.csv"))
        if len(existing) >= int(config["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] authorized held-yaw trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = output_root / f"{stamp}_held_cartesian_yaw.csv"
        direct_mode = "execute"
        confirmation = "HELD_CARTESIAN_YAW_AUTHORIZED"
    with tempfile.TemporaryDirectory(prefix="fr3_held_yaw_") as temp:
        model_path = Path(temp) / "model.csv"
        write_dynamic_model(model_path, dynamic)
        completed = subprocess.run([
            str(binary), direct_mode, config["robot_ip"], str(output),
            confirmation, str(model_path),
        ], check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if args.prepare_controller_preview:
        print("Prepared controller preview passed. No yaw control session was started; the Hand remains closed.")


if __name__ == "__main__":
    main()
