#!/usr/bin/env python3
"""Preview, prepare, or run the compliant-axis replay of Pilot Thread Drive 1."""

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

from expert_demo.hybrid_thread_plan import build_hybrid_thread_plan  # noqa: E402
from expert_demo.piloted_thread_replay import build_piloted_thread_drive  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_hybrid_thread_replay_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def write_plan(path: Path, plan) -> None:
    matrix = np.column_stack(
        [
            plan.time_s,
            plan.local_lateral_m,
            plan.relative_quaternion_xyzw,
            plan.source_q7_rad,
        ]
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header="time_s,lx,ly,lz,qx,qy,qz,qw,source_q7_rad",
        comments="",
    )


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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument(
        "--prepare-controller-preview",
        action="store_true",
        help="physically reset/grasp, then run the direct controller read-only preview",
    )
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume-held",
        action="store_true",
        help="execute from the current prepared Drive 1 start with the bolt already securely head-held",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.resume_held and not (args.prepare_controller_preview or args.execute):
        raise SystemExit(
            "--resume-held is valid only with --prepare-controller-preview or --execute"
        )

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    source = project_path(config["source_episode"])
    dynamic_path = project_path(config["dynamic_model_calibration"])
    dynamic = json.loads(dynamic_path.read_text())
    ratchet = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
    head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    direct = config["direct_controller"]
    binary = project_path(direct["binary"])
    drive = build_piloted_thread_drive(
        source,
        cycle=int(config["source_cycle"]),
        command_rate_hz=float(config["command_rate_hz"]),
        maximum_q7_rad=float(config["maximum_source_q7_rad"]),
    )
    plan = build_hybrid_thread_plan(
        drive,
        playback_time_scale=float(config["playback_time_scale"]),
        trim_stationary_prefix=bool(config["trim_stationary_source_prefix"]),
        stationary_lead_in_s=float(config["stationary_target_lead_in_s"]),
        replay_lateral_translation=bool(config["replay_lateral_translation"]),
        replay_yaw_only=bool(config["replay_yaw_only"]),
    )
    dynamic_knots = np.asarray(dynamic["knots"], dtype=np.float64)
    start_pose = drive.pose_targets[0]
    desired_xyz = (
        start_pose[:3, 3]
        + (start_pose[:3, :3] @ plan.local_lateral_m.T).T
    )
    q7_min = float(ratchet["summary"]["q7_prewind_rad"])
    q7_max = float(ratchet["summary"]["q7_forward_rad"])
    local_min = np.asarray(config["local_workspace_bounds_m"]["min_xyz"])
    local_max = np.asarray(config["local_workspace_bounds_m"]["max_xyz"])

    blockers: list[str] = []
    if not binary.is_file():
        blockers.append("hybrid direct-controller binary is missing")
    if np.max(np.abs(plan.local_lateral_m[:, 2])) > 1e-12:
        blockers.append("generated plan contains forbidden tool-axis position replay")
    if np.any(desired_xyz < local_min) or np.any(desired_xyz > local_max):
        blockers.append("generated compliant-axis path leaves the local workspace")
    if np.any(plan.source_q7_rad < q7_min) or np.any(plan.source_q7_rad > q7_max):
        blockers.append("source-conditioned orientation plan leaves the cable-safe q7 range")
    if float(config["approved_axial_command_ceiling_N"]) != 1.5:
        blockers.append("axial command ceiling differs from the user-approved value")
    if float(config["approved_axial_command_slew_N_s"]) != 0.5:
        blockers.append("axial slew differs from the user-approved value")
    if float(config["approved_hard_absolute_axial_delta_N"]) != 2.0:
        blockers.append("hard axial cutoff differs from the user-approved value")
    if abs(float(config["approved_hard_negative_residual_Tz_Nm"]) + 0.5383051567216975) > 1e-12:
        blockers.append("residual-torque cutoff differs from the user-approved value")
    if binary.is_file():
        linked = subprocess.run(
            ["ldd", str(binary)], capture_output=True, text=True, check=True
        ).stdout
        if direct["linked_libfranka"] not in linked:
            blockers.append("binary is not linked to the verified repository libfranka 0.21 path")
        if "libfranka-6ee8c992.so.0.18.0" in linked:
            blockers.append("binary unexpectedly links Franky's private libfranka 0.18")
    if args.prepare_controller_preview and config.get("prepare_controller_preview_enabled") is not True:
        blockers.append("physical preview preparation remains locked pending the read-only reset preview")
    if args.execute and config.get("execution_enabled") is not True:
        blockers.append("hybrid execution remains locked pending the prepared controller preview")

    mode_name = (
        "DRY_RUN"
        if args.dry_run
        else "READ_ONLY_PREVIEW"
        if args.read_only_preview
        else "PREPARE_CONTROLLER_PREVIEW"
        if args.prepare_controller_preview
        else "EXECUTE"
    )
    model_q7_max = float(dynamic_knots[-1, 0])
    first_model_boundary = int(np.argmax(plan.source_q7_rad >= model_q7_max))
    print("=== HYBRID PILOTED BOLT THREAD REPLAY ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"source:                   {source}")
    print(f"source drive/duration:    {config['source_cycle']} / {drive.source_duration_s:.3f} s")
    print(f"playback/duration:        {config['playback_time_scale']:.1f}x slower / {plan.time_s[-1]:.3f} s")
    print(f"removed stationary dwell: {plan.source_trim_start_s:.3f} s")
    print("recorded tool-axis motion:REMOVED EXACTLY (local Z = 0)")
    print("lateral translation:      HELD AT LIVE START (Pilot drift removed)")
    print("roll/pitch:               HELD AT LIVE START")
    print("tightening yaw:           REPLAYED FROM SUCCESSFUL PILOT DRIVE")
    print("target pacing:            slows above 0.10 mm; pauses at 0.25 mm lateral lag")
    print(f"planned lateral range:    {np.max(np.linalg.norm(plan.local_lateral_m, axis=1))*1000.0:.3f} mm")
    print(f"dynamic-model boundary:   q7={model_q7_max:+.6f} rad near t={plan.time_s[first_model_boundary]:.3f} s")
    print("axial regulation:         measured target +1.200360 N; P-only Kp=0.25")
    print("command ceiling/slew:     +1.500 N / 0.500 N/s")
    print("hard axial/residual stop: +/-2.000 N / -0.538305 Nm")
    print("automatic ratchet/camera: DISABLED")
    print(f"controller status:        {config['controller_status']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    if blockers:
        print("\nStatic blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return
    # The read-only preview may report a physical-preparation lock; that lock
    # is intentionally irrelevant to this no-motion reset-path inspection.
    fatal_static = [
        item
        for item in blockers
        if not (
            args.read_only_preview
            and item.startswith("physical preview preparation remains locked")
        )
    ]
    if fatal_static:
        raise SystemExit("[SAFETY] hybrid replay launcher refused")

    print(
        (
            "\nKeep the upright bolt securely head-held and do not move the arm. "
            "Press Enter to verify the prepared start (Ctrl+C cancels): "
            if args.resume_held
            else "\nKeep the bolt removed for any later automatic reset. "
            "Press Enter to connect read-only and inspect the current-to-start path (Ctrl+C cancels): "
        )
    )
    input()
    from franky import Frame, Robot

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)

    def fk_xyz(q_values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                (
                    np.asarray(
                        robot.model.pose(
                            Frame.Joint7, q, state.F_T_EE, state.EE_T_K
                        ).matrix,
                        dtype=np.float64,
                    )
                    @ joint7_to_tcp
                )[:3, 3]
                for q in q_values
            ]
        )

    reset = config["automatic_reset_to_start"]
    reset_delta = float(np.max(np.abs(drive.expected_start_q - current_q)))
    segment_limit = float(reset["maximum_joint_delta_per_segment_rad"])
    segment_count = max(1, int(math.ceil(reset_delta / segment_limit)))
    segment_targets = np.linspace(
        current_q, drive.expected_start_q, num=segment_count + 1
    )[1:]
    reset_q = np.linspace(current_q, drive.expected_start_q, num=201)
    reset_xyz = fk_xyz(reset_q)
    reset_fk_error = float(np.linalg.norm(reset_xyz[-1] - start_pose[:3, 3]))
    reset_min = np.asarray(reset["workspace_bounds_m"]["min_xyz"])
    reset_max = np.asarray(reset["workspace_bounds_m"]["max_xyz"])
    runtime_blockers: list[str] = []
    margin = float(reset["minimum_mechanical_margin_rad"])
    if segment_count > int(reset["maximum_segments"]):
        runtime_blockers.append("automatic reset needs too many stationary segments")
    if np.any(reset_q < JOINT_MIN + margin) or np.any(reset_q > JOINT_MAX - margin):
        runtime_blockers.append("automatic reset violates guarded mechanical joint ranges")
    if np.any(reset_q[:, 6] < q7_min) or np.any(reset_q[:, 6] > q7_max):
        runtime_blockers.append("automatic reset leaves the cable-safe q7 range")
    if np.any(reset_xyz < reset_min) or np.any(reset_xyz > reset_max):
        runtime_blockers.append("automatic reset leaves its Cartesian workspace")
    if reset_fk_error > float(reset["maximum_fk_target_error_m"]):
        runtime_blockers.append("automatic reset endpoint does not match the recorded start")
    print(f"live q:                   {np.array2string(current_q, precision=5)}")
    print(f"saved Drive 1 start q:    {np.array2string(drive.expected_start_q, precision=5)}")
    print(f"automatic reset max dq:  {reset_delta:.6f} rad")
    print(f"automatic reset segments:{segment_count}")
    print(f"reset xyz min/max:        {np.array2string(reset_xyz.min(0), precision=5)} / {np.array2string(reset_xyz.max(0), precision=5)}")
    print(f"reset target FK error:    {reset_fk_error:.6f} m")
    print(f"hybrid desired xyz min:   {np.array2string(desired_xyz.min(0), precision=5)}")
    print(f"hybrid desired xyz max:   {np.array2string(desired_xyz.max(0), precision=5)}")
    if args.resume_held:
        live_start_error = float(
            np.max(np.abs(current_q[:6] - drive.expected_start_q[:6]))
        )
        print(f"prepared q1-q6 error:      {live_start_error:.6f} rad")
        if live_start_error > float(reset["terminal_joint_tolerance_rad"]):
            runtime_blockers.append(
                "resume-held q1-q6 no longer match the prepared Drive 1 geometry"
            )
        if current_q[6] < dynamic_knots[0, 0] - 0.02 or current_q[6] > dynamic_knots[-1, 0]:
            runtime_blockers.append(
                "resume-held q7 is outside the calibrated dynamic torque model"
            )
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in runtime_blockers:
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] physical preparation refused")

    # From this point onward the mode explicitly authorizes physical setup.
    from franky import Gripper, JointMotion, JointState

    gripper = Gripper(config["robot_ip"])
    grasp_accepted = True
    if args.resume_held:
        print("[resume] preserving the existing arm pose and closed Hand; no reset or Hand command will be issued")
    else:
        input("Confirm there is NO BOLT and the entire reset path is clear. Press Enter for the slow segmented reset: ")
        for index, target in enumerate(segment_targets, start=1):
            robot.move(
                JointMotion(
                    JointState(target),
                    relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
                )
            )
            error = float(np.max(np.abs(np.asarray(robot.state.q) - target)))
            print(f"[auto-reset {index}/{segment_count}] terminal error={error:.6f} rad")
            if error > float(reset["terminal_joint_tolerance_rad"]):
                raise SystemExit("[SAFETY] automatic reset terminal tolerance failed")
        gripper.open(float(config["grasp_speed_m_s"]))
        input("Place the upright, unthreaded bolt in the guide between the open fingers. Press Enter to grasp its head: ")
        grasp_accepted = bool(
            gripper.grasp(
                float(config["grasp_width_m"]),
                float(config["grasp_speed_m_s"]),
                float(config["grasp_force_N"]),
            )
        )
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    print(f"[head grasp] accepted/width/grasped: {grasp_accepted} / {width:.6f} m / {grasped}")
    if not grasp_accepted or not grasped or not accepted_head_width(config, head, width):
        raise SystemExit("[SAFETY] automatic head-grasp validation failed; Hand remains closed")
    visual = input("Visually confirm the bolt is upright, securely head-held, and unloaded. Press Enter to continue; type anything to abort: ")
    if visual:
        print("[abort] Hand remains closed; no controller session was started")
        return
    del gripper
    del robot
    gc.collect()
    time.sleep(0.5)

    output_root = project_path(direct["output_dir"])
    output = Path("-")
    direct_mode = "read-only-preview"
    confirmation = "PREVIEW_ONLY"
    if args.execute:
        output_root.mkdir(parents=True, exist_ok=True)
        existing = list(output_root.glob("*_hybrid_thread_replay.csv"))
        if len(existing) >= int(config["maximum_physical_trial_count"]):
            raise SystemExit("[SAFETY] authorized hybrid physical-trial allowance is exhausted")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = output_root / f"{stamp}_hybrid_thread_replay.csv"
        direct_mode = "execute"
        confirmation = "HYBRID_EXECUTION_AUTHORIZED"
    with tempfile.TemporaryDirectory(prefix="fr3_hybrid_thread_") as temp:
        temp_root = Path(temp)
        plan_path = temp_root / "plan.csv"
        model_path = temp_root / "model.csv"
        write_plan(plan_path, plan)
        write_dynamic_model(model_path, dynamic)
        command = [
            str(binary),
            direct_mode,
            config["robot_ip"],
            str(output),
            str(plan_path),
            str(model_path),
            confirmation,
        ]
        completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    if args.prepare_controller_preview:
        print("Prepared controller preview passed. No hybrid control session was started; the Hand remains closed.")


if __name__ == "__main__":
    main()
