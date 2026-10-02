#!/usr/bin/env python3
"""Move from a seated bolt to the exact guarded THREAD_READY setup.

Usage from ``fr3_real/``::

    python model-running/move_seated_bolt_to_thread_ready.py --read-only-preview
    python model-running/move_seated_bolt_to_thread_ready.py --execute

The bolt must already be upright and seated in the bracket. The tool moves the
empty Hand to the commissioned regrasp pose, prewinds q7 to the engagement
start, and then performs the calibrated head grasp. It never starts threading.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from record_autonomous_bolt_engagement_demo import load_inputs, pickup_grasp_evidence  # noqa: E402
from run_bolt_dropregrasp_to_thread_ready_fast import check_head_grasp  # noqa: E402
from expert_demo.retimed_trajectory import build_retimed_trajectory  # noqa: E402

JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
WORKSPACE_MIN = np.asarray([0.57, 0.30, 0.08])
WORKSPACE_MAX = np.asarray([0.63, 0.35, 0.15])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--speed-config", type=Path, default=FR3_ROOT / "configs/full_assembly_4x_production_v1.json")
    parser.add_argument("--recording-config", type=Path, default=FR3_ROOT / "configs/autonomous_demo_collection_v1.json")
    parser.add_argument(
        "--reset-dynamics-factor",
        type=float,
        default=0.15,
        help="relative dynamics factor for the empty-Hand move to the regrasp pose (0.05-0.20)",
    )
    args = parser.parse_args()
    if not 0.05 <= args.reset_dynamics_factor <= 0.20:
        raise SystemExit("--reset-dynamics-factor must be between 0.05 and 0.20")
    speed_config = json.loads(args.speed_config.expanduser().resolve().read_text())
    collection, task, preinsert, drop, engagement, commissioning, plans = load_inputs(args.recording_config)
    del preinsert, engagement, commissioning
    stage = build_retimed_trajectory(
        plans["drop"]["reposition_regrasp_open"],
        playback_time_scale=float(
            speed_config["playback_time_scale_by_stage"]["bolt_reposition_regrasp_open"]
        ),
    )
    target_q = np.asarray(stage.expected_stop_q, dtype=np.float64)
    head_grasp = task["thread_ready_head_grasp"]

    print("=== SEATED BOLT TO THREAD_READY ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {collection['robot_ip']}")
    print(f"target regrasp q:         {np.array2string(target_q, precision=5)}")
    print(f"engagement start q7:      {float(task['engagement_start_q7_rad']):+.6f} rad")
    print("threading:                DISABLED; this script stops at THREAD_READY")
    print(f"regrasp move dynamics:    {args.reset_dynamics_factor:.2f}")
    print("Hand:                     open until calibrated head grasp")
    input("Confirm bolt is upright and seated in the bracket; press Enter to inspect (Ctrl+C cancels): ")

    from franky import Frame, Gripper, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(collection["robot_ip"])
    gripper = Gripper(collection["robot_ip"])
    current_q = np.asarray(robot.state.q, dtype=np.float64)
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    blockers = []
    if grasped or width < float(drop["gripper"]["minimum_open_width_m"]):
        blockers.append("Hand is not open; seated-bolt mode requires an empty Hand")
    if np.any(target_q < JOINT_MIN + 0.12) or np.any(target_q > JOINT_MAX - 0.12):
        blockers.append("target regrasp pose violates guarded joint limits")
    if np.max(np.abs(target_q - current_q)) > 1.65:
        blockers.append("direct move to regrasp pose exceeds the 1.65 rad reset bound")
    measured_tcp = np.asarray(robot.state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, robot.state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(current_q, target_q, num=201)
    xyz = np.asarray([
        (np.asarray(robot.model.pose(Frame.Joint7, q, robot.state.F_T_EE, robot.state.EE_T_K).matrix) @ joint7_to_tcp)[:3, 3]
        for q in q_path
    ])
    if np.any(xyz < WORKSPACE_MIN) or np.any(xyz > WORKSPACE_MAX):
        blockers.append("predicted empty-Hand path leaves the local workspace")
    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"live Hand width/grasped:  {width:.6f} m / {grasped}")
    print(f"maximum joint delta:      {np.max(np.abs(target_q - current_q)):.6f} rad")
    print(f"predicted TCP xyz min:    {np.array2string(xyz.min(axis=0), precision=5)}")
    print(f"predicted TCP xyz max:    {np.array2string(xyz.max(axis=0), precision=5)}")
    if blockers:
        print("\nBlockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] seated-bolt THREAD_READY transition refused")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if input("Press Enter to move to the regrasp pose; type anything to abort: "):
        return
    robot.relative_dynamics_factor = args.reset_dynamics_factor

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    robot.move(JointMotion(JointState(target_q), relative_dynamics_factor=args.reset_dynamics_factor))
    terminal_error = float(np.max(np.abs(np.asarray(robot.state.q) - target_q)))
    if terminal_error > 0.01:
        raise RuntimeError(f"regrasp pose terminal error too large: {terminal_error:.6f} rad")
    prewind_q = np.array(robot.state.q, dtype=np.float64, copy=True)
    prewind_q[6] = float(task["engagement_start_q7_rad"])
    print(f"[prewind] q7 {robot.state.q[6]:+.6f} -> {prewind_q[6]:+.6f} rad")
    robot.move(JointMotion(JointState(prewind_q), relative_dynamics_factor=float(task["empty_wrist_prewind_relative_dynamics_factor"])))
    accepted = gripper.grasp(float(head_grasp["target_width_m"]), float(head_grasp["speed_m_s"]), float(head_grasp["force_N"]), float(head_grasp["epsilon_inner_m"]), float(head_grasp["epsilon_outer_m"]))
    held_width = float(gripper.state.width)
    held = bool(gripper.is_grasped)
    strict, bounded = check_head_grasp(head_grasp, accepted=bool(accepted), is_grasped=held, width=held_width)
    print(f"[head grasp] accepted={accepted} grasped={held} width={held_width:.6f} strict={strict} bounded={bounded}")
    if not (strict or bounded):
        raise RuntimeError("[SAFETY] calibrated head grasp was not verified")
    input("Confirm THREAD_READY: bolt upright, securely held, unloaded. Press Enter to finish: ")
    print("[THREAD_READY] confirmed. No threading command was issued.")


if __name__ == "__main__":
    main()
