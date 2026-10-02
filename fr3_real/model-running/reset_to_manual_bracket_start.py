#!/usr/bin/env python3
"""Preview or execute one slow guarded reset to a manual bracket episode's
IK-candidate stage start.

Mirrors reset_to_piloted_trajectory_start.py's guarded pattern exactly:
read-only preview by default; --execute refuses to even connect while any
reset_to_start field in the config is NOT_CALIBRATED; once connected it
checks the live joint range, the recorded target's joint range, the
predicted Cartesian path against configured workspace bounds, and a
forward-kinematics discrepancy against the recorded target -- and even after
every check passes, motion only happens after the operator types the exact
confirmation phrase at the terminal. No gripper command is ever issued.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_manual_bracket_replay import STAGE_MANUAL_EVENTS, build_manual_stage_plan  # noqa: E402

DEFAULT_CONFIG = FR3_ROOT / "configs" / "bracket_manual_replay_reset_v1.json"
# Same mechanical ranges reset_to_piloted_trajectory_start.py uses.
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
NOT_CALIBRATED = "NOT_CALIBRATED"


def reset_config_blockers(reset: dict, target_q: np.ndarray, current_q: np.ndarray) -> list[str]:
    blockers: list[str] = []
    if reset.get("enabled") is not True:
        blockers.append("reset_to_start.enabled is false")
    for key in (
        "relative_dynamics_factor",
        "joint_margin_rad",
        "maximum_joint_delta_rad",
        "workspace_bounds_m",
        "target_fk_error_limit_m",
    ):
        value = reset.get(key)
        if value is None or value == NOT_CALIBRATED:
            blockers.append(f"reset_to_start.{key} is NOT_CALIBRATED")
    dynamics = reset.get("relative_dynamics_factor")
    if isinstance(dynamics, (int, float)) and not 0.0 < float(dynamics) <= 1.0:
        blockers.append("relative_dynamics_factor must be in (0, 1]")
    joint_margin = reset.get("joint_margin_rad")
    if isinstance(joint_margin, (int, float)) and not 0.0 <= float(joint_margin) < 0.5:
        blockers.append("joint_margin_rad must be in [0, 0.5)")
    maximum_delta = reset.get("maximum_joint_delta_rad")
    if isinstance(maximum_delta, (int, float)):
        if float(maximum_delta) <= 0.0:
            blockers.append("maximum_joint_delta_rad must be positive")
        actual = float(np.max(np.abs(target_q - current_q)))
        if actual > float(maximum_delta):
            blockers.append(
                f"required max joint delta {actual:.6f} rad exceeds configured "
                f"maximum {float(maximum_delta):.6f} rad"
            )
    fk_limit = reset.get("target_fk_error_limit_m")
    if isinstance(fk_limit, (int, float)) and float(fk_limit) <= 0.0:
        blockers.append("target_fk_error_limit_m must be positive")
    return blockers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_MANUAL_EVENTS), default="approach_open")
    parser.add_argument("--command-rate-hz", type=float, default=15.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    candidate_path = args.candidate.expanduser().resolve()
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        plan = build_manual_stage_plan(
            candidate_path, args.stage, command_rate_hz=args.command_rate_hz
        )
        reset = config["reset_to_start"]
        recorded_start_xyz = plan.xyz_targets[0]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"[reset] invalid input: {error}") from error

    print("=== RESET TO MANUAL BRACKET REPLAY START ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"candidate:                {candidate_path}")
    print(f"stage:                    {plan.name}")
    print(f"target q:                 {np.array2string(plan.expected_start_q, precision=5)}")
    print(f"recorded target EE xyz:   {np.array2string(recorded_start_xyz, precision=5)}")
    print(f"reset enabled:            {reset.get('enabled')}")
    print(f"relative dynamics factor: {reset.get('relative_dynamics_factor')}")
    print(f"joint margin:             {reset.get('joint_margin_rad')}")
    print(f"maximum joint delta:      {reset.get('maximum_joint_delta_rad')}")
    print(f"workspace bounds:         {reset.get('workspace_bounds_m')}")
    print("gripper connection:       DISABLED BY DESIGN")
    print("gripper commands:         DISABLED BY DESIGN")

    # Do not even connect in execute mode while static commissioning fields are absent.
    static_blockers = reset_config_blockers(reset, plan.expected_start_q, plan.expected_start_q)
    if args.execute and static_blockers:
        print("\nExecution blockers:")
        for blocker in static_blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] reset execution is not commissioned; no robot connection attempted")

    input("Press Enter to connect and inspect the live arm state (Ctrl+C cancels): ")
    from franky import Frame, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    target_q = plan.expected_start_q
    q_delta = target_q - current_q

    margin_value = reset.get("joint_margin_rad")
    margin = float(margin_value) if isinstance(margin_value, (int, float)) else 0.0
    lower = JOINT_MIN + margin
    upper = JOINT_MAX - margin
    runtime_blockers = reset_config_blockers(reset, target_q, current_q)
    if np.any(current_q < lower) or np.any(current_q > upper):
        runtime_blockers.append("live joints violate the configured guarded joint range")
    if np.any(target_q < lower) or np.any(target_q > upper):
        runtime_blockers.append("recorded target violates the configured guarded joint range")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(current_q, target_q, num=101)
    path_xyz = np.asarray(
        [
            (
                np.asarray(
                    robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix,
                    dtype=np.float64,
                )
                @ joint7_to_tcp
            )[:3, 3]
            for q in q_path
        ]
    )
    predicted_target_xyz = path_xyz[-1]
    target_fk_error = float(np.linalg.norm(predicted_target_xyz - recorded_start_xyz))

    workspace = reset.get("workspace_bounds_m")
    if isinstance(workspace, dict):
        minimum = np.asarray(workspace.get("min_xyz"), dtype=np.float64)
        maximum = np.asarray(workspace.get("max_xyz"), dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            runtime_blockers.append("configured reset workspace bounds are invalid")
        elif np.any(path_xyz < minimum) or np.any(path_xyz > maximum):
            runtime_blockers.append(
                "predicted reset path leaves configured Cartesian workspace"
            )
    fk_limit = reset.get("target_fk_error_limit_m")
    if isinstance(fk_limit, (int, float)) and target_fk_error > float(fk_limit):
        runtime_blockers.append(
            f"predicted target FK error {target_fk_error:.6f} m exceeds configured "
            f"limit {float(fk_limit):.6f} m"
        )

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"required q delta:         {np.array2string(q_delta, precision=5)}")
    print(f"max required |q delta|:  {float(np.max(np.abs(q_delta))):.6f} rad")
    print(f"live EE xyz:              {np.array2string(measured_tcp[:3, 3], precision=5)}")
    print(f"predicted path xyz min:   {np.array2string(path_xyz.min(axis=0), precision=5)}")
    print(f"predicted path xyz max:   {np.array2string(path_xyz.max(axis=0), precision=5)}")
    print(f"predicted target EE xyz: {np.array2string(predicted_target_xyz, precision=5)}")
    print(f"target FK discrepancy:   {target_fk_error:.6f} m")

    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or gripper command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] reset execution refused")

    confirmation = input(
        "Type MOVE_TO_RECORDED_START to execute one slow arm-only reset; anything else aborts: "
    )
    if confirmation != "MOVE_TO_RECORDED_START":
        print("[reset] aborted without motion")
        return

    robot.relative_dynamics_factor = float(reset["relative_dynamics_factor"])

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop()")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    try:
        robot.move(JointMotion(JointState(target_q)))
    except BaseException:
        robot.stop()
        raise
    final_state = robot.state
    final_q = np.asarray(final_state.q, dtype=np.float64)
    final_error = float(np.max(np.abs(final_q - target_q)))
    print(f"[DONE] final q:           {np.array2string(final_q, precision=5)}")
    print(f"[DONE] max |q error|:    {final_error:.6f} rad")
    print("[DONE] no gripper command was issued")


if __name__ == "__main__":
    main()
