#!/usr/bin/env python3
"""Preview or execute a guarded empty-Hand reset to the dynamic-baseline pose.

Usage from ``fr3_real/``::

    python model-running/reset_to_empty_wrist_baseline.py --read-only-preview
    python model-running/reset_to_empty_wrist_baseline.py --execute

The target comes from the extended dynamic-baseline configuration. The tool
does not connect to or command the Hand; the operator must remove the bolt.
"""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
DEFAULT_CONFIG = FR3_ROOT / "configs/bolt_wrist_dynamic_baseline_extended_100deg_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text())
    if args.execute and config.get("reset_execution_enabled") is not True:
        raise SystemExit("[SAFETY] reset execution is not enabled in this reset configuration")

    print("=== EMPTY-HAND RESET TO DYNAMIC-BASELINE START ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print("Hand connection/command:  DISABLED")
    print("contact/bolt/threading:   DISABLED; BOLT MUST BE REMOVED")
    print(f"relative dynamics factor: {config['relative_dynamics_factor']}")
    input("Confirm the Hand is empty and press Enter to connect (Ctrl+C cancels): ")

    from franky import Frame, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    target_q = np.asarray(config["expected_start_q"], dtype=np.float64)
    blockers: list[str] = []
    if np.any(target_q <= JOINT_MIN + float(config["joint_mechanical_margin_rad"])):
        blockers.append("target is too close to a lower joint limit")
    if np.any(target_q >= JOINT_MAX - float(config["joint_mechanical_margin_rad"])):
        blockers.append("target is too close to an upper joint limit")
    if np.max(np.abs(target_q - current_q)) > float(config["maximum_joint_delta_rad"]):
        blockers.append("reset joint delta exceeds configured maximum")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(current_q, target_q, num=101)
    path_xyz = np.asarray([
        (np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
         @ joint7_to_tcp)[:3, 3]
        for q in q_path
    ])
    bounds = config["workspace_bounds_m"]
    minimum = np.asarray(bounds["min_xyz"], dtype=np.float64)
    maximum = np.asarray(bounds["max_xyz"], dtype=np.float64)
    if np.any(path_xyz < minimum) or np.any(path_xyz > maximum):
        blockers.append("predicted reset path leaves configured local workspace")
    translation = float(np.max(np.linalg.norm(path_xyz - path_xyz[0], axis=1)))
    if translation > float(config["maximum_predicted_translation_m"]):
        blockers.append("predicted TCP translation exceeds configured maximum")

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"target q:                 {np.array2string(target_q, precision=5)}")
    print(f"maximum joint delta:      {np.max(np.abs(target_q - current_q)):.6f} rad")
    print(f"maximum TCP translation:  {translation * 1000.0:.3f} mm")
    if blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if blockers:
        raise SystemExit("[SAFETY] empty-Hand baseline reset refused")
    if input("Press Enter to move to the empty-Hand baseline pose; type anything to abort: "):
        print("[reset] aborted without motion")
        return

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop()")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    robot.relative_dynamics_factor = float(config["relative_dynamics_factor"])
    try:
        robot.move(JointMotion(JointState(target_q)))
    except BaseException:
        robot.stop()
        raise
    final_q = np.asarray(robot.state.q, dtype=np.float64)
    print(f"[DONE] terminal q error: {np.max(np.abs(final_q - target_q)):.6f} rad")
    print("[DONE] arm is at the empty-Hand dynamic-baseline start; no Hand command was issued")


if __name__ == "__main__":
    main()
