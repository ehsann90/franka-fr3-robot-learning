#!/usr/bin/env python3
"""Preview or execute one guarded reverse q7 reset with an empty Hand and no bolt."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.wrist_commissioning import build_guarded_reverse_q7_target  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_wrist_reset_commissioning_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


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
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        limits = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load reverse wrist commissioning configuration: {error}") from error

    print("=== EMPTY-HAND BOLT WRIST REVERSE-RESET COMMISSIONING ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"required setup:           {config['required_setup']}")
    print(f"test rotation:            -{config['reverse_test_angle_deg']:.1f} deg q7")
    print(f"cable-safe q7 interval:   {limits['summary']['q7_prewind_rad']:+.6f} .. {limits['summary']['q7_forward_rad']:+.6f} rad")
    print(f"relative dynamics factor: {config['relative_dynamics_factor']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    print("Hand connection/command:  DISABLED")
    print("contact/bolt/threading:    DISABLED; BOLT MUST BE REMOVED")
    if args.execute and config.get("execution_enabled") is not True:
        raise SystemExit("[SAFETY] reverse execution is not commissioned; no robot connection attempted")

    input("Confirm there is NO BOLT and press Enter to connect read-only (Ctrl+C cancels): ")
    from franky import Frame, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    try:
        target_q, blockers = build_guarded_reverse_q7_target(
            current_q,
            config["expected_start_q"],
            reverse_angle_rad=float(config["reverse_test_angle_rad"]),
            cable_q7_min_rad=float(limits["summary"]["q7_prewind_rad"]),
            cable_q7_max_rad=float(limits["summary"]["q7_forward_rad"]),
            minimum_backward_reserve_after_test_rad=float(
                config["minimum_backward_reserve_after_test_rad"]
            ),
            start_q7_tolerance_rad=float(config["start_q7_tolerance_rad"]),
            other_joint_start_tolerance_rad=float(config["other_joint_start_tolerance_rad"]),
            maximum_joint_delta_rad=float(config["maximum_joint_delta_rad"]),
            mechanical_min=JOINT_MIN,
            mechanical_max=JOINT_MAX,
            mechanical_margin_rad=float(config["joint_mechanical_margin_rad"]),
        )
    except (TypeError, ValueError) as error:
        raise SystemExit(f"invalid reverse commissioning configuration: {error}") from error

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(current_q, target_q, num=101)
    path_xyz = np.asarray([
        (np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix, dtype=np.float64) @ joint7_to_tcp)[:3, 3]
        for q in q_path
    ])
    bounds = config["workspace_bounds_m"]
    minimum = np.asarray(bounds["min_xyz"], dtype=np.float64)
    maximum = np.asarray(bounds["max_xyz"], dtype=np.float64)
    if np.any(path_xyz < minimum) or np.any(path_xyz > maximum):
        blockers.append("predicted reverse path leaves configured local workspace")
    translation = float(np.max(np.linalg.norm(path_xyz - path_xyz[0], axis=1)))
    if translation > float(config["maximum_predicted_translation_m"]):
        blockers.append("predicted TCP translation exceeds configured maximum")

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"target q:                 {np.array2string(target_q, precision=5)}")
    print(f"q7 start/target:          {current_q[6]:+.6f} / {target_q[6]:+.6f} rad")
    print(f"predicted path xyz min:   {np.array2string(path_xyz.min(axis=0), precision=5)}")
    print(f"predicted path xyz max:   {np.array2string(path_xyz.max(axis=0), precision=5)}")
    print(f"maximum TCP translation: {translation * 1000.0:.3f} mm")
    if blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if blockers:
        raise SystemExit("[SAFETY] empty-Hand reverse reset refused")

    confirmation = input(
        "Type RESET_EMPTY_WRIST_MINUS_90_DEG to execute the no-bolt reverse test; "
        "anything else aborts: "
    )
    if confirmation != "RESET_EMPTY_WRIST_MINUS_90_DEG":
        print("[commissioning] aborted without motion")
        return
    robot.relative_dynamics_factor = float(config["relative_dynamics_factor"])

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
    final_q = np.asarray(robot.state.q, dtype=np.float64)
    print(f"[DONE] final q7:          {final_q[6]:+.6f} rad")
    print(f"[DONE] q7 target error:   {abs(final_q[6] - target_q[6]):.6f} rad")
    print("[DONE] empty-Hand -90-degree q7 reset complete; no Hand command was issued")


if __name__ == "__main__":
    main()
