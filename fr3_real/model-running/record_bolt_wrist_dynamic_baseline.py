#!/usr/bin/env python3
"""Preview or record an empty-Hand +90/-90 q7 dynamic wrench baseline."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.wrist_commissioning import (  # noqa: E402
    build_guarded_q7_target,
    build_guarded_reverse_q7_target,
)


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_wrist_dynamic_baseline_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def path_xyz(robot, state, start_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
    from franky import Frame

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    return np.asarray([
        (np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix, dtype=np.float64) @ joint7_to_tcp)[:3, 3]
        for q in np.linspace(start_q, target_q, num=101)
    ])


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
        raise SystemExit(f"cannot load dynamic baseline configuration: {error}") from error

    print("=== EMPTY-HAND BOLT WRIST DYNAMIC BASELINE ===")
    print(f"mode:                     {'READ_ONLY_PREVIEW' if args.read_only_preview else 'EXECUTE_RECORD'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"required setup:           {config['required_setup']}")
    print(f"motion:                   +{config['forward_angle_deg']:.1f} deg then -{config['reverse_angle_deg']:.1f} deg q7")
    print(f"relative dynamics factor: {config['relative_dynamics_factor']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    print("signals:                  q, dq, K wrench, external joint torque, EE XYZ at control rate")
    print("Hand connection/command:  DISABLED")
    print("contact/bolt/threading:    DISABLED; BOLT MUST BE REMOVED")
    print("threshold derivation:      DISABLED")
    if args.execute and config.get("execution_enabled") is not True:
        raise SystemExit("[SAFETY] dynamic baseline execution is not commissioned; no robot connection attempted")

    input("Confirm there is NO BOLT and press Enter to connect read-only (Ctrl+C cancels): ")
    from franky import JointMotion, JointState, Robot

    robot = Robot(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    expected_start = np.asarray(config["expected_start_q"], dtype=np.float64)
    prewind_q = np.asarray(limits["prewind_snapshot"]["q"], dtype=np.float64)
    blockers: list[str] = []
    if abs(current_q[6] - expected_start[6]) > float(config["start_q7_tolerance_rad"]):
        blockers.append("live q7 is not at the passed reverse-reset endpoint")
    if np.max(np.abs(current_q[:6] - expected_start[:6])) > float(config["other_joint_start_tolerance_rad"]):
        blockers.append("live joints 1-6 are not near the passed reverse-reset geometry")
    forward_q, forward_blockers = build_guarded_q7_target(
        current_q,
        prewind_q,
        forward_angle_rad=float(config["forward_angle_rad"]),
        cable_q7_min_rad=float(limits["summary"]["q7_prewind_rad"]),
        cable_q7_max_rad=float(limits["summary"]["q7_forward_rad"]),
        minimum_forward_reserve_after_test_rad=float(config["minimum_forward_reserve_rad"]),
        other_joint_start_tolerance_rad=0.15,
        maximum_joint_delta_rad=float(config["maximum_joint_delta_rad"]),
        mechanical_min=JOINT_MIN,
        mechanical_max=JOINT_MAX,
        mechanical_margin_rad=float(config["joint_mechanical_margin_rad"]),
    )
    blockers.extend(forward_blockers)
    reverse_q, reverse_blockers = build_guarded_reverse_q7_target(
        forward_q,
        forward_q,
        reverse_angle_rad=float(config["reverse_angle_rad"]),
        cable_q7_min_rad=float(limits["summary"]["q7_prewind_rad"]),
        cable_q7_max_rad=float(limits["summary"]["q7_forward_rad"]),
        minimum_backward_reserve_after_test_rad=float(config["minimum_backward_reserve_rad"]),
        start_q7_tolerance_rad=0.001,
        other_joint_start_tolerance_rad=0.001,
        maximum_joint_delta_rad=float(config["maximum_joint_delta_rad"]),
        mechanical_min=JOINT_MIN,
        mechanical_max=JOINT_MAX,
        mechanical_margin_rad=float(config["joint_mechanical_margin_rad"]),
    )
    blockers.extend(reverse_blockers)
    forward_xyz = path_xyz(robot, state, current_q, forward_q)
    reverse_xyz = path_xyz(robot, state, forward_q, reverse_q)
    all_xyz = np.vstack((forward_xyz, reverse_xyz))
    bounds = config["workspace_bounds_m"]
    minimum = np.asarray(bounds["min_xyz"], dtype=np.float64)
    maximum = np.asarray(bounds["max_xyz"], dtype=np.float64)
    if np.any(all_xyz < minimum) or np.any(all_xyz > maximum):
        blockers.append("predicted cycle leaves configured local workspace")
    translation = float(np.max(np.linalg.norm(all_xyz - all_xyz[0], axis=1)))
    if translation > float(config["maximum_predicted_translation_m"]):
        blockers.append("predicted TCP translation exceeds configured maximum")

    print(f"\nlive q7:                  {current_q[6]:+.6f} rad")
    print(f"forward target q7:        {forward_q[6]:+.6f} rad")
    print(f"reverse target q7:        {reverse_q[6]:+.6f} rad")
    print(f"predicted cycle xyz min:  {np.array2string(all_xyz.min(axis=0), precision=5)}")
    print(f"predicted cycle xyz max:  {np.array2string(all_xyz.max(axis=0), precision=5)}")
    print(f"maximum TCP translation: {translation * 1000.0:.3f} mm")
    if blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if blockers:
        raise SystemExit("[SAFETY] dynamic baseline execution refused")

    confirmation = input(
        "Type RECORD_EMPTY_WRIST_BASELINE to run the commissioned +90/-90 no-bolt cycle; "
        "anything else aborts: "
    )
    if confirmation != "RECORD_EMPTY_WRIST_BASELINE":
        print("[baseline] aborted without motion")
        return

    stationary = []
    deadline = time.perf_counter() + float(config["stationary_baseline_duration_s"])
    period = 1.0 / float(config["stationary_sample_hz"])
    while time.perf_counter() < deadline:
        sample = robot.state
        stationary.append([
            float(sample.time.to_sec()),
            *np.asarray(sample.q, dtype=np.float64),
            *np.asarray(sample.dq, dtype=np.float64),
            *np.asarray(sample.K_F_ext_hat_K, dtype=np.float64),
            *np.asarray(sample.tau_ext_hat_filtered, dtype=np.float64),
        ])
        time.sleep(period)

    capacity = int(config["callback_capacity_per_phase"])
    trace = np.full((capacity * 2, 32), np.nan, dtype=np.float64)
    trace_count = 0

    def make_callback(phase: int):
        def callback(callback_state, *unused) -> None:
            nonlocal trace_count
            del unused
            if trace_count >= len(trace):
                return
            trace[trace_count] = [
                float(callback_state.time.to_sec()),
                float(phase),
                *np.asarray(callback_state.q, dtype=np.float64),
                *np.asarray(callback_state.dq, dtype=np.float64),
                *np.asarray(callback_state.K_F_ext_hat_K, dtype=np.float64),
                *np.asarray(callback_state.tau_ext_hat_filtered, dtype=np.float64),
                *np.asarray(callback_state.O_T_EE.matrix, dtype=np.float64)[:3, 3],
            ]
            trace_count += 1
        return callback

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop()")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    robot.relative_dynamics_factor = float(config["relative_dynamics_factor"])
    try:
        forward_motion = JointMotion(JointState(forward_q))
        forward_motion.register_callback(make_callback(1))
        robot.move(forward_motion)
        reverse_motion = JointMotion(JointState(reverse_q))
        reverse_motion.register_callback(make_callback(2))
        robot.move(reverse_motion)
    except BaseException:
        robot.stop()
        raise

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = project_path(config["output_dir"]) / f"{stamp}_empty_wrist_dynamic_baseline"
    output.mkdir(parents=True, exist_ok=False)
    columns = [
        "robot_time_s", "phase", *[f"q{i + 1}" for i in range(7)],
        *[f"dq{i + 1}" for i in range(7)],
        "K_Fx", "K_Fy", "K_Fz", "K_Tx", "K_Ty", "K_Tz",
        *[f"tau_ext_{i + 1}" for i in range(7)], "x", "y", "z",
    ]
    np.savez_compressed(
        output / "dynamic_baseline.npz",
        stationary=np.asarray(stationary, dtype=np.float64),
        trace=trace[:trace_count],
        columns=np.asarray(columns),
    )
    phase1 = trace[:trace_count][trace[:trace_count, 1] == 1]
    phase2 = trace[:trace_count][trace[:trace_count, 1] == 2]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": trace_count,
        "stationary_sample_count": len(stationary),
        "forward_sample_count": len(phase1),
        "reverse_sample_count": len(phase2),
        "forward_K_Fz_range_N": [float(np.min(phase1[:, 18])), float(np.max(phase1[:, 18]))],
        "forward_K_Tz_range_Nm": [float(np.min(phase1[:, 21])), float(np.max(phase1[:, 21]))],
        "reverse_K_Fz_range_N": [float(np.min(phase2[:, 18])), float(np.max(phase2[:, 18]))],
        "reverse_K_Tz_range_Nm": [float(np.min(phase2[:, 21])), float(np.max(phase2[:, 21]))],
        "automatic_thresholds_derived": False,
        "bolt_present": False,
        "Hand_commands_enabled": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[saved] control samples:  {trace_count}")
    print(f"[saved] forward K_Tz:     {result['forward_K_Tz_range_Nm']}")
    print(f"[saved] reverse K_Tz:     {result['reverse_K_Tz_range_Nm']}")
    print(f"[saved] {output}")
    print("No bolt, Hand, contact, threading, or automatic threshold command was used.")


if __name__ == "__main__":
    main()
