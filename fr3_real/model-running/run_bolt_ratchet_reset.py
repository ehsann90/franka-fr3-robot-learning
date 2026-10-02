#!/usr/bin/env python3
"""Preview or execute one guarded open/reset/regrasp bolt ratchet cycle."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_ratchet_reset_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def accepted_head_width(config: dict, calibration: dict, width: float) -> bool:
    margin = float(config["gripper_width_comparison_margin_m"])
    primary_min = float(calibration["minimum_width_m"]) - margin
    primary_max = float(calibration["maximum_width_m"]) + margin
    alternate_min, alternate_max = map(float, config["accepted_alternate_head_width_m"])
    return primary_min <= width <= primary_max or alternate_min <= width <= alternate_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--read-only-preview", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        ratchet = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
        dynamic = json.loads(project_path(config["dynamic_model_calibration"]).read_text())
        head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load ratchet-reset configuration: {error}") from error

    mode_name = "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE"
    bolt_state = str(config.get("bolt_state_label", "engaged"))
    automatic = bool(config.get("automatic_continuation_after_initial_confirmation", False))
    print("=== GUARDED BOLT EMPTY-WRIST RESET ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"required start:           {config['required_start']}")
    print("sequence:                 stationary open -> empty q7 reverse -> stationary head regrasp")
    print(f"target q7:                {float(config['target_q7_rad']):+.6f} rad")
    print(f"relative dynamics factor: {float(config['relative_dynamics_factor']):.3f}")
    print("axial/forward/thread drive: DISABLED")
    print(f"automatic continuation:   {automatic}")
    print("collision/reflex settings:UNCHANGED")
    print(f"controller status:        {config['controller_status']}")
    print(f"execution enabled:        {config['execution_enabled']}")
    if args.execute and config.get("execution_enabled") is not True:
        raise SystemExit("[SAFETY] ratchet-reset execution remains disabled pending read-only preview")

    initial_prompt = (
        f"Keep the {bolt_state} bolt stationary and securely head-held. "
        + (
            "Press Enter to connect, validate, and automatically run open/reset/regrasp if preflight passes (Ctrl+C cancels): "
            if automatic else
            "Press Enter to connect read-only (Ctrl+C cancels): "
        )
    )
    input(initial_prompt)
    from franky import Frame, Gripper, JointMotion, JointState, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    current_q = np.asarray(state.q, dtype=np.float64)
    expected_q = np.asarray(config["expected_start_q"], dtype=np.float64)
    target_q = current_q.copy()
    target_q[6] = float(config["target_q7_rad"])
    delta = target_q - current_q
    blockers: list[str] = []

    cable_min = float(ratchet["summary"]["q7_prewind_rad"])
    cable_max = float(ratchet["summary"]["q7_forward_rad"])
    knots = np.asarray(dynamic["knots"], dtype=np.float64)
    dynamic_min = float(knots[0, 0])
    dynamic_max = float(knots[-1, 0])
    mechanical_min = JOINT_MIN + float(config["joint_mechanical_margin_rad"])
    mechanical_max = JOINT_MAX - float(config["joint_mechanical_margin_rad"])

    if abs(current_q[6] - expected_q[6]) > float(config["start_q7_tolerance_rad"]):
        blockers.append("live q7 is not near the configured reset start")
    if float(np.max(np.abs(current_q[:6] - expected_q[:6]))) > float(config["other_joint_start_tolerance_rad"]):
        blockers.append("live q1-q6 are not near the configured reset start")
    if not cable_min <= current_q[6] <= cable_max:
        blockers.append("live q7 is outside the calibrated cable-safe interval")
    if not cable_min <= target_q[6] <= cable_max:
        blockers.append("ratchet target leaves the calibrated cable-safe interval")
    model_margin = float(config["dynamic_model_endpoint_margin_rad"])
    if not dynamic_min + model_margin <= target_q[6] <= dynamic_max - model_margin:
        blockers.append("ratchet target is outside the guarded dynamic-model interior")
    if delta[6] >= 0.0:
        blockers.append("ratchet target is not a reverse q7 motion")
    if abs(delta[6]) > float(config["maximum_joint_delta_rad"]):
        blockers.append("ratchet reset exceeds its maximum joint delta")
    if np.any(current_q < mechanical_min) or np.any(current_q > mechanical_max):
        blockers.append("live joints violate guarded mechanical ranges")
    if np.any(target_q < mechanical_min) or np.any(target_q > mechanical_max):
        blockers.append("ratchet target violates guarded mechanical ranges")

    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(current_q, target_q, num=151)
    path_xyz = np.asarray([
        (
            np.asarray(
                robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix,
                dtype=np.float64,
            )
            @ joint7_to_tcp
        )[:3, 3]
        for q in q_path
    ])
    bounds = config["workspace_bounds_m"]
    xyz_min = np.asarray(bounds["min_xyz"], dtype=np.float64)
    xyz_max = np.asarray(bounds["max_xyz"], dtype=np.float64)
    if np.any(path_xyz < xyz_min) or np.any(path_xyz > xyz_max):
        blockers.append("predicted empty-wrist path leaves its guarded workspace")
    translation = float(np.max(np.linalg.norm(path_xyz - path_xyz[0], axis=1)))
    if translation > float(config["maximum_predicted_translation_m"]):
        blockers.append("predicted TCP translation exceeds the ratchet-reset limit")

    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    if not grasped or not accepted_head_width(config, head, width):
        blockers.append("Hand state is inconsistent with the securely head-held bolt")

    print(f"\nlive q:                   {np.array2string(current_q, precision=5)}")
    print(f"target q:                 {np.array2string(target_q, precision=5)}")
    print(f"q7 reset:                 {current_q[6]:+.6f} -> {target_q[6]:+.6f} rad ({np.degrees(delta[6]):+.3f} deg)")
    print(f"cable-safe interval:      {cable_min:+.6f} .. {cable_max:+.6f} rad")
    print(f"dynamic-model interval:   {dynamic_min:+.6f} .. {dynamic_max:+.6f} rad")
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    print(f"predicted path xyz min:   {np.array2string(path_xyz.min(0), precision=5)}")
    print(f"predicted path xyz max:   {np.array2string(path_xyz.max(0), precision=5)}")
    print(f"maximum TCP translation: {translation * 1000.0:.3f} mm")
    if blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No arm or Hand command was issued.")
        return
    if blockers:
        raise SystemExit("[SAFETY] ratchet reset refused")

    output_root = project_path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    if len(list(output_root.glob("*_bolt_ratchet_reset.json"))) >= int(config["maximum_physical_trial_count"]):
        raise SystemExit("[SAFETY] authorized ratchet-reset trial allowance is exhausted")

    confirmation = "" if automatic else input(
        f"Confirm the {bolt_state} bolt is upright and the OPEN-finger sweep is clear. "
        "Press Enter to run the single ratchet reset; type anything to abort: "
    )
    if confirmation:
        print("[abort] Hand remains closed; no command was issued")
        return

    def stop(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop()")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    try:
        print("[ratchet] opening Hand while the arm is stationary")
        gripper.open(float(config["open_speed_m_s"]))
        open_width = float(gripper.state.width)
        if open_width < float(config["minimum_open_width_m"]):
            raise RuntimeError("Hand did not reach the guarded open width")
        print(f"[ratchet] Hand open width: {open_width:.6f} m")
        visual = "" if automatic else input(
            f"Confirm the {bolt_state} bolt stayed fixed/upright and the open fingers are clear. "
            "Press Enter for the empty-wrist reset; type anything to abort: "
        )
        if visual:
            print("[abort] Hand remains open; no arm motion was issued")
            return
        robot.move(
            JointMotion(
                JointState(target_q),
                relative_dynamics_factor=float(config["relative_dynamics_factor"]),
            )
        )
        final_q = np.asarray(robot.state.q, dtype=np.float64)
        q7_error = abs(float(final_q[6] - target_q[6]))
        other_drift = float(np.max(np.abs(final_q[:6] - target_q[:6])))
        print(f"[ratchet] final q7/error:  {final_q[6]:+.6f} / {q7_error:.6f} rad")
        print(f"[ratchet] q1-q6 drift:    {other_drift:.6f} rad")
        if q7_error > float(config["terminal_joint_tolerance_rad"]):
            raise RuntimeError("empty-wrist reset missed its q7 terminal tolerance")
        if other_drift > float(config["maximum_q1_q6_drift_rad"]):
            raise RuntimeError("q1-q6 drift exceeded the ratchet-reset limit")
        print("[ratchet] regrasping the stationary bolt head")
        accepted = bool(
            gripper.grasp(
                float(config["grasp_width_m"]),
                float(config["grasp_speed_m_s"]),
                float(config["grasp_force_N"]),
            )
        )
        held_width = float(gripper.state.width)
        held = bool(gripper.is_grasped)
        valid = accepted and held and accepted_head_width(config, head, held_width)
        print(f"[regrasp] accepted/width/grasped: {accepted} / {held_width:.6f} m / {held}")
        if not valid:
            raise RuntimeError("head regrasp validation failed; inspect physically before continuing")
    except BaseException:
        robot.stop()
        raise

    visual = "" if automatic else input(
        f"Visually confirm the {bolt_state} bolt is upright, securely head-held, and was not pulled or turned during reset. "
        "Press Enter to accept; type anything to reject while keeping it held: "
    )
    accepted_by_operator = not bool(visual)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_by_operator": accepted_by_operator,
        "start_q": current_q.tolist(),
        "target_q": target_q.tolist(),
        "final_q": final_q.tolist(),
        "q7_rotation_rad": float(final_q[6] - current_q[6]),
        "q7_target_error_rad": q7_error,
        "maximum_q1_q6_drift_rad": other_drift,
        "predicted_maximum_tcp_translation_m": translation,
        "open_width_m": open_width,
        "regrasp_width_m": held_width,
        "regrasp_is_grasped": held,
        "axial_or_forward_thread_command_issued": False,
        "collision_reflex_settings_changed": False,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = output_root / f"{stamp}_bolt_ratchet_reset.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[saved] {output}")
    if not accepted_by_operator:
        raise SystemExit("[REJECTED] Hand remains closed; do not continue to another drive")
    print(f"[SUCCESS] reset complete; {bolt_state} bolt remains head-held at the forward-drive start")
    print("No axial, tightening, seated-classification, release, camera, or collision-setting command was issued.")


if __name__ == "__main__":
    main()
