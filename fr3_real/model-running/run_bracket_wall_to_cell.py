#!/usr/bin/env python3
"""Supervised wall-to-cell setup transfer; no dataset episode or camera capture.

Start with the bracket seated on the pins and the empty Hand clear. The arm
can begin at the wall pose or at another pose whose slow segmented reset to
the recorded wall retreat passes the configured joint and workspace checks.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_demo.retimed_trajectory import write_retimed_csv  # noqa: E402
from prepare_bracket_wall_cell_cycle import prepare, project_path, reverse_plan, retime  # noqa: E402
from prepare_manual_bracket_replay import build_manual_stage_plan  # noqa: E402
from reset_to_manual_bracket_start import JOINT_MAX, JOINT_MIN  # noqa: E402


def run_direct(binary: Path, robot_ip: str, trajectory, name: str,
               bounds: dict, telemetry: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fr3_bracket_inverse_") as temporary:
        plan_path = Path(temporary) / "plan.csv"
        write_retimed_csv(plan_path, trajectory)
        command = [
            str(binary), "execute", robot_ip, str(plan_path), str(telemetry),
            "RUN_RETIMED_ACCELERATED_STAGE", name,
            *(str(value) for value in bounds["min_xyz"]),
            *(str(value) for value in bounds["max_xyz"]),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(input="\n\n")
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.communicate()
            raise
        print(stdout, end="")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if process.returncode != 0 or "[result] stop reason:      motion_completed" not in stdout:
            raise RuntimeError(f"{name} did not finish normally")


def plan_empty_hand_reset(robot, state, current_q: np.ndarray,
                          target_q: np.ndarray, target_xyz: np.ndarray,
                          config: dict) -> tuple[np.ndarray, int, list[str]]:
    """Audit a slow joint-linear reset without issuing a command."""
    from franky import Frame  # noqa: PLC0415

    maximum_delta = float(config["reset_maximum_joint_delta_rad"])
    segment_count = max(1, math.ceil(float(np.max(np.abs(target_q - current_q))) / maximum_delta))
    measured_tcp = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, state).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    q_path = np.linspace(current_q, target_q, num=201)
    xyz_path = np.asarray([
        (np.asarray(robot.model.pose(Frame.Joint7, q, state.F_T_EE, state.EE_T_K).matrix)
         @ joint7_to_tcp)[:3, 3]
        for q in q_path
    ])
    workspace = config["reset_workspace_bounds_m"]
    lower_xyz = np.asarray(workspace["min_xyz"], dtype=float)
    upper_xyz = np.asarray(workspace["max_xyz"], dtype=float)
    blockers = []
    if not np.isfinite(q_path).all() or not np.isfinite(xyz_path).all():
        blockers.append("non-finite reset path")
    if segment_count > int(config["reset_maximum_segments"]):
        blockers.append(f"reset needs {segment_count} segments; configured maximum is {config['reset_maximum_segments']}")
    if np.any(q_path < JOINT_MIN + 0.02) or np.any(q_path > JOINT_MAX - 0.02):
        blockers.append("reset path enters a joint-limit margin")
    if np.any(xyz_path < lower_xyz) or np.any(xyz_path > upper_xyz):
        blockers.append("reset path leaves the configured workspace")
    target_fk_error = float(np.linalg.norm(xyz_path[-1] - target_xyz))
    if target_fk_error > 0.01:
        blockers.append(f"target FK differs from recorded retreat by {target_fk_error:.4f} m")
    print(f"automatic reset: {segment_count} slow segment(s)")
    print(f"reset TCP xyz min/max: {xyz_path.min(axis=0)} / {xyz_path.max(axis=0)}")
    print(f"reset target FK error: {target_fk_error:.6f} m")
    return xyz_path, segment_count, blockers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--execute-supervised-trial", action="store_true")
    parser.add_argument("--inverse-only", action="store_true", help="Stop after placing the bracket at the cell")
    parser.add_argument(
        "--setup-cycle-only", action="store_true",
        help="After the unrecorded wall-to-cell setup, return cell-to-wall without cameras or a dataset episode",
    )
    args = parser.parse_args()
    if args.inverse_only and args.setup_cycle_only:
        parser.error("--inverse-only and --setup-cycle-only are mutually exclusive")
    if args.episode_index < 1:
        raise SystemExit("episode index must be positive")
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text())
    if config.get("calibration_only") and not (args.inverse_only or args.setup_cycle_only):
        raise SystemExit("[SAFETY] calibration-only config cannot launch a placement recording")
    manifest = prepare(config_path, None)
    if manifest["blockers"]:
        raise SystemExit(f"[SAFETY] inverse offline blockers: {manifest['blockers']}")
    if not args.execute_supervised_trial:
        print(json.dumps(manifest, indent=2))
        print("Read-only offline preview complete; no robot or camera connection.")
        return
    if config.get("inverse_setup_motion", {}).get("execution_enabled") is False:
        raise SystemExit("[SAFETY] this new-cell inverse setup has not been approved for physical execution")
    if args.setup_cycle_only and any(
        not bool(stage.get("execution_enabled", False)) for stage in config["stages"].values()
    ):
        raise SystemExit("[SAFETY] cell-to-wall stages remain disabled; no motion started")

    candidate = project_path(config["candidate"])
    binary = project_path(config["binary"])
    if not binary.is_file():
        raise SystemExit("[SAFETY] direct trajectory controller is missing")
    source = {
        name: build_manual_stage_plan(candidate, name, command_rate_hz=float(config["command_rate_hz"]))
        for name in ("approach_open", "transport_place_closed", "retreat_open")
    }
    reverse_retreat = retime(reverse_plan(source["retreat_open"]), config, stage_name="wall_approach_open")
    reverse_transport = retime(reverse_plan(source["transport_place_closed"]), config, stage_name="wall_to_cell_held")
    reverse_approach = retime(reverse_plan(source["approach_open"]), config, stage_name="cell_retreat_open")
    grasp_cfg = config["gripper"]
    release_speed = float(config.get("inverse_setup_motion", {}).get(
        "cell_release_open_speed_m_s", grasp_cfg["open_speed_m_s"]
    ))
    if not 0.0 < release_speed <= float(grasp_cfg["open_speed_m_s"]):
        raise SystemExit("[SAFETY] cell release speed must be positive and no faster than the configured open speed")
    setup_cfg = config.get("inverse_setup_motion", {})
    setup_root = ROOT / "diagnostics" / "bracket_wall_to_cell_setup_trials_v1"
    prior_trials = list(setup_root.glob(f"*_setup_*_{config['bracket_slot']}"))
    trial_limit = int(setup_cfg.get("maximum_physical_trials", 0))
    if trial_limit and len(prior_trials) >= trial_limit:
        raise SystemExit(
            f"[SAFETY] {config['bracket_slot']} inverse physical-trial allowance "
            f"exhausted ({len(prior_trials)}/{trial_limit}); no motion started"
        )
    before_open_s = float(setup_cfg.get("cell_settle_before_open_s", 0.0))
    after_open_s = float(setup_cfg.get("cell_settle_after_open_s", 0.0))
    if not 0.0 <= before_open_s <= 3.0 or not 0.0 <= after_open_s <= 3.0:
        raise SystemExit("[SAFETY] cell settle holds must each be between 0 and 3 seconds")
    print("=== BRACKET WALL TO CELL: UNRECORDED SETUP MOVE ===")
    print(f"cell: {config['bracket_slot']}; no cameras or dataset episode")
    print(f"held transfer: {reverse_transport.duration_s:.2f} s; cell opening: {release_speed:.3f} m/s")
    print("Start: bracket seated on wall pins; open Hand at recorded wall grasp or retreat pose.")
    print("End: bracket at the recorded cell pose; open Hand at the source approach start.")
    print("Connecting read-only to check the automatic start reset...")

    from franky import Gripper, JointMotion, JointState, Robot  # noqa: PLC0415
    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    q = np.asarray(state.q, dtype=float)
    wall_error = float(np.max(np.abs(q - reverse_transport.expected_start_q)))
    retreat_error = float(np.max(np.abs(q - reverse_retreat.expected_start_q)))
    approach_from_retreat = retreat_error <= float(config["controller_start_tolerance_rad"])
    width = float(gripper.state.width)
    print(f"wall grasp error: {wall_error:.6f} rad; wall retreat error: {retreat_error:.6f} rad")
    print(f"Hand width: {width:.6f} m")
    blockers = []
    if bool(gripper.is_grasped):
        blockers.append("Hand reports a held object")
    if float(np.max(np.abs(np.asarray(state.dq, dtype=float)))) > 0.02:
        blockers.append("arm is moving during reset preflight")
    reset_required = min(wall_error, retreat_error) > float(config["controller_start_tolerance_rad"])
    reset_segments = 0
    if reset_required:
        _, reset_segments, reset_blockers = plan_empty_hand_reset(
            robot, state, q, reverse_retreat.expected_start_q,
            source["retreat_open"].xyz_targets[-1], config,
        )
        blockers.extend(reset_blockers)
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] automatic start reset refused; leave the arm at the wall grasp/retreat pose only if these checks cannot be met")
    input("Keep the bracket on the pins and the Hand/reset path clear. Press Enter for the automatic start and inverse transfer: ")
    if width < 0.075:
        gripper.open(float(grasp_cfg["open_speed_m_s"]))
        width = float(gripper.state.width)
        print(f"[auto-open] Hand width: {width:.6f} m")
    if width < 0.075 or bool(gripper.is_grasped):
        raise SystemExit("[SAFETY] Hand failed to reach a clear open width")
    if reset_required:
        initial_q = q.copy()
        target_q = reverse_retreat.expected_start_q
        for segment_index in range(1, reset_segments + 1):
            segment_q = initial_q + (segment_index / reset_segments) * (target_q - initial_q)
            robot.move(JointMotion(
                JointState(segment_q),
                relative_dynamics_factor=float(config["reset_relative_dynamics_factor"]),
            ))
            error = float(np.max(np.abs(np.asarray(robot.state.q) - segment_q)))
            print(f"[auto-reset {segment_index}/{reset_segments}] terminal error: {error:.6f} rad")
            if error > float(config["reset_terminal_tolerance_rad"]):
                raise SystemExit("[SAFETY] automatic reset segment missed its target")
        approach_from_retreat = True
    elif wall_error <= float(config["controller_start_tolerance_rad"]):
        approach_from_retreat = False

    stamp = time.strftime("%Y%m%d_%H%M%S")
    trial = setup_root / (
        f"{stamp}_setup_{args.episode_index:04d}_{config['bracket_slot']}"
    )
    trial.mkdir(parents=True, exist_ok=False)
    diagnostics = {
        "cell": config["bracket_slot"], "episode_index_for_following_placement": args.episode_index,
        "source_candidate": str(candidate), "dataset_episode": False, "camera_recording": False,
        "cell_release_open_speed_m_s": release_speed,
        "cell_settle_before_open_s": before_open_s,
        "cell_settle_after_open_s": after_open_s,
        "stages": {}, "completed": False,
        "operator_confirmed_stationary_at_cell": False,
    }
    may_hold = False

    def stage(name, trajectory, holding):
        nonlocal robot
        live_error = float(np.max(np.abs(np.asarray(robot.state.q) - trajectory.expected_start_q)))
        if live_error > float(config["controller_start_tolerance_rad"]):
            raise RuntimeError(f"{name} start error {live_error:.6f} rad")
        robot = None
        gc.collect()
        time.sleep(0.25)
        telemetry = trial / f"{name}.csv"
        bounds_name = "retreat_open" if name == "wall_approach_open" else (
            "transport_place_closed" if holding else "approach_open"
        )
        run_direct(binary, config["robot_ip"], trajectory, name,
                   config["stages"][bounds_name]["workspace_bounds_m"], telemetry)
        robot = Robot(config["robot_ip"])
        error = float(np.max(np.abs(np.asarray(robot.state.q) - trajectory.expected_stop_q)))
        if error > float(config["reset_terminal_tolerance_rad"]):
            raise RuntimeError(f"{name} terminal error {error:.6f} rad")
        diagnostics["stages"][name] = {
            "duration_s": trajectory.duration_s, "terminal_error_rad": error,
            "telemetry": str(telemetry), "holding_bracket": holding,
        }

    try:
        if approach_from_retreat:
            stage("wall_approach_open", reverse_retreat, False)
        accepted = bool(gripper.grasp(
            float(grasp_cfg["grasp_target_width_m"]), float(grasp_cfg["grasp_speed_m_s"]),
            float(grasp_cfg["grasp_force_n"]), float(grasp_cfg["epsilon_inner_m"]),
            float(grasp_cfg["epsilon_outer_m"]),
        ))
        held_width = float(gripper.state.width)
        may_hold = True
        held = bool(gripper.is_grasped)
        diagnostics["wall_grasp"] = {"accepted": accepted, "is_grasped": held, "width_m": held_width}
        if not accepted or not held or not (
            float(grasp_cfg["held_width_min_m"]) <= held_width <= float(grasp_cfg["held_width_max_m"])
        ):
            raise RuntimeError("wall bracket grasp failed validation")

        stage("wall_to_cell_held", reverse_transport, True)
        if not bool(gripper.is_grasped):
            raise RuntimeError("Hand lost bracket during wall-to-cell transfer")
        print(f"[cell] stationary settle before opening: {before_open_s:.1f} s")
        time.sleep(before_open_s)
        gripper.open(release_speed)
        open_width = float(gripper.state.width)
        diagnostics["cell_release"] = {"open_width_m": open_width, "speed_m_s": release_speed}
        if open_width < float(grasp_cfg["minimum_open_width_m"]) or bool(gripper.is_grasped):
            raise RuntimeError("Hand did not open at cell")
        may_hold = False
        print(f"[cell] stationary settle before retreat: {after_open_s:.1f} s")
        time.sleep(after_open_s)
        stage("cell_retreat_open", reverse_approach, False)
        diagnostics["completed"] = True
        print(f"[DONE] setup move diagnostics: {trial}")
        print("Bracket is at the cell; no dataset episode was created.")
    except BaseException as error:
        diagnostics["failure_reason"] = str(error)
        if may_hold:
            print("[SAFETY] Hand may hold the bracket; no automatic release issued")
        raise
    finally:
        (trial / "setup_result.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    if args.setup_cycle_only:
        robot = None
        gripper = None
        gc.collect()
        command = [
            sys.executable, str(ROOT / "model-running" / "run_manual_bracket_pick_and_place.py"),
            "--config", str(config_path),
        ]
        print("[setup cycle] returning the bracket from the same cell pose to the fixed wall pose; no cameras or dataset episode")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(f"[STOPPED] cell-to-wall setup exited {completed.returncode}")
        return
    if not args.inverse_only:
        if input("Confirm bracket is stable at the cell; press Enter for the separate placement recording, type anything to stop: "):
            print("[STOPPED] placement recording was not started")
            return
        diagnostics["operator_confirmed_stationary_at_cell"] = True
        (trial / "setup_result.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        robot = None
        gripper = None
        gc.collect()
        command = [
            sys.executable, str(ROOT / "model-running" / "record_autonomous_bolt_engagement_demo.py"),
            "--record-demo", "--full-assembly", "--bracket-only", "--four-x",
            "--bracket-source", "manual_2026_08_16", "--bracket-arm-config", str(config_path),
            "--episode-index", str(args.episode_index),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(f"[STOPPED] placement recording exited {completed.returncode}")


if __name__ == "__main__":
    main()
