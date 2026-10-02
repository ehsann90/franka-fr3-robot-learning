#!/usr/bin/env python3
"""Preview or execute one guarded B1/CENTER bracket episode with Hand actions."""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.object_episode_execution import (  # noqa: E402
    REQUIRED_STAGES,
    automatic_reset_blockers,
    object_episode_blockers,
)
from expert_demo.replay_execution import smoothed_position_velocities  # noqa: E402
from expert_demo.replay_plan import ReplayPlanError, build_replay_stage  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json"
DEFAULT_RECORDING_CONFIG = FR3_ROOT / "configs" / "autonomous_demo_collection_v1.json"
JOINT_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
JOINT_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_recording_config(path: Path) -> tuple[dict, dict]:
    document = json.loads(path.expanduser().resolve().read_text())
    task = document["tasks"]["bracket_placement"]
    roles = [entry.get("role") for entry in document["camera"]["serials_in_dataset_order"]]
    if document.get("schema_version") != 1:
        raise ValueError("autonomous recording config must use schema_version=1")
    if roles != ["exterior", "wrist"]:
        raise ValueError("camera order must be exactly exterior, wrist")
    if document["dataset"].get("schema") != "droid_joint_position":
        raise ValueError("autonomous bracket demos require droid_joint_position")
    return document, task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument(
        "--recording-preview",
        action="store_true",
        help="validate both cameras and synchronized read-only state without motion",
    )
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--record-demo", action="store_true")
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recording-config", type=Path, default=DEFAULT_RECORDING_CONFIG)
    args = parser.parse_args()
    if args.record_demo and not args.execute:
        raise SystemExit("--record-demo requires --execute")
    if args.episode_index < 1:
        raise SystemExit("--episode-index must be at least 1")

    recording_document: dict | None = None
    recording_task: dict | None = None
    if args.recording_preview or args.record_demo:
        try:
            recording_document, recording_task = load_recording_config(args.recording_config)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"[recording] invalid autonomous recording config: {error}") from error
        if project_path(recording_task["trajectory_dir"]) != args.trajectory_dir.expanduser().resolve():
            raise SystemExit("[recording] production task trajectory does not match --trajectory-dir")
        if args.record_demo and recording_task.get("enabled") is not True:
            raise SystemExit("[SAFETY] autonomous bracket recording remains disabled pending preview")

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    try:
        root_config = json.loads(args.config.expanduser().resolve().read_text())
        episode = root_config["replay"]["object_episode_execution"]
        plans = {
            name: build_replay_stage(
                trajectory_dir,
                name,
                command_rate_hz=float(root_config["replay"]["command_rate_hz"]),
            )
            for name in REQUIRED_STAGES
        }
        static_blockers = object_episode_blockers(episode, plans)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[episode] invalid input: {error}") from error

    scale = float(episode["playback_time_scale"])
    smoothing_passes = int(episode["command_velocity_smoothing_passes"])
    velocity_source = episode["command_velocity_source"]
    if velocity_source == "recorded_scaled":
        command_velocities = {name: plan.dq_targets / scale for name, plan in plans.items()}
    elif velocity_source == "position_derivative_smoothed":
        command_velocities = {
            name: smoothed_position_velocities(plan, scale, smoothing_passes)
            for name, plan in plans.items()
        }
    else:
        raise SystemExit(f"[episode] unknown command velocity source {velocity_source!r}")
    maximum_command_velocity = max(
        float(np.max(np.abs(velocities))) for velocities in command_velocities.values()
    )
    maximum_command_jump = max(
        float(np.max(np.abs(np.diff(velocities, axis=0))))
        for velocities in command_velocities.values()
    )
    gripper_config = episode["gripper"]
    preview_mode = args.read_only_preview or args.recording_preview
    mode_name = "RECORDING_PREVIEW" if args.recording_preview else "READ_ONLY_PREVIEW" if args.read_only_preview else "EXECUTE_RECORD" if args.record_demo else "EXECUTE"
    print("=== GUARDED PILOTED BRACKET OBJECT EPISODE ===")
    print(f"mode:                     {mode_name}")
    print(f"robot IP:                 {root_config['robot_ip']}")
    print(f"trajectory:               {trajectory_dir}")
    print(f"variation:                {episode['bracket_slot']} / {episode['grasp_style']}")
    print(f"playback:                 {scale:.1f}x slower than piloted recording")
    print(f"expected arm duration:    {sum(p.source_duration_s for p in plans.values()) * scale:.3f} s")
    print(f"command velocity source:  {velocity_source}")
    print(f"maximum command |dq|:     {maximum_command_velocity:.6f} rad/s")
    print(f"maximum command dq jump:  {maximum_command_jump:.6f} rad/s")
    print(f"stage stationary lead-in: {episode['stage_lead_in_enabled']}")
    print(f"grasp target width:       {gripper_config['grasp_target_width_m']:.3f} m")
    print(f"grasp speed:              {gripper_config['grasp_speed_m_s']:.3f} m/s")
    print(f"grasp force:              {gripper_config['grasp_force_n']:.1f} N")
    print(f"accepted held width:      {gripper_config['held_width_min_m']:.3f} .. {gripper_config['held_width_max_m']:.3f} m")
    print(f"camera recording:         {'PRODUCTION EPISODE' if args.record_demo else '2-SECOND READ-ONLY SMOKE' if args.recording_preview else 'DISABLED'}")
    print("sequence:                 open -> approach -> grasp/verify -> transport/place -> open -> retreat")
    if static_blockers:
        print("\nStatic execution blockers:")
        for blocker in static_blockers:
            print(f"  - {blocker}")
        if args.execute:
            raise SystemExit("[SAFETY] object episode is not commissioned; no hardware connection attempted")

    input("Press Enter to connect and validate live arm and Hand state (Ctrl+C cancels): ")
    from franky import (  # noqa: PLC0415
        Duration,
        Frame,
        Gripper,
        JointMotion,
        JointState,
        JointWaypoint,
        JointWaypointMotion,
        Robot,
    )

    robot = Robot(root_config["robot_ip"])
    gripper = Gripper(root_config["robot_ip"])
    stage_global_dynamics = float(episode["stage_robot_global_dynamics_factor"])
    robot.relative_dynamics_factor = stage_global_dynamics
    absolute_velocity_limits = np.asarray(robot.joint_velocity_limit.max, dtype=np.float64)
    effective_velocity_limits = (
        absolute_velocity_limits
        * stage_global_dynamics
        * float(episode["relative_dynamics_factor"])
    )
    per_joint_command_peaks = np.max(
        np.abs(np.concatenate(list(command_velocities.values()), axis=0)), axis=0
    )
    initial = robot.state
    current_q = np.asarray(initial.q, dtype=np.float64)
    start_error = float(np.max(np.abs(current_q - plans["approach_open"].expected_start_q)))
    runtime_blockers = list(static_blockers)
    reset_required = start_error > float(episode["start_joint_tolerance_rad"])
    if absolute_velocity_limits.shape != (7,) or not np.all(np.isfinite(absolute_velocity_limits)):
        runtime_blockers.append("Franky returned invalid joint velocity limits")
    elif np.any(per_joint_command_peaks > 0.95 * effective_velocity_limits):
        runtime_blockers.append(
            "2x command velocities exceed 95% of the live Franky dynamics-scaled limits"
        )

    all_q = np.concatenate([plans[name].q_targets for name in REQUIRED_STAGES])
    margin = float(episode["joint_margin_rad"])
    if np.any(all_q < JOINT_MIN + margin) or np.any(all_q > JOINT_MAX - margin):
        runtime_blockers.append("recorded episode violates guarded joint range")
    measured_tcp = np.asarray(initial.O_T_EE.matrix, dtype=np.float64)
    joint7_current = np.asarray(robot.model.pose(Frame.Joint7, initial).matrix, dtype=np.float64)
    joint7_to_tcp = np.linalg.solve(joint7_current, measured_tcp)
    all_recorded_xyz = np.concatenate([plans[name].xyz_targets for name in REQUIRED_STAGES])
    predicted_xyz = np.asarray([
        (
            np.asarray(
                robot.model.pose(Frame.Joint7, q, initial.F_T_EE, initial.EE_T_K).matrix,
                dtype=np.float64,
            )
            @ joint7_to_tcp
        )[:3, 3]
        for q in all_q
    ])
    fk_error = float(np.max(np.linalg.norm(predicted_xyz - all_recorded_xyz, axis=1)))
    if fk_error > float(episode["maximum_fk_path_error_m"]):
        runtime_blockers.append(f"FK path discrepancy {fk_error:.6f} m exceeds configured maximum")
    workspace = episode["workspace_bounds_m"]
    minimum = np.asarray(workspace["min_xyz"], dtype=np.float64)
    maximum = np.asarray(workspace["max_xyz"], dtype=np.float64)
    if np.any(predicted_xyz < minimum) or np.any(predicted_xyz > maximum):
        runtime_blockers.append("predicted episode path leaves configured workspace")
    initial_gripper_width = float(gripper.state.width)
    initial_is_grasped = bool(gripper.is_grasped)

    reset = episode["automatic_reset_to_start"]
    reset_q_path = np.linspace(current_q, plans["approach_open"].expected_start_q, num=101)
    reset_xyz = np.asarray([
        (
            np.asarray(
                robot.model.pose(Frame.Joint7, q, initial.F_T_EE, initial.EE_T_K).matrix,
                dtype=np.float64,
            )
            @ joint7_to_tcp
        )[:3, 3]
        for q in reset_q_path
    ])
    reset_target_fk_error = float(
        np.linalg.norm(reset_xyz[-1] - plans["approach_open"].xyz_targets[0])
    )
    reset_blockers = automatic_reset_blockers(
        reset,
        current_q,
        plans["approach_open"].expected_start_q,
        reset_xyz,
        reset_target_fk_error,
    )
    reset_margin = float(reset["joint_margin_rad"])
    if np.any(reset_q_path < JOINT_MIN + reset_margin) or np.any(
        reset_q_path > JOINT_MAX - reset_margin
    ):
        reset_blockers.append("computed automatic reset violates guarded joint range")
    if initial_is_grasped:
        reset_blockers.append("Hand reports a grasp; automatic empty-gripper reset is refused")
        runtime_blockers.append("Hand must be empty before starting a bracket episode")

    print(f"\nlive start error:         {start_error:.6f} rad")
    print(f"live gripper width:       {initial_gripper_width:.4f} m")
    print(f"live gripper is_grasped:  {initial_is_grasped}")
    print(f"Franky velocity limits:   {np.array2string(absolute_velocity_limits, precision=4)} rad/s")
    print(
        f"effective limits ({stage_global_dynamics:.2f}x global * "
        f"{episode['relative_dynamics_factor']:.2f}x motion): "
        f"{np.array2string(effective_velocity_limits, precision=4)} rad/s"
    )
    print(f"command peaks by joint:   {np.array2string(per_joint_command_peaks, precision=4)} rad/s")
    print(f"predicted path xyz min:   {np.array2string(predicted_xyz.min(axis=0), precision=5)}")
    print(f"predicted path xyz max:   {np.array2string(predicted_xyz.max(axis=0), precision=5)}")
    print(f"maximum FK path error:    {fk_error:.6f} m")
    if reset_required:
        print("\nautomatic reset:          REQUIRED")
        print(f"reset maximum q delta:    {float(np.max(np.abs(reset_q_path[-1] - reset_q_path[0]))):.6f} rad")
        print(f"reset path xyz min:       {np.array2string(reset_xyz.min(axis=0), precision=5)}")
        print(f"reset path xyz max:       {np.array2string(reset_xyz.max(axis=0), precision=5)}")
        print(f"reset target FK error:    {reset_target_fk_error:.6f} m")
        if reset_blockers:
            print("Automatic reset blockers:")
            for blocker in dict.fromkeys(reset_blockers):
                print(f"  - {blocker}")
    else:
        print("automatic reset:          NOT NEEDED")
    if runtime_blockers:
        print("\nExecution blockers:")
        for blocker in dict.fromkeys(runtime_blockers):
            print(f"  - {blocker}")
    if preview_mode:
        if args.recording_preview:
            assert recording_document is not None
            sys.path.insert(0, str(FR3_ROOT / "franka-tests"))
            from realsense_recorder import RealSenseEpisodeRecorder  # noqa: PLC0415

            camera = recording_document["camera"]
            serials = [entry["serial"] for entry in camera["serials_in_dataset_order"]]
            with tempfile.TemporaryDirectory(prefix="fr3_bracket_recording_preview_") as temp:
                recorder = RealSenseEpisodeRecorder(
                    temp,
                    serials=serials,
                    width=int(camera["width"]),
                    height=int(camera["height"]),
                    fps=int(camera["source_fps"]),
                )
                try:
                    recorder.attach_robot(
                        robot,
                        gripper=None,
                        sample_hz=float(recording_document["robot_state_sample_hz"]),
                        gripper_command_source="unavailable",
                    )
                    recorder.set_achieved_gripper_state(
                        initial_gripper_width,
                        initial_is_grasped,
                        "recording_preview_initial_hand_state",
                        command_width=initial_gripper_width,
                    )
                    recorder.start_episode(
                        "bracket_recording_preview",
                        {"task_name": "bracket", "bc_training_eligible": False},
                    )
                    time.sleep(2.0)
                    frame_counts = dict(recorder.camera_frame_counts)
                    state_count = len(recorder.state_samples)
                    recorder.stop_episode(False, {"preview_only": True})
                finally:
                    recorder.close()
            minimum_frames = int(camera["source_fps"])
            camera_failures = [name for name, count in frame_counts.items() if count < minimum_frames]
            minimum_states = int(float(recording_document["robot_state_sample_hz"]))
            if camera_failures or state_count < minimum_states:
                print("\nRecording preview blockers:")
                if camera_failures:
                    print("  - insufficient frames from: " + ", ".join(camera_failures))
                if state_count < minimum_states:
                    print(f"  - only {state_count} robot states were sampled")
                raise SystemExit("[SAFETY] production recording preview failed")
            print(f"\nrecording preview frames: {frame_counts}")
            print(f"recording preview states: {state_count}")
            print("Recording preview passed. Cameras and state sampling were read-only.")
        print("\nPreview complete. No arm or gripper command was issued.")
        return
    if runtime_blockers:
        raise SystemExit("[SAFETY] object episode execution refused")

    stop_requested = threading.Event()
    may_be_holding = False

    def stop_all() -> None:
        stop_requested.set()
        robot.stop()
        try:
            gripper.stop()
        except BaseException:
            pass

    def interrupt(signum, frame) -> None:
        del signum, frame
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; stopping arm and Hand")
        stop_all()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)

    if reset_required:
        if reset_blockers:
            raise SystemExit("[SAFETY] automatic reset refused")
        input(
            "Remove the bracket from B1, clear the robot path, and press Enter for the "
            "slow automatic arm-only reset (Ctrl+C cancels): "
        )
        try:
            robot.move(
                JointMotion(
                    JointState(plans["approach_open"].expected_start_q),
                    relative_dynamics_factor=float(reset["relative_dynamics_factor"]),
                )
            )
        except BaseException:
            stop_all()
            raise
        reset_final_q = np.asarray(robot.state.q, dtype=np.float64)
        reset_final_error = float(
            np.max(np.abs(reset_final_q - plans["approach_open"].expected_start_q))
        )
        print(f"[auto-reset] final q:     {np.array2string(reset_final_q, precision=5)}")
        print(f"[auto-reset] max error:   {reset_final_error:.6f} rad")
        if reset_final_error > float(reset["terminal_joint_tolerance_rad"]):
            raise SystemExit("[SAFETY] automatic reset ended outside its terminal tolerance")
        print("[auto-reset] complete; no gripper command was issued")

    # The reset motion has its own local factor; stage motions always use this
    # explicit global factor multiplied by their configured motion factor.
    robot.relative_dynamics_factor = stage_global_dynamics

    recorder = None
    episode_dir = None
    if args.record_demo:
        assert recording_document is not None and recording_task is not None
        sys.path.insert(0, str(FR3_ROOT / "franka-tests"))
        from realsense_recorder import RealSenseEpisodeRecorder  # noqa: PLC0415

        camera = recording_document["camera"]
        serials = [entry["serial"] for entry in camera["serials_in_dataset_order"]]
        print("[record] initializing exterior and wrist cameras ...")
        recorder = RealSenseEpisodeRecorder(
            project_path(recording_task["output_dir"]),
            serials=serials,
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["source_fps"]),
        )
        recorder.attach_robot(
            robot,
            gripper=None,
            sample_hz=float(recording_document["robot_state_sample_hz"]),
            gripper_command_source="unavailable",
        )
        recorder.set_achieved_gripper_state(
            float(gripper.state.width),
            bool(gripper.is_grasped),
            "initial_hand_state",
            command_width=float(gripper.state.width),
        )

    print(
        f"\nPlace the bracket precisely in {episode['bracket_slot']}. "
        "Ensure the wall pins and full robot path are clear."
    )
    if args.record_demo:
        confirmation = input(
            "Press Enter to begin the recorded bracket episode; type anything to abort: "
        ).strip()
        confirmed = confirmation == ""
    else:
        confirmation = input(
            "Type RUN_BRACKET_B1_CENTER to run the live bracket episode; anything else aborts: "
        )
        confirmed = confirmation == "RUN_BRACKET_B1_CENTER"
    if not confirmed:
        print("[episode] aborted without bracket-episode motion")
        if recorder is not None:
            recorder.close()
        return

    if recorder is not None:
        assert recording_document is not None and recording_task is not None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        episode_name = (
            f"{stamp}_bracket_autonomous_{args.episode_index:04d}_"
            f"{episode['bracket_slot']}_{episode['grasp_style']}"
        )
        episode_dir = recorder.start_episode(
            episode_name,
            {
                "task_name": "bracket",
                "task": recording_task["language_prompt"],
                "language_prompt": recording_task["language_prompt"],
                "collection_mode": recording_document["collection_mode"],
                "robot_ip": root_config["robot_ip"],
                "bracket_slot": episode["bracket_slot"],
                "bracket_grasp_style": episode["grasp_style"],
                "screw_slot": None,
                "screw_grasp_style": None,
                "episode_index": args.episode_index,
                "random_seed": args.seed,
                "source_trajectory": str(trajectory_dir),
                "commissioned_replay_config": str(args.config.expanduser().resolve()),
                "recording_config": str(args.recording_config.expanduser().resolve()),
                "dataset_target_fps": recording_document["dataset"]["target_fps"],
                "dataset_schema": recording_document["dataset"]["schema"],
                "action_dimension": recording_document["dataset"]["action_dimension"],
                "action_semantics": recording_document["dataset"]["action_semantics"],
                "expert_command_available": True,
                "gripper_command_available": True,
                "gripper_state_source": "automatic_command_measured_at_transition",
                "bc_training_eligible": True,
                "success_endpoint": recording_task["success_endpoint"],
                "camera_roles": {
                    entry["role"]: entry["serial"]
                    for entry in recording_document["camera"]["serials_in_dataset_order"]
                },
            },
        )
        recorder.add_event("autonomous_bracket_episode_started")

    def run_arm_stage(name: str, *, monitor_grasp: bool) -> None:
        plan = plans[name]
        scaled_dq = command_velocities[name]
        waypoint_ms = int(math.ceil(1000.0 * scale / plan.command_rate_hz))
        waypoints = []
        if episode["stage_lead_in_enabled"]:
            waypoints.append(
                JointWaypoint(
                    JointState(plan.q_targets[0], np.zeros(7)),
                    minimum_time=Duration(int(episode["stage_lead_in_ms"])),
                    hold_target_duration=Duration(int(episode["stage_lead_in_hold_ms"])),
                    max_total_duration=Duration(3000),
                )
            )
        waypoints.extend([
            JointWaypoint(
                JointState(q, dq),
                minimum_time=Duration(waypoint_ms),
                max_total_duration=Duration(2000),
            )
            for q, dq in zip(plan.q_targets[1:], scaled_dq[1:], strict=True)
        ])
        motion = JointWaypointMotion(
            waypoints,
            relative_dynamics_factor=float(episode["relative_dynamics_factor"]),
        )
        monitor_stop = threading.Event()
        lost_grasp = threading.Event()

        def monitor() -> None:
            period = 1.0 / float(gripper_config["monitor_hz"])
            while not monitor_stop.wait(period):
                try:
                    if not bool(gripper.is_grasped):
                        lost_grasp.set()
                        robot.stop()
                        return
                except BaseException:
                    lost_grasp.set()
                    robot.stop()
                    return

        thread = threading.Thread(target=monitor, name="grasp-monitor", daemon=True)
        if monitor_grasp:
            thread.start()
        timer = threading.Timer(float(episode["state_timeout_s"]), stop_all)
        timer.daemon = True
        timer.start()
        try:
            robot.move(motion)
        finally:
            timer.cancel()
            monitor_stop.set()
            if monitor_grasp:
                thread.join(timeout=1.0)
        if stop_requested.is_set():
            raise RuntimeError(f"{name} stopped by timeout or emergency stop")
        if lost_grasp.is_set():
            raise RuntimeError(f"{name} aborted because the Hand no longer reported a grasp")
        terminal_error = float(np.max(np.abs(np.asarray(robot.state.q) - plan.expected_stop_q)))
        print(f"[{name}] terminal q error={terminal_error:.6f} rad")
        if terminal_error > float(episode["terminal_joint_tolerance_rad"]):
            raise RuntimeError(f"{name} ended outside terminal joint tolerance")

    try:
        print("[episode] opening Hand")
        gripper.open(float(gripper_config["open_speed_m_s"]))
        open_width = float(gripper.state.width)
        if open_width < float(gripper_config["minimum_open_width_m"]):
            raise RuntimeError("Hand failed to reach the required open width")
        if recorder is not None:
            recorder.set_achieved_gripper_state(
                open_width,
                False,
                "episode_hand_opened",
                command_width=0.08,
            )
            recorder.add_event("approach_open_started")
        run_arm_stage("approach_open", monitor_grasp=False)
        if recorder is not None:
            recorder.add_event("approach_open_completed")

        print("[episode] grasping bracket while arm is stationary")
        accepted = gripper.grasp(
            float(gripper_config["grasp_target_width_m"]),
            float(gripper_config["grasp_speed_m_s"]),
            float(gripper_config["grasp_force_n"]),
            float(gripper_config["epsilon_inner_m"]),
            float(gripper_config["epsilon_outer_m"]),
        )
        may_be_holding = True
        held_width = float(gripper.state.width)
        grasped = bool(gripper.is_grasped)
        print(f"[grasp] accepted={accepted} is_grasped={grasped} width={held_width:.4f} m")
        if not accepted or not grasped:
            raise RuntimeError("Hand did not verify the bracket grasp; transport was not started")
        if not float(gripper_config["held_width_min_m"]) <= held_width <= float(gripper_config["held_width_max_m"]):
            raise RuntimeError("measured held width is outside the bracket acceptance range")
        if recorder is not None:
            recorder.set_achieved_gripper_state(
                held_width,
                grasped,
                "bracket_grasped",
                command_width=float(gripper_config["grasp_target_width_m"]),
            )
            recorder.add_event("transport_place_closed_started")

        run_arm_stage("transport_place_closed", monitor_grasp=True)
        if recorder is not None:
            recorder.add_event("transport_place_closed_completed")
        if not bool(gripper.is_grasped):
            raise RuntimeError("grasp was lost before release")
        print("[episode] opening Hand at the recorded wall placement")
        gripper.open(float(gripper_config["open_speed_m_s"]))
        may_be_holding = False
        release_width = float(gripper.state.width)
        if release_width < float(gripper_config["minimum_open_width_m"]):
            raise RuntimeError("Hand did not reach the required release width")
        if recorder is not None:
            recorder.set_achieved_gripper_state(
                release_width,
                False,
                "bracket_released",
                command_width=0.08,
            )
            recorder.add_event("retreat_open_started")
        run_arm_stage("retreat_open", monitor_grasp=False)
        if recorder is not None:
            recorder.add_event("retreat_open_completed")
    except BaseException:
        stop_all()
        if recorder is not None and recorder.active:
            recorder.add_event("autonomous_bracket_episode_failed")
            recorder.stop_episode(
                False,
                {
                    "failure_reason": "execution_exception",
                    "termination_reason": "guard_or_controller_exception",
                    "bc_training_eligible": False,
                },
            )
            recorder.close()
        if may_be_holding:
            print("[SAFETY] The Hand may still hold the bracket; it was NOT automatically opened.")
        raise
    if recorder is not None:
        accepted_episode = input(
            "Visually confirm the bracket is mounted, the Hand is open, and the arm retreated. "
            "Press Enter to accept this training demo; type anything to reject: "
        ) == ""
        recorder.add_event(
            "autonomous_bracket_episode_finished",
            {"accepted_by_operator": accepted_episode},
        )
        recorder.stop_episode(
            accepted_episode,
            {
                "accepted_by_operator": accepted_episode,
                "failure_reason": None if accepted_episode else "operator_rejected_final_result",
                "termination_reason": "verified_success" if accepted_episode else "operator_rejected",
                "bc_training_eligible": accepted_episode,
            },
        )
        recorder.close()
        if not accepted_episode:
            raise SystemExit(f"[REJECTED] raw episode retained outside training split: {episode_dir}")
        print(f"[SUCCESS] autonomous bracket training demo saved: {episode_dir}")
    else:
        print("[SUCCESS] bracket episode completed; camera recording was disabled")


if __name__ == "__main__":
    main()
