#!/usr/bin/env python3
"""Record a manually piloted FR3 bracket demonstration without commanding motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.calibration import CalibrationStore  # noqa: E402
from realsense_recorder import RealSenseEpisodeRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bracket_manual_recording_v1.json"


def load_config(path: Path) -> tuple[dict, Path, str]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read recording config {resolved}: {error}") from error
    if config.get("schema_version") != 1:
        raise SystemExit(f"unsupported recording config schema: {config.get('schema_version')!r}")
    if config.get("task_name") != "bracket":
        raise SystemExit("manual bracket recording config task_name must be 'bracket'")
    camera_entries = config.get("camera", {}).get("serials_in_dataset_order", [])
    if [entry.get("role") for entry in camera_entries] != ["exterior", "wrist"]:
        raise SystemExit("camera order must be exactly exterior, wrist")
    if config.get("dataset", {}).get("schema") != "droid_joint_position":
        raise SystemExit("manual bracket recorder requires dataset.schema=droid_joint_position")
    manual_states = config.get("manual_gripper_states_m", {})
    if not {"open", "bracket_center_closed"}.issubset(manual_states):
        raise SystemExit("manual bracket recorder requires open/closed gripper marker widths")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def print_banner(config: dict, config_path: Path, output_dir: Path, mode: str) -> None:
    camera = config["camera"]
    dataset = config["dataset"]
    print("=== FR3 MANUAL BRACKET DEMO RECORDER ===")
    print(f"mode:                {mode}")
    print(f"robot IP:            {config['robot_ip']}")
    print(f"prompt:              {config['language_prompt']}")
    print(f"config:              {config_path}")
    print(f"raw output:          {output_dir}")
    print(f"camera source FPS:   {camera['source_fps']}")
    print(f"robot state Hz:      {config['robot_state_sample_hz']}")
    print(f"dataset target FPS:  {dataset['target_fps']}")
    print(f"dataset schema:      {dataset['schema']}")
    print(f"action semantics:    {dataset['action_semantics']}")
    for entry in camera["serials_in_dataset_order"]:
        print(f"camera {entry['role']:<8} serial: {entry['serial']}")
    print("arm motion commands: DISABLED BY DESIGN")
    print("gripper commands:   DISABLED BY DESIGN")
    print("Hand state client:  DISABLED (Desk/Pilot retains Hand control)")
    print("gripper state:      explicit operator-marked transitions")
    print("guide-mode calls:   DISABLED BY DESIGN")
    print("fixed scene:        frame/pins unchanged; bolt remains at S1")


def finalize_operator_review(
    episode_dir: Path,
    *,
    success: bool,
    termination_reason: str,
    failure_reason: str | None,
) -> None:
    """Finalize eligibility after cameras and state sampling have stopped.

    A stopped episode is deliberately left failed/pending until this review is
    written. An interruption can therefore never make an unreviewed capture
    training-eligible.
    """

    with (episode_dir / "events.jsonl").open("a") as events_file:
        events_file.write(
            json.dumps(
                {
                    "time_ns": time.time_ns(),
                    "event": "manual_demo_finished",
                    "payload": {
                        "success": success,
                        "reason": termination_reason,
                        "recording_already_stopped": True,
                    },
                }
            )
            + "\n"
        )
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "success": success,
            "accepted_by_operator": success,
            "pending_visual_review": False,
            "termination_reason": termination_reason,
            "failure_reason": failure_reason,
            "bc_training_eligible": success,
        }
    )
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_path.replace(metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--slot", default="B1")
    parser.add_argument("--grasp", default="CENTER")
    parser.add_argument(
        "--wrist-side",
        choices=["COMMISSIONED", "OUTER"],
        default="COMMISSIONED",
        help=(
            "camera-side label for the manually selected wrist posture; OUTER "
            "must be achieved by safe Pilot positioning, never by an automatic q7 offset"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config, config_path, config_hash = load_config(args.config)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else resolve_project_path(config["output_dir"])
    )
    slot = args.slot.strip().upper()
    grasp = args.grasp.strip().upper()
    wrist_side = args.wrist_side.strip().upper()
    if args.episode_index < 1:
        raise SystemExit("--episode-index must be at least 1")
    if slot not in config["allowed_bracket_slots"] and not re.fullmatch(r"B\d+", slot):
        raise SystemExit(
            f"slot {slot!r} is not enabled; allowed={config['allowed_bracket_slots']} "
            "or any B<number> label (e.g. B3, B4, ...)"
        )
    if grasp not in config["allowed_grasp_styles"]:
        raise SystemExit(f"grasp {grasp!r} is not enabled; allowed={config['allowed_grasp_styles']}")

    print_banner(config, config_path, output_dir, "DRY_RUN" if args.dry_run else "RECORD_READ_ONLY")
    print(f"episode:             {args.episode_index:03d}")
    print(f"bracket slot:        {slot}")
    print(f"grasp style:         {grasp}")
    print(f"wrist camera side:   {wrist_side}")
    print(f"random seed:         {args.seed}")
    if args.dry_run:
        print("\nDry-run complete. No camera, robot, or gripper connection was attempted.")
        return

    print("\nUse only supported Franka Desk/Pilot controls for hand guiding and gripper operation.")
    print("Keep the operator outside both camera views as much as practical.")
    input("Press Enter to initialize cameras and read robot state: ")

    # Hardware libraries are intentionally imported only after explicit confirmation.
    from franky import Robot  # noqa: PLC0415

    camera = config["camera"]
    serials = [entry["serial"] for entry in camera["serials_in_dataset_order"]]
    recorder: RealSenseEpisodeRecorder | None = None
    try:
        print("[record] initializing cameras in fixed exterior/wrist order ...")
        recorder = RealSenseEpisodeRecorder(
            output_dir,
            serials=serials,
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["source_fps"]),
        )
        print(f"[record] connecting read-only state clients to {config['robot_ip']} ...")
        robot = Robot(config["robot_ip"])
        recorder.attach_robot(
            robot,
            gripper=None,
            sample_hz=float(config["robot_state_sample_hz"]),
            gripper_command_source="unavailable",
        )
        gripper_states = config["manual_gripper_states_m"]
        recorder.set_manual_gripper_state(
            float(gripper_states["open"]), False, "initial_hand_open"
        )

        calibration_path = resolve_project_path(config["calibration_file"])
        calibration_hash = None
        if calibration_path.exists():
            calibration_hash = CalibrationStore(calibration_path, config["robot_ip"]).calibration_hash

        print(f"\nEPISODE {args.episode_index:03d}")
        print(f"Place the bracket in {slot}; use the {grasp} grasp variation.")
        print("Place the bolt at S1 in its commissioned orientation and do not move it.")
        print("Keep the frame, wall pins, cameras, lighting, and goal placement unchanged.")
        if wrist_side == "OUTER":
            print(
                "Use Pilot to keep the wrist camera on the OUTER side for approach, "
                "pickup, placement, and retreat."
            )
            print(
                "Select the reverse-roll direction opposite bolt tightening. Do not "
                "force an exact q7 -180-degree offset; choose a comfortable collision-free "
                "whole-arm posture and stop if clearance is uncertain."
            )
        print("Start with the robot at your chosen safe episode-start configuration.")
        print("The recording must include approach, grasp, lift, placement, release, and retreat.")
        input("Press Enter when the scene and cameras are ready to start recording: ")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        episode_name = f"{timestamp}_bracket_manual_{args.episode_index:04d}_{slot}_{grasp}"
        metadata = {
            "task_name": config["task_name"],
            "language_prompt": config["language_prompt"],
            "collection_mode": "manual_pilot_read_only",
            "robot_ip": config["robot_ip"],
            "bracket_slot": slot,
            "bracket_grasp_style": grasp,
            "wrist_camera_side": wrist_side,
            "wrist_posture_source": "operator_selected_kinesthetic_pilot",
            "wrist_roll_direction": (
                "reverse_opposite_bolt_tightening" if wrist_side == "OUTER" else "commissioned"
            ),
            "automatic_q7_offset_applied": False,
            "screw_slot": "S1",
            "screw_grasp_style": "CENTER",
            "fixed_scene_profile": {
                "frame": "commissioned_unchanged",
                "wall_pins": "commissioned_unchanged",
                "bracket_goal": "commissioned_unchanged",
                "bolt_start": "S1_commissioned_orientation_stationary_distractor",
                "camera_layout": "commissioned_unchanged",
            },
            "random_seed": args.seed,
            "episode_index": args.episode_index,
            "notes": args.notes,
            "recording_config": str(config_path),
            "recording_config_sha256": config_hash,
            "calibration_file": str(calibration_path) if calibration_path.exists() else None,
            "calibration_sha256": calibration_hash,
            "camera_roles": {entry["role"]: entry["serial"] for entry in camera["serials_in_dataset_order"]},
            "dataset_target_fps": config["dataset"]["target_fps"],
            "dataset_schema": config["dataset"]["schema"],
            "action_dimension": config["dataset"]["action_dimension"],
            "action_semantics": config["dataset"]["action_semantics"],
            "expert_command_available": False,
            "gripper_command_available": False,
            "gripper_state_source": config["manual_gripper_state_source"],
            "gripper_state_measured": False,
            "setup_validation_only": False,
            "bc_training_eligible": False,
        }
        episode_dir = recorder.start_episode(episode_name, metadata)
        recorder.add_event("manual_demo_started", {"slot": slot, "grasp": grasp})
        print("\n[RECORDING] Pilot toward the bracket with the Hand open.")
        input("At the CENTER grasp pose, hold the arm stationary and press Enter: ")
        recorder.add_event("bracket_grasp_pose_reached")
        input("Use the Pilot top button to close securely; press Enter once held: ")
        recorder.set_manual_gripper_state(
            float(gripper_states["bracket_center_closed"]),
            True,
            "bracket_gripper_closed",
        )
        input(
            "Pilot the lift, transport, and placement. Hold the seated bracket and arm "
            "stationary, then press Enter: "
        )
        recorder.add_event("bracket_place_pose_reached")
        input("Use Pilot to open the Hand without moving the bracket; then press Enter: ")
        recorder.set_manual_gripper_state(
            float(gripper_states["open"]), False, "bracket_gripper_opened"
        )
        print("[RECORDING] Retreat, then hold the arm stationary for one second.")
        print("[RECORDING] Press Enter to stop recording before review.")
        input("Completion: ")

        recorder.add_event(
            "training_motion_completed",
            {
                "task": "bracket",
                "stationary_terminal_context_s": 1.0,
                "operator_review_follows_after_recording": True,
            },
        )
        recorder.stop_episode(
            False,
            {
                "pending_visual_review": True,
                "termination_reason": "pending_visual_review",
                "failure_reason": "pending_visual_review",
                "bc_training_eligible": False,
                "jam_recovery_count": 0,
                "tightening_termination_reason": None,
            },
        )
        print("[record] cameras and robot-state sampling stopped before operator review")

        input("Press Enter after confirming the mounted bracket: ")
        success = True
        failure_reason = None
        termination = "operator_marked_success"
        finalize_operator_review(
            episode_dir,
            success=success,
            termination_reason=termination,
            failure_reason=None if success else failure_reason,
        )
        print(
            f"[record] episode finalized success={success} bc_training_eligible={success}; "
            "failed episodes remain excluded from conversion by default"
        )
    except KeyboardInterrupt:
        print("\n[record] interrupted; marking any active episode failed")
        if recorder is not None and recorder.active:
            recorder.add_event("manual_demo_interrupted")
            recorder.stop_episode(False, {"failure_reason": "keyboard_interrupt"})
        raise SystemExit(130) from None
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
