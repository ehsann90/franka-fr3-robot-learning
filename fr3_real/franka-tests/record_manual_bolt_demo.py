#!/usr/bin/env python3
"""Record one complete Pilot-guided bolt fastening demonstration with cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from realsense_recorder import RealSenseEpisodeRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_manual_recording_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_config(path: Path) -> tuple[dict, Path, str]:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1 or config.get("task_name") != "bolt":
        raise ValueError("expected schema_version=1 and task_name=bolt")
    roles = [entry.get("role") for entry in config["camera"]["serials_in_dataset_order"]]
    if roles != ["exterior", "wrist"]:
        raise ValueError("camera order must be exactly exterior, wrist")
    if config["dataset"]["schema"] != "droid_joint_position":
        raise ValueError("bolt recorder requires droid_joint_position schema")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--slot", default="S1")
    parser.add_argument("--grasp", default="CENTER")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    try:
        config, config_path, config_hash = load_config(args.config)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot load bolt recording config: {error}") from error
    if args.episode_index < 1:
        raise SystemExit("--episode-index must be at least 1")
    slot = args.slot.strip().upper()
    grasp = args.grasp.strip().upper()
    if slot not in config["allowed_screw_slots"]:
        raise SystemExit(f"slot {slot!r} is not enabled")
    if grasp not in config["allowed_grasp_styles"]:
        raise SystemExit(f"grasp {grasp!r} is not enabled")

    camera = config["camera"]
    dataset = config["dataset"]
    output_root = project_path(config["output_dir"])
    print("=== FR3 FULL PILOTED BOLT DEMO RECORDER ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:                {config['robot_ip']}")
    print(f"prompt:                  {config['language_prompt']}")
    print(f"variation:               {slot} / {grasp}")
    print(f"episode index/seed:      {args.episode_index:03d} / {args.seed}")
    print(f"camera source FPS:       {camera['source_fps']}")
    print(f"robot state Hz:          {config['robot_state_sample_hz']}")
    print(f"dataset target/schema:   {dataset['target_fps']} Hz / {dataset['schema']}")
    print(f"action dimension:        {dataset['action_dimension']}")
    print("episode:                 pickup -> drop/stand -> head regrasp -> ratchet/thread -> seat -> release -> retreat")
    print("arm/Hand commands:       DISABLED BY DESIGN")
    print("Hand state client:       DISABLED (Desk/Pilot retains Hand control)")
    print("gripper state:           explicit operator-marked discrete transitions")
    print("automatic threading:     DISABLED")
    if args.dry_run:
        print("\nDry-run complete. No camera, robot, or Hand connection was attempted.")
        return

    input("Press Enter to initialize cameras and the read-only arm client (Ctrl+C cancels): ")
    from franky import Robot  # noqa: PLC0415

    serials = [entry["serial"] for entry in camera["serials_in_dataset_order"]]
    recorder: RealSenseEpisodeRecorder | None = None
    success = False
    try:
        recorder = RealSenseEpisodeRecorder(
            output_root,
            serials=serials,
            width=int(camera["width"]),
            height=int(camera["height"]),
            fps=int(camera["source_fps"]),
        )
        robot = Robot(config["robot_ip"])
        recorder.attach_robot(
            robot,
            gripper=None,
            sample_hz=float(config["robot_state_sample_hz"]),
            gripper_command_source="unavailable",
        )
        widths = config["manual_gripper_states_m"]
        recorder.set_manual_gripper_state(widths["open"], False, "initial_hand_open")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        episode_name = f"{stamp}_bolt_manual_{args.episode_index:04d}_{slot}_{grasp}"
        metadata = {
            "task_name": config["task_name"],
            "language_prompt": config["language_prompt"],
            "collection_mode": "manual_pilot_read_only_with_operator_gripper_markers",
            "robot_ip": config["robot_ip"],
            "bracket_slot": None,
            "bracket_grasp_style": None,
            "screw_slot": slot,
            "screw_grasp_style": grasp,
            "random_seed": args.seed,
            "episode_index": args.episode_index,
            "notes": args.notes,
            "recording_config": str(config_path),
            "recording_config_sha256": config_hash,
            "threading_config": str(project_path(config["threading_config"])),
            "head_width_calibration": str(project_path(config["head_width_calibration"])),
            "camera_roles": {entry["role"]: entry["serial"] for entry in camera["serials_in_dataset_order"]},
            "dataset_target_fps": dataset["target_fps"],
            "dataset_schema": dataset["schema"],
            "action_dimension": dataset["action_dimension"],
            "action_semantics": dataset["action_semantics"],
            "expert_command_available": False,
            "gripper_command_available": False,
            "gripper_state_source": config["manual_gripper_state_source"],
            "bc_training_eligible": bool(config["bc_training_eligible_default"]),
            "intended_use": config["intended_use"],
        }

        print(f"\nPlace the bolt in {slot}; leave the bracket installed.")
        print("Start with the Hand open and the arm at a safe episode-start pose.")
        print("Keep yourself outside both camera views as much as practical.")
        input("Press Enter immediately before beginning the complete demonstration: ")
        recorder.start_episode(episode_name, metadata)
        recorder.add_event("full_bolt_demo_started", {"slot": slot, "grasp": grasp})
        print("[RECORDING] Pilot to the bolt with the Hand open.")

        input("At the pickup grasp pose, keep the arm stationary and press Enter: ")
        recorder.add_event("pickup_grasp_pose_reached")
        input("Use Desk/Pilot to close securely around the pickup grasp; press Enter when held: ")
        recorder.set_manual_gripper_state(
            widths["pickup_closed"], True, "pickup_gripper_closed"
        )

        input(
            "Pilot lift/transfer/lower until the bolt is upright and self-supported in the guide; "
            "hold still and press Enter: "
        )
        recorder.add_event("bolt_self_supported")
        input("Open the Hand without side-loading the bolt; verify it stays upright, then press Enter: ")
        recorder.set_manual_gripper_state(widths["open"], False, "drop_gripper_opened")

        input(
            "With the Hand open, move clear, prewind opposite tightening, and reapproach the head; "
            "press Enter at the head-grasp pose: "
        )
        recorder.add_event("head_regrasp_pose_reached")
        input("Close securely around the head without turning; press Enter when THREAD_READY: ")
        recorder.set_manual_gripper_state(
            widths["thread_ready_head_closed"], True, "head_gripper_closed"
        )

        ratchet_count = 0
        seated = False
        for cycle in range(1, int(config["maximum_manual_ratchet_cycles"]) + 1):
            recorder.add_event("thread_drive_started", {"cycle": cycle})
            response = input(
                f"Drive {cycle}: gently contact and tighten through the safe wrist window. "
                "Press Enter if another ratchet is needed; type s if fully seated; type x to reject: "
            ).strip().lower()
            if response == "s":
                recorder.add_event("thread_drive_complete", {"cycle": cycle, "outcome": "seated"})
                ratchet_count = cycle - 1
                seated = True
                break
            if response != "":
                recorder.add_event("thread_drive_complete", {"cycle": cycle, "outcome": "rejected"})
                break
            recorder.add_event("thread_drive_complete", {"cycle": cycle, "outcome": "ratchet_required"})
            ratchet_count = cycle
            input("Keep the bolt supported, open the Hand without side load, then press Enter: ")
            recorder.set_manual_gripper_state(widths["open"], False, "ratchet_gripper_opened")
            input(
                "Rewind the empty wrist opposite tightening, reapproach, and close securely on the "
                "head; press Enter when ready for the next drive: "
            )
            recorder.set_manual_gripper_state(
                widths["thread_ready_head_closed"], True, "ratchet_head_regrasped"
            )

        if not seated:
            recorder.add_event("full_bolt_demo_finished", {"success": False, "reason": "not_seated"})
            recorder.stop_episode(
                False,
                {
                    "failure_reason": "bolt_not_confirmed_seated",
                    "termination_reason": "operator_rejected_or_cycle_limit",
                    "jam_recovery_count": 0,
                    "ratchet_count": ratchet_count,
                    "tightening_termination_reason": None,
                },
            )
            raise SystemExit("[record] failed episode retained outside the normal BC split")

        input("Open the Hand without disturbing the seated bolt, then press Enter: ")
        recorder.set_manual_gripper_state(widths["open"], False, "seated_release_opened")
        final = input(
            "Pilot a safe retreat. Press Enter to accept the successful episode; type anything to reject: "
        )
        success = final == ""
        recorder.add_event("full_bolt_demo_finished", {"success": success})
        recorder.stop_episode(
            success,
            {
                "failure_reason": None if success else "operator_rejected_final_episode",
                "termination_reason": "operator_confirmed_seated" if success else "operator_rejected",
                "jam_recovery_count": 0,
                "ratchet_count": ratchet_count,
                "tightening_termination_reason": "operator_confirmed_seated" if success else None,
            },
        )
        print(f"[record] episode accepted={success}; raw output root={output_root}")
    except KeyboardInterrupt:
        print("\n[record] interrupted; marking the active episode failed")
        if recorder is not None and recorder.active:
            recorder.add_event("full_bolt_demo_interrupted")
            recorder.stop_episode(False, {"failure_reason": "keyboard_interrupt"})
        raise SystemExit(130) from None
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
