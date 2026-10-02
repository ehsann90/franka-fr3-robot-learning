#!/usr/bin/env python3
"""Read-only capture from held PREINSERT through drop and head regrasp."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bolt_drop_regrasp_v1.json"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trajectory-index", type=int, default=1)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        raw_config = config_path.read_bytes()
        config = json.loads(raw_config)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read capture config {config_path}: {error}") from error
    if config.get("capture_scope") != "preinsert_through_thread_ready_only":
        raise SystemExit("capture must start at PREINSERT and stop at THREAD_READY")
    if config.get("replay", {}).get("enabled") is not False:
        raise SystemExit("replay must remain disabled")
    if config.get("contact_threading", {}).get("enabled") is not False:
        raise SystemExit("contact/threading must remain disabled")
    if args.trajectory_index < 1:
        raise SystemExit("--trajectory-index must be at least 1")
    output_root = project_path(config["output_dir"])
    source = project_path(config["source_preinsert_trajectory"])

    print("=== FR3 PILOTED BOLT DROP + HEAD REGRASP CAPTURE ===")
    print(f"mode:                 {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:             {config['robot_ip']}")
    print(f"source PREINSERT:     {source}")
    print(f"robot sample Hz:      {config['robot_sample_hz']}")
    print(f"output root:          {output_root}")
    print("start state:          bolt HELD at validated NO-CONTACT PREINSERT")
    print("endpoint:             bolt UPRIGHT and REGRASPED by head; no rotation")
    print("arm/Hand commands:    DISABLED BY DESIGN")
    print("contact/threading:    DISABLED / NOT_CALIBRATED")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return

    if not (source / "metadata.json").exists():
        raise SystemExit(f"source PREINSERT trajectory does not exist: {source}")
    print("\nFirst use the commissioned pickup-to-PREINSERT replay so the bolt is held at PREINSERT.")
    print("Use supported Franka Desk/Pilot controls for all motion and Hand operation below.")
    input("Press Enter to connect read-only once the held bolt is at PREINSERT (Ctrl+C cancels): ")

    from franky import Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    recorder = PilotedTrajectoryRecorder(
        robot,
        None,
        robot_hz=float(config["robot_sample_hz"]),
        gripper_hz=float(config["gripper_sample_hz"]),
    )
    preflight = recorder.preflight()
    xyz = preflight["translation_xyz_m"]
    print(
        f"[capture] preflight OK; xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f}) "
        "Hand client=DISABLED"
    )
    input("Press Enter immediately before piloting the held bolt down into the guide: ")

    recorder.start(start_gripper_thread=False)
    recorder.add_event("trajectory_started_preinsert_held")
    stop_requested = threading.Event()
    result = {"accepted": False}
    interrupted = False

    def wait_for_markers() -> None:
        input(
            "Pilot downward until the bolt is safely self-supported and upright in the guide; "
            "hold still, then press Enter: "
        )
        recorder.add_event("drop_pose_reached_held")
        input(
            "Open the Hand without moving sideways; visually verify the bolt remains upright, "
            "then press Enter: "
        )
        recorder.add_event("bolt_released_upright")
        input(
            "With fingers open, pilot clear and reapproach the bolt head; hold at the head-grasp "
            "pose, then press Enter: "
        )
        recorder.add_event("thread_regrasp_pose_reached_open")
        response = input(
            "Close securely around the head without turning the bolt. Visually verify THREAD_READY. "
            "Press Enter to stop and ACCEPT; type anything to stop and reject: "
        ).strip()
        recorder.add_event("thread_ready_regrasped")
        result["accepted"] = response == ""
        stop_requested.set()

    print("[CAPTURING] Each Enter records the completed physical phase.")
    input_thread = threading.Thread(target=wait_for_markers, daemon=True)
    input_thread.start()
    try:
        stop_requested.wait()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[capture] interrupted; raw trajectory will be retained as rejected")
    finally:
        recorder.stop()
    if not interrupted:
        input_thread.join(timeout=1.0)

    accepted = not interrupted and bool(result["accepted"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_bolt_drop_regrasp_pilot_{args.trajectory_index:04d}_S1_CENTER"
    capture_dir = output_root / name
    recorder.write(
        capture_dir,
        {
            "trajectory_name": name,
            "task_name": config["task_name"],
            "language_prompt": config["language_prompt"],
            "capture_scope": config["capture_scope"],
            "collection_mode": "kinesthetic_pilot_read_only",
            "robot_ip": config["robot_ip"],
            "screw_slot": config["screw_slot"],
            "screw_grasp_style": config["screw_grasp_style"],
            "trajectory_index": args.trajectory_index,
            "source_preinsert_trajectory": str(source),
            "mechanical_normalization_method": "release_into_guide_then_regrasp_head",
            "bolt_upright_at_endpoint": accepted,
            "bolt_regrasped_at_endpoint": accepted,
            "rotation_or_threading_included": False,
            "cameras_recorded": False,
            "arm_command_available": False,
            "gripper_command_available": False,
            "gripper_state_recorded": False,
            "gripper_event_mode": "ordered_semantic_markers",
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(raw_config).hexdigest(),
            "notes": args.notes,
            "interrupted": interrupted,
        },
        accepted=accepted,
    )
    print(f"[capture] wrote {len(recorder.samples)} samples to {capture_dir}")
    print(f"[capture] accepted_by_operator={accepted} sample_errors={len(recorder.errors)}")
    print("[capture] automatic replay and threading remain disabled; validate offline next")


if __name__ == "__main__":
    main()
