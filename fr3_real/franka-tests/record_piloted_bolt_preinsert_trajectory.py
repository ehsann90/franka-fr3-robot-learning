#!/usr/bin/env python3
"""Read-only capture of a piloted FR3 bolt pickup-through-PREINSERT trajectory."""

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

from expert_demo.calibration import CalibrationStore  # noqa: E402
from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bolt_preinsert_v1.json"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_config(path: Path) -> tuple[dict, Path, str]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read bolt capture config {resolved}: {error}") from error
    if config.get("schema_version") != 1 or config.get("task_name") != "bolt":
        raise SystemExit("expected schema_version=1 and task_name='bolt'")
    if config.get("capture_scope") != "pickup_through_preinsert_only":
        raise SystemExit("bolt capture must explicitly stop at PREINSERT")
    if config.get("replay", {}).get("enabled") is not False:
        raise SystemExit("bolt capture config must explicitly keep replay.enabled=false")
    if config.get("contact_threading", {}).get("enabled") is not False:
        raise SystemExit("bolt capture config must keep contact/threading disabled")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trajectory-index", type=int, default=1)
    parser.add_argument("--slot", default="S1")
    parser.add_argument("--grasp", default="CENTER")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config, config_path, config_hash = load_config(args.config)
    slot = args.slot.strip().upper()
    grasp = args.grasp.strip().upper()
    if args.trajectory_index < 1:
        raise SystemExit("--trajectory-index must be at least 1")
    if slot not in config["allowed_screw_slots"]:
        raise SystemExit(f"screw slot {slot!r} is not enabled")
    if grasp not in config["allowed_grasp_styles"]:
        raise SystemExit(f"screw grasp {grasp!r} is not enabled")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_path(config["output_dir"])
    )

    print("=== FR3 PILOTED BOLT PREINSERT CAPTURE ===")
    print(f"mode:                 {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:             {config['robot_ip']}")
    print(f"task/prompt:          {config['task_name']} / {config['language_prompt']}")
    print(f"variation:            {slot} / {grasp}")
    print(f"trajectory index:     {args.trajectory_index:03d}")
    print(f"robot sample Hz:      {config['robot_sample_hz']}")
    print(f"output root:          {output_root}")
    print("capture endpoint:     PREINSERT — NO CONTACT")
    print("camera recording:     DISABLED (not a training episode)")
    print("arm commands:         DISABLED BY DESIGN")
    print("gripper commands:     DISABLED BY DESIGN")
    print("contact/threading:    DISABLED / NOT_CALIBRATED")
    if args.dry_run:
        print("\nDry-run complete. No robot, Hand, or camera connection was attempted.")
        return

    print("\nLeave the bracket installed on the frame.")
    print("Use supported Franka Desk/Pilot controls for all arm and Hand operation.")
    input("Press Enter to connect the read-only arm-state client (Ctrl+C cancels): ")

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

    calibration_path = project_path(config["calibration_file"])
    calibration_hash = None
    if calibration_path.exists():
        calibration_hash = CalibrationStore(
            calibration_path, config["robot_ip"]
        ).calibration_hash

    print(f"\nPlace the bolt precisely in {slot}, in one repeatable pickup orientation.")
    print(f"Pilot one successful {grasp} pickup-through-PREINSERT trajectory:")
    print("  approach -> grasp -> lift -> transfer -> PREINSERT -> stop")
    print("PREINSERT must remain visibly clear of the hole; do not touch or turn the bolt.")
    input("Press Enter immediately before beginning trajectory capture: ")

    recorder.start(start_gripper_thread=False)
    recorder.add_event("trajectory_started")
    interrupted = False
    stop_requested = threading.Event()
    result = {"accepted": False}

    def wait_for_markers() -> None:
        input(
            "Pilot to the bolt grasp pose with fingers open; hold still, then press Enter: "
        )
        recorder.add_event("grasp_pose_reached")
        input(
            "Use Desk/Pilot to close securely without moving the arm; hold still, then press Enter: "
        )
        recorder.add_event("gripper_closed")
        response = input(
            "Pilot lift/transfer to a NO-CONTACT PREINSERT pose and hold still. "
            "Press Enter to stop and ACCEPT; type anything to stop and reject: "
        ).strip()
        recorder.add_event("preinsert_pose_reached")
        result["accepted"] = response == ""
        stop_requested.set()

    print("[CAPTURING] Follow the ordered prompts; each Enter records one phase marker.")
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

    accepted = False if interrupted else bool(result["accepted"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    trajectory_name = f"{stamp}_bolt_preinsert_pilot_{args.trajectory_index:04d}_{slot}_{grasp}"
    trajectory_dir = output_root / trajectory_name
    metadata = {
        "trajectory_name": trajectory_name,
        "task_name": config["task_name"],
        "language_prompt": config["language_prompt"],
        "capture_scope": config["capture_scope"],
        "collection_mode": "kinesthetic_pilot_read_only",
        "robot_ip": config["robot_ip"],
        "screw_slot": slot,
        "screw_grasp_style": grasp,
        "bracket_installed_at_start": True,
        "trajectory_index": args.trajectory_index,
        "notes": args.notes,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "calibration_path": str(calibration_path) if calibration_path.exists() else None,
        "calibration_sha256": calibration_hash,
        "arm_command_available": False,
        "gripper_command_available": False,
        "gripper_state_recorded": False,
        "gripper_event_mode": "ordered_semantic_markers",
        "cameras_recorded": False,
        "positional_replay_endpoint": "PREINSERT",
        "contact_or_threading_included": False,
        "interrupted": interrupted,
    }
    recorder.write(trajectory_dir, metadata, accepted=accepted)
    print(f"[capture] wrote {len(recorder.samples)} samples to {trajectory_dir}")
    print(f"[capture] accepted_by_operator={accepted} sample_errors={len(recorder.errors)}")
    print("[capture] replay/contact/threading remain disabled; validate offline next")
    print("Use Pilot controls to retract safely and return the bolt after the program exits.")


if __name__ == "__main__":
    main()
