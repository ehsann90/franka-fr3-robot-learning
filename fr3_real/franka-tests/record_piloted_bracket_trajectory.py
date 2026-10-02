#!/usr/bin/env python3
"""Capture a camera-free, read-only FR3 trajectory while the operator pilots it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.calibration import CalibrationStore  # noqa: E402
from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def load_config(path: Path) -> tuple[dict, Path, str]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read trajectory config {resolved}: {error}") from error
    if config.get("schema_version") != 1 or config.get("task_name") != "bracket":
        raise SystemExit("expected schema_version=1 and task_name='bracket'")
    if config.get("replay", {}).get("enabled") is not False:
        raise SystemExit("capture config must explicitly keep replay.enabled=false")
    return config, resolved, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--gripper-smoke", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trajectory-index", type=int, default=1)
    parser.add_argument("--slot", default="B1")
    parser.add_argument("--grasp", default="CENTER")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--calibration-only", action="store_true",
        help="Mark this camera-free capture as calibration only, never a training episode",
    )
    args = parser.parse_args()

    config, config_path, config_hash = load_config(args.config)
    slot = args.slot.strip().upper()
    grasp = args.grasp.strip().upper()
    if args.trajectory_index < 1:
        raise SystemExit("--trajectory-index must be at least 1")
    if slot not in config["allowed_bracket_slots"] and not (
        args.calibration_only and re.fullmatch(r"B\d+", slot)
    ):
        raise SystemExit(f"slot {slot!r} is not enabled")
    if grasp not in config["allowed_grasp_styles"]:
        raise SystemExit(f"grasp {grasp!r} is not enabled")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_path(config["output_dir"])
    )
    if args.calibration_only:
        if args.output_dir is None or output_root == FR3_ROOT / "recordings" or (
            FR3_ROOT / "recordings" in output_root.parents
        ):
            raise SystemExit("--calibration-only requires --output-dir outside the recordings root")

    print("=== FR3 PILOTED TRAJECTORY CAPTURE ===")
    selected_mode = "DRY_RUN" if args.dry_run else "GRIPPER_SMOKE_READ_ONLY" if args.gripper_smoke else "RECORD_READ_ONLY"
    print(f"mode:                 {selected_mode}")
    print(f"robot IP:             {config['robot_ip']}")
    print(f"task:                 {config['task_name']}")
    print(f"prompt:               {config['language_prompt']}")
    print(f"trajectory variation: {slot} / {grasp}")
    print(f"trajectory index:     {args.trajectory_index:03d}")
    print(f"robot sample Hz:      {config['robot_sample_hz']}")
    print(
        f"gripper sample Hz:    "
        f"{config['gripper_sample_hz'] if args.gripper_smoke else 'N/A (Hand client disabled)'}"
    )
    print(f"output root:          {output_root}")
    print(f"replay enabled:       {config['replay']['enabled']}")
    print(f"calibration only:     {args.calibration_only}")
    print("camera recording:     DISABLED (this is not a training episode)")
    print("arm commands:         DISABLED BY DESIGN")
    print("gripper commands:     DISABLED BY DESIGN")
    print("guide-mode calls:     DISABLED BY DESIGN")
    if args.record:
        print("Hand state client:    DISABLED (leaves Desk/Pilot gripper control available)")
        print("gripper events:       explicit ordered markers while arm is stationary")
    if args.dry_run:
        print("\nDry-run complete. No robot, gripper, or camera connection was attempted.")
        return

    print("\nUse supported Franka Desk/Pilot controls for all motion and gripper operation.")
    input("Press Enter to connect read-only state clients (Ctrl+C cancels): ")

    from franky import Robot  # noqa: PLC0415

    print(f"[capture] connecting read-only state clients to {config['robot_ip']} ...")
    robot = Robot(config["robot_ip"])
    gripper = None
    if args.gripper_smoke:
        from franky import Gripper  # noqa: PLC0415

        gripper = Gripper(config["robot_ip"])
    recorder = PilotedTrajectoryRecorder(
        robot,
        gripper,
        robot_hz=float(config["robot_sample_hz"]),
        gripper_hz=float(config["gripper_sample_hz"]),
    )
    preflight = recorder.preflight()
    xyz = preflight["translation_xyz_m"]
    preflight_text = f"[capture] preflight OK; xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})"
    if preflight["gripper"].get("available"):
        preflight_text += f" gripper_width={preflight['gripper']['width_m']:.4f} m"
    else:
        preflight_text += " Hand client=DISABLED"
    print(preflight_text)

    if args.gripper_smoke:
        print("[smoke] sampling arm in the background and Hand on the main thread for 2 seconds ...")
        recorder.start(start_gripper_thread=False)
        successful_reads = 0
        try:
            for index in range(20):
                if recorder.poll_gripper_once():
                    successful_reads += 1
                    latest = recorder.latest_gripper_snapshot()
                    print(f"[smoke] Hand read {index + 1:02d}/20 width={latest['width_m']:.4f} m")
                time.sleep(0.1)
        finally:
            recorder.stop()
        print(
            f"[smoke] complete: Hand reads={successful_reads}/20 "
            f"arm samples={len(recorder.samples)} errors={len(recorder.errors)}"
        )
        if successful_reads != 20 or recorder.errors:
            raise SystemExit("[smoke] FAILED; no trajectory was written and replay remains disabled")
        print("[smoke] PASS; no trajectory was written and no motion command was issued")
        return

    calibration_path = project_path(config["calibration_file"])
    calibration_hash = None
    if calibration_path.exists():
        calibration_hash = CalibrationStore(calibration_path, config["robot_ip"]).calibration_hash

    print(f"\nPlace the bracket precisely in {slot}.")
    print(f"Pilot one complete successful {grasp} trajectory:")
    print("  approach -> grasp -> lift -> transfer -> place -> release -> retreat")
    input("Press Enter immediately before beginning trajectory capture: ")

    recorder.start(start_gripper_thread=False)
    recorder.add_event("trajectory_started")
    interrupted = False
    stop_requested = threading.Event()

    marker_sequence = (
        (
            "grasp_pose_reached",
            "Pilot to the grasp pose with fingers open; keep the arm stationary, then press Enter: ",
        ),
        (
            "gripper_closed",
            "Use Desk/Pilot to close securely without moving the arm; then press Enter: ",
        ),
        (
            "place_pose_reached",
            "Pilot lift/transfer/place; keep the seated bracket and arm stationary, then press Enter: ",
        ),
        (
            "gripper_opened",
            "Use Desk/Pilot to open without moving the arm; then press Enter: ",
        ),
    )
    marker_result = {"accepted": False}

    def wait_for_markers() -> None:
        for event_name, prompt in marker_sequence:
            input(prompt)
            recorder.add_event(event_name)
        retreat_response = input(
            "Pilot the retreat and keep the arm stationary. Press Enter to stop and ACCEPT; "
            "type anything to stop and reject: "
        ).strip()
        recorder.add_event("retreat_complete")
        marker_result["accepted"] = retreat_response == ""
        stop_requested.set()

    print("[CAPTURING] Follow the ordered prompts; each Enter records one phase marker.")
    stop_input_thread = threading.Thread(target=wait_for_markers, daemon=True)
    stop_input_thread.start()
    try:
        stop_requested.wait()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[capture] interrupted; stopping and retaining the raw trajectory as rejected")
    finally:
        recorder.stop()
    if not interrupted:
        stop_input_thread.join(timeout=1.0)

    accepted = False if interrupted else bool(marker_result["accepted"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    trajectory_name = (
        f"{stamp}_bracket_pilot_{args.trajectory_index:04d}_{slot}_{grasp}"
    )
    trajectory_dir = output_root / trajectory_name
    metadata = {
        "trajectory_name": trajectory_name,
        "task_name": config["task_name"],
        "language_prompt": config["language_prompt"],
        "collection_mode": "kinesthetic_pilot_read_only",
        "robot_ip": config["robot_ip"],
        "bracket_slot": slot,
        "bracket_grasp_style": grasp,
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
        "operator_checkpoint_input": "press_enter_in_fixed_phase_order",
        "cameras_recorded": False,
        "dataset_episode": False,
        "bc_training_eligible": False,
        "calibration_only": args.calibration_only,
        "interrupted": interrupted,
    }
    recorder.write(trajectory_dir, metadata, accepted=accepted)
    print(f"[capture] wrote {len(recorder.samples)} samples to {trajectory_dir}")
    print(f"[capture] accepted_by_operator={accepted} sample_errors={len(recorder.errors)}")
    print("[capture] replay remains disabled; validate this trajectory offline next")


if __name__ == "__main__":
    main()
