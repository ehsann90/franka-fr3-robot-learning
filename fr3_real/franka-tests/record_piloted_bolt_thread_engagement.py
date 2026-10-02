#!/usr/bin/env python3
"""Read-only Pilot capture of the smallest unambiguous thread engagement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.thread_engagement import summarize_thread_engagement  # noqa: E402
from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_threading_v1.json"


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
    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        capture = config["manual_engagement_capture"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise SystemExit(f"cannot read thread-engagement configuration: {error}") from error
    if capture.get("enabled") is not True:
        raise SystemExit("manual engagement capture is disabled")
    if capture.get("automatic_motion_enabled") is not False:
        raise SystemExit("automatic motion must remain disabled")
    if capture.get("automatic_rotation_enabled") is not False:
        raise SystemExit("automatic rotation must remain disabled")
    if config["automatic_thread_engagement"].get("enabled") is not False:
        raise SystemExit("automatic thread engagement must remain disabled")
    if args.trajectory_index < 1:
        raise SystemExit("--trajectory-index must be at least 1")

    output_root = project_path(capture["output_dir"])
    print("=== PILOTED INITIAL THREAD-ENGAGEMENT CALIBRATION ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:                {config['robot_ip']}")
    print("required start:          bolt upright, head-held, self-supported, and not engaged")
    print("prewind:                 release; empty-wrist turn opposite tightening; regrasp")
    print("operator motion:         repeated tighten -> release -> empty-wrist reset -> regrasp")
    print("endpoint:                first unambiguous engagement/axial advancement")
    print("arm/Hand commands:       DISABLED")
    print("automatic rotation:      DISABLED")
    print("threshold calculation:   DISABLED")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return

    print("\nUse supported Desk/Pilot hand guiding only.")
    print("Do not rotate the supported bolt during wrist prewind: release it first.")
    print("Prewind approximately 180 degrees opposite tightening, only within known safe limits.")
    print("Use each known-safe tightening wrist window; never turn through a joint/cable limit.")
    print("Repeat as many manual ratchet cycles as needed, up to the configured capture limit.")
    input("Press Enter while stationary and unloaded to connect/start (Ctrl+C cancels): ")

    from franky import Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    recorder = PilotedTrajectoryRecorder(
        robot,
        None,
        robot_hz=float(capture["robot_sample_hz"]),
        gripper_hz=1.0,
    )
    preflight = recorder.preflight()
    xyz = preflight["translation_xyz_m"]
    print(f"[capture] preflight xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})")
    recorder.start(start_gripper_thread=False)
    accepted = False
    interrupted = False
    try:
        recorder.add_event("wrist_prewind_started")
        input(
            "In Pilot: keep the bolt self-supported, OPEN the Hand, move clear without "
            "side load, rotate the EMPTY wrist approximately 180 degrees opposite the "
            "tightening direction within cable/joint limits, reapproach, and regrasp the "
            "head securely. Keep the bolt upright and unloaded. Press Enter when PREWOUND "
            "and stationary (Ctrl+C rejects): "
        )
        recorder.add_event("wrist_prewind_complete")
        recorder.add_event("unloaded_baseline_started")
        print("[baseline] hands off for 3 seconds")
        time.sleep(3.0)
        recorder.add_event("unloaded_baseline_complete")
        max_cycles = int(capture["maximum_manual_ratchet_cycles"])
        for cycle in range(1, max_cycles + 1):
            input(f"Press Enter when ready to begin tightening drive {cycle}: ")
            recorder.add_event("ratchet_drive_started", {"cycle": cycle})
            response = input(
                "In Pilot: apply only gentle axial contact and rotate slowly through the known-safe "
                "tightening window; stop before the wrist/cable limit. Press Enter if another "
                "ratchet is required, type e if clear engagement/axial advancement occurred, or "
                "type x to reject: "
            ).strip().lower()
            if response not in ("", "e"):
                recorder.add_event(
                    "ratchet_drive_complete", {"cycle": cycle, "outcome": "rejected"}
                )
                break
            outcome = "engaged" if response == "e" else "continue"
            recorder.add_event("ratchet_drive_complete", {"cycle": cycle, "outcome": outcome})
            if outcome == "engaged":
                accepted = True
                recorder.add_event("engaged_stationary_started")
                print("[engaged] hands off for 3 seconds; do not continue turning")
                time.sleep(3.0)
                recorder.add_event("engaged_stationary_complete")
                break
            recorder.add_event("ratchet_reset_started", {"cycle": cycle})
            input(
                "Keep the bolt supported. OPEN the Hand, move clear without side load, rewind the "
                "EMPTY wrist opposite tightening, reapproach, and regrasp the head securely. "
                "Press Enter when stationary and ready for the next drive (Ctrl+C rejects): "
            )
            recorder.add_event("ratchet_reset_complete", {"cycle": cycle})
        else:
            print(f"[capture] no confirmed engagement within {max_cycles} manual ratchet cycles")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[capture] interrupted; stopping read-only capture and retaining it as rejected")
    finally:
        recorder.stop()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_bolt_thread_engagement_pilot_{args.trajectory_index:04d}"
    output = output_root / name
    recorder.write(
        output,
        {
            "trajectory_name": name,
            "capture_type": "piloted_initial_thread_engagement_calibration",
            "task_name": config["task_name"],
            "language_prompt": config["language_prompt"],
            "robot_ip": config["robot_ip"],
            "endpoint": capture["endpoint"],
            "arm_commands_enabled": False,
            "gripper_connection_enabled": False,
            "gripper_commands_enabled": False,
            "automatic_rotation_enabled": False,
            "automatic_threading_enabled": False,
            "interrupted": interrupted,
            "notes": args.notes,
        },
        accepted=accepted and not interrupted and len(recorder.errors) == 0,
    )
    if not accepted or interrupted:
        print(f"[saved] rejected raw capture: {output}")
        raise SystemExit("[capture] engagement was not confirmed; raw capture retained as rejected")
    try:
        summary = summarize_thread_engagement(recorder.samples, recorder.events)
    except ValueError as error:
        raise SystemExit(f"[capture] raw data saved but summary failed: {error}") from error
    (output / "thread_engagement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[saved] accepted/errors: {accepted} / {len(recorder.errors)}")
    print(
        f"[saved] prewind Kz/q7:  "
        f"{summary['wrist_prewind_rotation_about_positive_Kz_deg']:+.3f} deg / "
        f"{summary['wrist_prewind_q7_delta_rad']:+.6f} rad"
    )
    print(f"[saved] Kz rotation:     {summary['signed_rotation_about_positive_Kz_deg']:+.3f} deg")
    print(f"[saved] Kz advancement:  {summary['axial_progress_along_positive_Kz_m'] * 1000:+.3f} mm")
    print(f"[saved] ratchet cycles:  {summary['ratchet_cycle_count']}")
    print(f"[saved] engaged dFz:     {summary['engaged_axial_force_delta_N']:+.6f} N")
    print(f"[saved] engaged dTz:     {summary['engaged_axial_torque_delta_Nm']:+.6f} Nm")
    print(f"[saved] {output}")
    print("No automatic threshold was selected and no robot/Hand command was issued.")
    print("Use Pilot to reverse/back out conservatively and retract to unloaded THREAD_READY.")


if __name__ == "__main__":
    main()
