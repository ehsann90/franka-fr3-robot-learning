#!/usr/bin/env python3
"""Validate the provisional bolt contact detector without commanding hardware."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.contact_detection import first_sustained_crossing  # noqa: E402
from expert_demo.trajectory_recording import PilotedTrajectoryRecorder  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_contact_v1.json"


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--record", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    try:
        config = json.loads(config_path.read_text())
        detector = config["contact_detection"]
        validation = config["read_only_validation"]
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise SystemExit(f"cannot read contact configuration: {error}") from error
    if detector.get("status") != "PROVISIONAL_OPERATOR_APPROVED":
        raise SystemExit("contact threshold has not been explicitly approved")
    if validation.get("enabled") is not True:
        raise SystemExit("read-only contact validation is disabled")
    if config["automatic_contact_motion"].get("enabled") is not False:
        raise SystemExit("automatic contact motion must remain disabled")
    if config["rotation_threading"].get("enabled") is not False:
        raise SystemExit("rotation/threading must remain disabled")

    sample_hz = float(detector["sample_hz"])
    threshold_delta = float(detector["threshold_relative_to_live_baseline_N"])
    persistence = int(detector["persistence_samples"])
    print("=== BOLT READ-ONLY CONTACT DETECTOR VALIDATION ===")
    print(f"mode:                    {'DRY_RUN' if args.dry_run else 'RECORD_READ_ONLY'}")
    print(f"robot IP:                {config['robot_ip']}")
    print("required start:          bolt held upright at unloaded THREAD_READY")
    print("downward axis/signal:    +Kz / K_F_ext_hat_K[2]")
    print(f"provisional detector:    +{threshold_delta:.3f} N for {persistence} samples")
    print(f"sample rate/persistence: {sample_hz:.1f} Hz / {persistence / sample_hz:.3f} s")
    print("arm/Hand commands:       DISABLED")
    print("automatic contact:       DISABLED")
    print("rotation/threading:      DISABLED")
    if args.dry_run:
        print("\nDry-run complete. No robot or Hand connection was attempted.")
        return

    print("\nUse supported Desk/Pilot hand guiding only.")
    print("Make one very gentle straight-down contact, then immediately retract upward.")
    print("Do not rotate, wait for force buildup, or leave the printed assembly preloaded.")
    input("Press Enter while stationary and unloaded to connect/start (Ctrl+C cancels): ")

    from franky import Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    recorder = PilotedTrajectoryRecorder(robot, None, robot_hz=sample_hz, gripper_hz=1.0)
    preflight = recorder.preflight()
    xyz = preflight["translation_xyz_m"]
    print(f"[monitor] preflight xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})")
    recorder.start(start_gripper_thread=False)
    try:
        recorder.add_event("live_baseline_started")
        duration = float(validation["baseline_duration_s"])
        print(f"[baseline] hands off for {duration:.1f} s")
        time.sleep(duration)
        recorder.add_event("live_baseline_complete")
        baseline_samples = [
            sample["K_F_ext_hat_K"][2]
            for sample in recorder.samples
            if "K_F_ext_hat_K" in sample
        ]
        if len(baseline_samples) < 2:
            raise RuntimeError("insufficient live baseline samples")
        baseline = float(np.mean(baseline_samples))
        baseline_std = float(np.std(baseline_samples))
        print(f"[baseline] F_Kz={baseline:+.6f} N std={baseline_std:.6f} N")
        input(
            "Press Enter when ready. Move to the robot during the setup delay; "
            "then make one gentle contact and immediately retract: "
        )
        delay = float(validation["operator_setup_delay_s"])
        print(f"[monitor] starts in {delay:.1f} s; terminal bell marks the start")
        time.sleep(delay)
        print("\a[monitor] ACTIVE — gently contact, then immediately retract")
        recorder.add_event("contact_monitor_started", {"live_baseline_F_Kz_N": baseline})
        monitor_start_ns = time.perf_counter_ns()
        processed = len(recorder.samples)
        observed: list[float] = []
        trigger_sample = None
        timeout_s = float(validation["monitor_timeout_s"])
        while (time.perf_counter_ns() - monitor_start_ns) / 1e9 < timeout_s:
            current = recorder.samples[processed:]
            processed += len(current)
            observed.extend(
                sample["K_F_ext_hat_K"][2]
                for sample in current
                if "K_F_ext_hat_K" in sample
            )
            crossing = first_sustained_crossing(
                observed, baseline, threshold_delta, persistence
            )
            if crossing is not None:
                trigger_sample = crossing
                recorder.add_event(
                    "provisional_contact_detected",
                    {
                        "threshold_delta_N": threshold_delta,
                        "persistence_samples": persistence,
                    },
                )
                print("\a[DETECTED] provisional contact threshold crossed; remain/retract unloaded")
                time.sleep(float(validation["post_trigger_capture_s"]))
                break
            time.sleep(0.01)
        recorder.add_event("contact_monitor_stopped", {"triggered": trigger_sample is not None})
    finally:
        recorder.stop()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    name = f"{stamp}_bolt_contact_detector_trial"
    capture_dir = project_path(validation["output_dir"]) / name
    accepted = trigger_sample is not None and len(recorder.errors) == 0
    recorder.write(
        capture_dir,
        {
            "trajectory_name": name,
            "capture_type": "read_only_contact_detector_validation",
            "robot_ip": config["robot_ip"],
            "axis": "+Kz",
            "live_baseline_F_Kz_N": baseline,
            "live_baseline_std_N": baseline_std,
            "threshold_delta_N": threshold_delta,
            "persistence_samples": persistence,
            "triggered": trigger_sample is not None,
            "arm_commands_enabled": False,
            "gripper_commands_enabled": False,
            "automatic_contact_motion_enabled": False,
            "rotation_threading_enabled": False,
            "notes": args.notes,
        },
        accepted=accepted,
    )
    peak_delta = max(observed, default=baseline) - baseline
    result = {
        "schema_version": 1,
        "triggered": trigger_sample is not None,
        "baseline_F_Kz_N": baseline,
        "baseline_std_N": baseline_std,
        "threshold_delta_N": threshold_delta,
        "persistence_samples": persistence,
        "peak_observed_delta_N": float(peak_delta),
        "automatic_motion_enabled": False,
    }
    (capture_dir / "detector_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[result] triggered:       {result['triggered']}")
    print(f"[result] peak delta:      {result['peak_observed_delta_N']:+.6f} N")
    print(f"[saved] {capture_dir}")
    print("No robot/Hand stop or motion command was available; confirm the bolt is unloaded.")
    if not accepted:
        raise SystemExit("[result] provisional detector did not pass this read-only trial")


if __name__ == "__main__":
    main()
