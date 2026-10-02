#!/usr/bin/env python3
"""Dry-run or read-only preview of the first guarded automatic thread drive."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.thread_drive_preview import (  # noqa: E402
    half_cosine_duration,
    preview_blockers,
)


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_thread_drive_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--read-only-preview", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.expanduser().resolve().read_text())
    limits = json.loads(project_path(config["ratchet_limit_calibration"]).read_text())
    dynamic = json.loads(project_path(config["dynamic_model_calibration"]).read_text())
    head = json.loads(project_path(config["thread_ready_head_width_calibration"]).read_text())
    reference = np.load(
        project_path(config["thread_ready_reference_capture"]) / "trajectory_arrays.npz"
    )
    knots = np.asarray(dynamic["knots"], dtype=np.float64)
    start_q7 = float(config["commissioned_start_q7_rad"])
    angle = float(config["drive_angle_rad"])
    nominal_target = start_q7 + angle
    computed_duration = half_cosine_duration(
        angle, float(config["planned_peak_q7_speed_rad_s"])
    )

    print("=== FIRST GUARDED AUTOMATIC BOLT DRIVE PREVIEW ===")
    print(f"mode:                     {'DRY_RUN' if args.dry_run else 'READ_ONLY_PREVIEW'}")
    print(f"robot IP:                 {config['robot_ip']}")
    print(f"scope:                    {config['scope']}")
    print(f"planned q7 interval:      {start_q7:+.6f} .. {nominal_target:+.6f} rad")
    print(f"planned rotation:         +{float(config['drive_angle_deg']):.1f} deg")
    print(f"profile/peak speed:       {config['planned_profile']} / {float(config['planned_peak_q7_speed_rad_s']):.3f} rad/s")
    print(f"planned duration:         {computed_duration:.3f} s")
    print(f"hard residual cutoff:     {float(config['approved_hard_negative_residual_Tz_Nm']):+.6f} Nm")
    print(f"hard axial cutoff:        +/-{float(config['approved_hard_absolute_axial_delta_N']):.3f} N")
    print(f"axial command:            0 -> {float(config['planned_axial_command_ceiling_N']):.3f} N at {float(config['planned_axial_command_slew_N_s']):.3f} N/s")
    print(f"combined-motion approval: {config['axial_command_during_rotation_status']}")
    print("automatic ratchet/seated: DISABLED")
    print("camera/Hand commands:     DISABLED")
    print(f"execution enabled:        {config['execution_enabled']}")

    static_blockers: list[str] = []
    if not np.isclose(computed_duration, float(config["planned_duration_s"]), atol=1e-9):
        static_blockers.append("configured drive duration does not match its motion profile")
    static_blockers.extend(
        preview_blockers(
            live_q7_rad=start_q7,
            start_q7_rad=start_q7,
            start_tolerance_rad=float(config["start_q7_tolerance_rad"]),
            target_q7_rad=nominal_target,
            cable_interval_rad=(
                limits["summary"]["q7_prewind_rad"],
                limits["summary"]["q7_forward_rad"],
            ),
            # The established free-spin validator permits 0.02 rad at the
            # low-speed endpoint before the first fitted knot. The planned
            # target remains strictly inside the final measured knot.
            dynamic_model_interval_rad=(knots[0, 0] - 0.02, knots[-1, 0]),
        )
    )
    if config.get("execution_enabled") is not False:
        static_blockers.append("preview configuration unexpectedly enables execution")

    if args.dry_run:
        if static_blockers:
            print("\nConfiguration blockers:")
            for blocker in dict.fromkeys(static_blockers):
                print(f"  - {blocker}")
        else:
            print("\nDry-run result: plan is internally consistent.")
        print("No robot or Hand connection was attempted.")
        return

    input("Press Enter to connect read-only with the bolt held at THREAD_READY (Ctrl+C cancels): ")
    from franky import Gripper, Robot

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    state = robot.state
    q = np.asarray(state.q, dtype=np.float64)
    T = np.asarray(state.O_T_EE.matrix, dtype=np.float64)
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    del gripper

    live_target = float(q[6] + angle)
    blockers = list(static_blockers)
    blockers.extend(
        preview_blockers(
            live_q7_rad=float(q[6]),
            start_q7_rad=start_q7,
            start_tolerance_rad=float(config["start_q7_tolerance_rad"]),
            target_q7_rad=live_target,
            cable_interval_rad=(
                limits["summary"]["q7_prewind_rad"],
                limits["summary"]["q7_forward_rad"],
            ),
            dynamic_model_interval_rad=(knots[0, 0] - 0.02, knots[-1, 0]),
        )
    )

    margin = float(config["gripper_width_comparison_margin_m"])
    accepted_min = float(head["minimum_width_m"]) - margin
    accepted_max = float(head["maximum_width_m"]) + margin
    if not grasped or not accepted_min <= width <= accepted_max:
        blockers.append("Hand state is inconsistent with the calibrated head grasp")

    reference_T = np.asarray(reference["T_base_ee"], dtype=np.float64).mean(axis=0)
    axis = T[:3, 2] / np.linalg.norm(T[:3, 2])
    error = T[:3, 3] - reference_T[:3, 3]
    axial_error = float(axis @ error)
    lateral_error = float(np.linalg.norm(error - axis * axial_error))
    tilt = float(np.degrees(np.arccos(np.clip(axis @ reference_T[:3, 2], -1.0, 1.0))))
    if abs(axial_error) > float(config["maximum_start_axial_error_m"]):
        blockers.append("live axial position is too far from THREAD_READY")
    if lateral_error > float(config["maximum_start_lateral_error_m"]):
        blockers.append("live lateral position is too far from THREAD_READY")
    if tilt > float(config["maximum_start_tool_axis_tilt_deg"]):
        blockers.append("live tool axis is too tilted from THREAD_READY")
    bounds_min = np.asarray(config["workspace_bounds_m"]["min_xyz"], dtype=np.float64)
    bounds_max = np.asarray(config["workspace_bounds_m"]["max_xyz"], dtype=np.float64)
    if np.any(T[:3, 3] < bounds_min) or np.any(T[:3, 3] > bounds_max):
        blockers.append("live TCP is outside the local workspace")

    baseline = []
    deadline = time.perf_counter() + float(config["baseline_duration_s"])
    while time.perf_counter() < deadline:
        sample = robot.state
        baseline.append((sample.K_F_ext_hat_K[2], np.max(np.abs(sample.dq))))
        time.sleep(1.0 / float(config["baseline_sample_hz"]))
    baseline = np.asarray(baseline, dtype=np.float64)
    fz_std = float(np.std(baseline[:, 0]))
    max_dq = float(np.max(baseline[:, 1]))
    if fz_std > float(config["maximum_baseline_Fz_std_N"]):
        blockers.append("live axial-force baseline is not stationary")
    if max_dq > float(config["maximum_baseline_abs_dq_rad_s"]):
        blockers.append("arm moved during the read-only baseline")

    print(f"\nlive/target q7:           {q[6]:+.6f} / {live_target:+.6f} rad")
    print(f"Hand width/grasped:       {width:.6f} m / {grasped}")
    print(f"accepted head width:      {accepted_min:.6f} .. {accepted_max:.6f} m")
    print(f"THREAD_READY error:       axial {axial_error*1000:+.3f} mm; lateral {lateral_error*1000:.3f} mm; tilt {tilt:.3f} deg")
    print(f"baseline F_Kz std:        {fz_std:.6f} N")
    print(f"maximum baseline |dq|:    {max_dq:.6f} rad/s")
    if blockers:
        print("\nPreview blockers:")
        for blocker in dict.fromkeys(blockers):
            print(f"  - {blocker}")
    else:
        print("\nRead-only geometry and calibration checks passed.")
    print("Execution remains disabled. No arm or Hand command was issued.")


if __name__ == "__main__":
    main()
