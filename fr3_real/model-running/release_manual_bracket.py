#!/usr/bin/env python3
"""Guarded, Hand-only release of the bracket at the manual candidate's
recorded place pose (the stationary point between transport_place_closed and
retreat_open).

No arm motion is ever issued by this script -- it only checks that the arm
is already at the recorded place pose (put there by a prior successful
transport_place_closed --execute) and that the Hand currently reports
holding something within the configured held-width band, then on --execute
issues one franky Gripper.open() call and reports the resulting width.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_manual_bracket_replay import build_manual_stage_plan  # noqa: E402
from manual_bracket_confirmation import confirm_hand_action  # noqa: E402

DEFAULT_CONFIG = FR3_ROOT / "configs" / "bracket_manual_replay_2x_commissioning_v1.json"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--read-only-preview", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = json.loads(args.config.expanduser().resolve().read_text())
    candidate_path = project_path(config["candidate"])
    plan = build_manual_stage_plan(
        candidate_path, "transport_place_closed", command_rate_hz=float(config["command_rate_hz"])
    )
    target_q = plan.expected_stop_q
    gripper_config = config["gripper"]

    print("=== MANUAL BRACKET RELEASE (HAND-ONLY, NO ARM MOTION) ===")
    print(f"mode:                    {'EXECUTE' if args.execute else 'READ_ONLY_PREVIEW'}")
    print(f"robot IP:                {config['robot_ip']}")
    print(f"candidate:               {candidate_path}")
    print(f"release target q:        {np.array2string(target_q, precision=5)}")
    print(f"release target xyz:      {np.array2string(plan.xyz_targets[-1], precision=5)}")
    print(f"open speed:              {gripper_config['open_speed_m_s']} m/s")
    print(f"held width band:         [{gripper_config['held_width_min_m']}, {gripper_config['held_width_max_m']}] m")
    print("arm commands:            DISABLED BY DESIGN")

    static_blockers = []
    if args.execute and gripper_config.get("enabled") is not True:
        static_blockers.append("gripper.enabled is false")
    if args.execute and static_blockers:
        for blocker in static_blockers:
            print(f"  - {blocker}")
        raise SystemExit("[SAFETY] release execution is not commissioned; no robot connection attempted")

    input(
        "Confirm the arm is holding the bracket at the recorded place pose, seated on the pins. "
        "Press Enter to connect and inspect live state (Ctrl+C cancels): "
    )
    from franky import Gripper, Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
    gripper = Gripper(config["robot_ip"])
    live_q = np.asarray(robot.state.q, dtype=np.float64)
    pose_error = float(np.max(np.abs(live_q - target_q)))
    width = float(gripper.state.width)
    grasped = bool(gripper.is_grasped)
    within_band = (
        float(gripper_config["held_width_min_m"]) <= width <= float(gripper_config["held_width_max_m"])
    )

    print(f"\nlive q:                  {np.array2string(live_q, precision=5)}")
    print(f"pose error:              {pose_error:.6f} rad")
    print(f"live Hand width/grasped: {width:.6f} m / {grasped}")
    print(f"within held band:        {within_band}")

    blockers = []
    if pose_error > float(gripper_config["grasp_pose_tolerance_rad"]):
        blockers.append("arm is not at the recorded place pose")
    if not grasped or not within_band:
        blockers.append("Hand does not currently read as holding the bracket within the expected band")
    if blockers:
        print("\nBlockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if args.read_only_preview:
        print("\nPreview complete. No Hand command was issued.")
        if blockers:
            raise SystemExit(3)
        return
    if blockers:
        raise SystemExit("[SAFETY] release execution refused")

    if not confirm_hand_action("release", config):
        raise SystemExit("[release] aborted without a Hand command")

    gripper.open(float(gripper_config["open_speed_m_s"]))
    final_width = float(gripper.state.width)
    final_grasped = bool(gripper.is_grasped)
    print(f"[DONE] final width:      {final_width:.6f} m")
    print(f"[DONE] final grasped:    {final_grasped}")
    if final_grasped:
        print("[WARNING] Hand still reports grasped after opening -- visually verify the bracket released cleanly")


if __name__ == "__main__":
    main()
