#!/usr/bin/env python3
# Usage: from fr3_real/, run: python franka-tests/hover_above_basket_center.py --robot-ip 172.16.0.2 --run
"""Move the FR3 to the calibrated basket center and close the gripper.

This setup-only utility reads ``pad_center`` and ``basket_rim`` from the
calibration file, rises before horizontal travel, then holds above the basket
while it is positioned. After confirmation, it descends to a configurable
height above the rim and can close the gripper. It does not collect data or
run the VLA. Use it when repositioning the physical basket before an
evaluation.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from franka_motion import MotionPlanner, clear_robot_errors, configure_collision_behavior
from repo_paths import CONFIGS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--points", type=Path, default=CONFIGS_DIR / "probed_points.json")
    parser.add_argument(
        "--hover-above-rim",
        type=float,
        default=0.12,
        help="Vertical clearance above basket_rim in meters.",
    )
    parser.add_argument(
        "--travel-extra-clearance",
        type=float,
        default=0.05,
        help="Additional vertical clearance used before horizontal travel in meters.",
    )
    parser.add_argument(
        "--final-above-rim",
        type=float,
        default=0.01,
        help="Final tool-center height above basket_rim in meters.",
    )
    parser.add_argument(
        "--close-gripper",
        action="store_true",
        help="Close the gripper after descending to the final basket-center height.",
    )
    parser.add_argument("--dynamics-factor", type=float, default=0.085)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if not args.run:
        raise SystemExit("Refusing robot motion without --run")
    if args.hover_above_rim <= 0 or args.final_above_rim <= 0 or args.travel_extra_clearance < 0:
        raise SystemExit("Hover clearances must be positive and travel clearance cannot be negative")

    points = json.loads(args.points.read_text())
    missing = [key for key in ("pad_center", "basket_rim") if key not in points]
    if missing:
        raise SystemExit(f"{args.points} missing: {missing}")
    center = np.asarray(points["pad_center"], dtype=float)
    rim = np.asarray(points["basket_rim"], dtype=float)
    hover_z = float(rim[2] + args.hover_above_rim)
    final_z = float(rim[2] + args.final_above_rim)
    travel_z = float(hover_z + args.travel_extra_clearance)

    print("[BASKET] Setup-only calibrated basket placement hover.")
    print(
        "[BASKET] center="
        f"[{center[0]:+.4f}, {center[1]:+.4f}, {center[2]:+.4f}] "
        f"rim_z={rim[2]:+.4f} hover_z={hover_z:+.4f}"
        f" final_z={final_z:+.4f}"
    )
    confirmation = input(
        "Workspace and path clear, basket absent or clear of the path, E-stop reachable. "
        "Type HOVER to move above the calibrated basket center: "
    )
    if confirmation.strip().upper() != "HOVER":
        print("[BASKET] aborted before connection.")
        return

    from franky import Robot

    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.dynamics_factor

    def on_sigint(_signum, _frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n[KILLSWITCH] Ctrl+C received; calling robot.stop() now.")
        robot.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)
    try:
        clear_robot_errors(robot)
        configure_collision_behavior(robot)
        motion = MotionPlanner(
            robot,
            dynamics_factor=args.dynamics_factor,
            max_step=args.max_step,
            travel_z_floor=travel_z,
        )
        motion.travel_to(center, hover_z, "basket center hover")
        actual = motion.current_xyz()
        print(
            "[BASKET] holding above center; place the basket directly below the tool. "
            f"actual_tcp=[{actual[0]:+.4f}, {actual[1]:+.4f}, {actual[2]:+.4f}]"
        )
        input("Press Enter only after basket placement is complete; the arm will remain in place: ")
        motion.move_in_steps(
            np.asarray([center[0], center[1], final_z], dtype=float),
            "basket rim hover",
        )
        actual = motion.current_xyz()
        print(
            "[BASKET] at calibrated center 1 cm above the rim: "
            f"actual_tcp=[{actual[0]:+.4f}, {actual[1]:+.4f}, {actual[2]:+.4f}]"
        )
        if args.close_gripper:
            from franky import Gripper

            gripper = Gripper(args.robot_ip)
            accepted = gripper.grasp(0.04, 0.02, 15.0, 0.01, 0.01)
            state = gripper.state
            print(
                "[BASKET] close command sent at basket center; "
                f"accepted={accepted} width={state.width:.4f} m"
            )
        input("Press Enter after confirming the final pose; the arm will remain in place: ")
    finally:
        # Holding position requires no additional command; avoid an automatic retreat near a new basket.
        del robot


if __name__ == "__main__":
    main()
