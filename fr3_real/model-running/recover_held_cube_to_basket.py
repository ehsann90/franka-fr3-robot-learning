#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/recover_held_cube_to_basket.py --robot-ip 172.16.0.2 --run
"""Recover a cube already held by the stopped FR3 policy streamer to the basket.

Use this only after a transport-phase infrastructure failure when the gripper
still holds the cube. The tool verifies that an object is held, lifts through a
safe Cartesian transit, releases only at the calibrated basket center, and
leaves the arm above the basket. It never invokes policy inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from franky import Gripper, Robot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from franka_motion import MotionPlanner, clear_robot_errors, configure_collision_behavior
from repo_paths import CONFIGS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--points", type=Path, default=CONFIGS_DIR / "probed_points.json")
    parser.add_argument("--dynamics-factor", type=float, default=0.085)
    parser.add_argument("--cube-height", type=float, default=0.04)
    parser.add_argument("--transport-clearance", type=float, default=0.05)
    parser.add_argument("--open-speed", type=float, default=0.05)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("Refusing robot motion without --run")

    points = json.loads(args.points.read_text())
    pad = np.asarray(points["pad_center"], dtype=float)
    rim = np.asarray(points["basket_rim"], dtype=float)
    release_xyz = np.array([pad[0], pad[1], pad[2] + args.cube_height / 2])

    print("[RECOVER] Held-object recovery only: no policy inference or camera use.")
    if input("Confirm the arm is stopped and the cube is visibly held. Type RECOVER to continue: ").strip() != "RECOVER":
        raise SystemExit("Cancelled")

    robot = Robot(args.robot_ip)
    robot.relative_dynamics_factor = args.dynamics_factor
    gripper = Gripper(args.robot_ip)
    clear_robot_errors(robot)
    configure_collision_behavior(robot)
    if not gripper.is_grasped:
        raise SystemExit("[RECOVER] No held object detected; refusing to move or open the gripper.")

    current = np.asarray(robot.current_pose.end_effector_pose.translation, dtype=float)
    transit_z = max(float(current[2]), float(rim[2]) + args.transport_clearance) + 0.05
    if transit_z > 0.65:
        raise SystemExit(f"[RECOVER] Required transit height {transit_z:.3f} exceeds guarded workspace.")
    print(
        "[RECOVER] current_tcp="
        f"{np.array2string(current, precision=3)} basket_release="
        f"{np.array2string(release_xyz, precision=3)} transit_z={transit_z:.3f}"
    )

    motion = MotionPlanner(
        robot,
        dynamics_factor=args.dynamics_factor,
        max_step=0.03,
        travel_z_floor=transit_z,
        settle_time=0.30,
    )
    motion.travel_to(pad, transit_z, "held recovery basket hover")
    motion.move_in_steps(release_xyz, "held recovery basket descend")
    if not gripper.is_grasped:
        raise RuntimeError("[RECOVER] Held-object state was lost before basket release")
    gripper.open(args.open_speed)
    motion.move_in_steps(np.array([pad[0], pad[1], transit_z]), "held recovery basket retreat")
    print("[RECOVER] Cube released in basket; arm is parked above the basket.")


if __name__ == "__main__":
    main()
