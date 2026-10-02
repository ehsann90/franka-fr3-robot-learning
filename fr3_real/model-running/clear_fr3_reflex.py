#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/clear_fr3_reflex.py
"""Clear a libfranka reflex so the streamer can take control again.

fr3_joint_velocity_streamer never calls automaticErrorRecovery, so after a
reflex (for example joint_motion_generator_velocity_discontinuity) the robot
stays latched and the next streamer start fails. This issues one recovery and
reports what the robot says before and after. It commands no motion.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot_ip", default="172.16.0.2")
    args = ap.parse_args()

    from franky import Robot

    robot = Robot(args.robot_ip)
    print(f"[RECOVER] has_errors={robot.has_errors} mode={robot.state.robot_mode}")
    if not robot.has_errors:
        print("[RECOVER] nothing latched; the streamer can start as-is.")
        return
    ok = robot.recover_from_errors()
    print(f"[RECOVER] recover_from_errors()={ok} has_errors={robot.has_errors} mode={robot.state.robot_mode}")
    if robot.has_errors:
        raise SystemExit("Still latched. Check the Desk UI -- this usually means the E-stop or the "
                         "guiding button is engaged, or the robot needs unlocking.")
    print("[RECOVER] cleared. The gripper was untouched, so a held bracket is still held.")


if __name__ == "__main__":
    main()
