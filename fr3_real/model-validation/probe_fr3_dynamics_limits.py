#!/usr/bin/env python3
"""Read the FR3's live velocity/acceleration/jerk limits and report the
relative-dynamics factors that reproduce the training executor's caps.

Background. Every one of the 50 bracket demonstrations was executed by
fr3_retimed_joint_trajectory_bracket_highcap_v2/v3, which were compiled with
explicit caps of 1.90 rad/s, 10.0-14.0 rad/s^2 and 260.0 rad/s^3. The eval
runner instead set a single scalar relative_dynamics_factor=0.9, which scales
velocity AND acceleration AND jerk together -- so it reproduced the training
velocity while permitting roughly an order of magnitude more jerk than any
training motion ever used. That is what makes the arm slam between chunks.

franky exposes RelativeDynamicsFactor(velocity, acceleration, jerk), so the
three can be set independently. This prints the exact values to put in
configs/bracket_policy_eval_v1.json, derived from the limits this robot
actually reports rather than from datasheet numbers.
"""

from __future__ import annotations

import argparse
import numpy as np

# The caps the training trajectories were compiled with (streaming/CMakeLists.txt).
TRAIN_VELOCITY_CAP = 1.90
TRAIN_ACCELERATION_CAP = 14.0
TRAIN_JERK_CAP = 260.0


def as_array(limit) -> np.ndarray:
    for attr in ("max", "value", "upper"):
        if hasattr(limit, attr):
            return np.asarray(getattr(limit, attr), dtype=float).reshape(-1)
    return np.asarray(limit, dtype=float).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot_ip", default="172.16.0.2")
    args = ap.parse_args()

    from franky import Robot

    robot = Robot(args.robot_ip)
    limits = {
        "velocity  (rad/s)  ": (as_array(robot.joint_velocity_limit), TRAIN_VELOCITY_CAP),
        "acceleration (r/s2)": (as_array(robot.joint_acceleration_limit), TRAIN_ACCELERATION_CAP),
        "jerk        (r/s3) ": (as_array(robot.joint_jerk_limit), TRAIN_JERK_CAP),
    }

    print("=" * 78)
    print("FR3 LIVE DYNAMICS LIMITS vs THE CAPS THE TRAINING DEMOS WERE EXECUTED WITH")
    print("=" * 78)
    factors = {}
    for name, (live, train_cap) in limits.items():
        binding = float(live.min())
        factor = train_cap / binding
        factors[name.split()[0]] = factor
        print(f"{name} live={np.array2string(live, precision=3)}")
        print(f"{'':19} binding (min) = {binding:.4f}")
        print(f"{'':19} training cap  = {train_cap:.4f}   -> factor {factor:.4f}"
              f"{'   [CLAMPED to 1.0]' if factor > 1.0 else ''}")
        print()

    v = min(factors["velocity"], 1.0)
    a = min(factors["acceleration"], 1.0)
    j = min(factors["jerk"], 1.0)

    print("-" * 78)
    print("Put this in configs/bracket_policy_eval_v1.json under \"motion\":")
    print()
    print('    "relative_dynamics_factor": {')
    print(f'      "velocity": {v:.4f},')
    print(f'      "acceleration": {a:.4f},')
    print(f'      "jerk": {j:.4f}')
    print('    },')
    print()
    old = 0.9
    print(f"For comparison, the previous scalar {old} permitted:")
    for name, (live, train_cap) in limits.items():
        print(f"  {name} {old * float(live.min()):9.3f}   vs training {train_cap:8.3f}"
              f"   ({old * float(live.min()) / train_cap:5.1f}x)")
    print()
    print("The velocity ratio is ~1x by design; the jerk ratio is the bug.")


if __name__ == "__main__":
    main()
