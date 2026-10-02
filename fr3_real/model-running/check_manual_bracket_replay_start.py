#!/usr/bin/env python3
"""Read-only comparison of live FR3 state with a manual bracket replay stage
start.

Mirrors check_piloted_bracket_replay_start.py's read-only pattern (single
robot.state read or a live-monitored hand-guiding loop, no motion or gripper
command ever issued), but builds its stage plan from a manually recorded
episode's IK candidate via prepare_manual_bracket_replay.build_manual_stage_plan
instead of the piloted trajectory.jsonl format.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_manual_bracket_replay import (  # noqa: E402
    STAGE_MANUAL_EVENTS,
    build_manual_stage_plan,
)

# Reference only: matches the start_joint_tolerance_rad used by the piloted
# stage configs. Not yet calibrated specifically for a manual-episode
# candidate, so results are reported against it but labeled as a reference.
REFERENCE_START_TOLERANCE_RAD = 0.005


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_MANUAL_EVENTS), default="approach_open")
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--command-rate-hz", type=float, default=15.0)
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuously display joint error while hand-guiding; press Enter to stop.",
    )
    args = parser.parse_args()

    candidate_path = args.candidate.expanduser().resolve()
    try:
        plan = build_manual_stage_plan(
            candidate_path, args.stage, command_rate_hz=args.command_rate_hz
        )
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"[start check] invalid input: {error}") from error

    print("=== MANUAL BRACKET REPLAY START CHECK ===")
    print("mode:                 READ_ONLY")
    print(f"robot IP:             {args.robot_ip}")
    print(f"candidate:            {candidate_path}")
    print(f"stage:                {plan.name}")
    print(f"expected start q:     {np.array2string(plan.expected_start_q, precision=5)}")
    print(f"expected start xyz:   {np.array2string(plan.xyz_targets[0], precision=5)}")
    print(f"reference tolerance:  {REFERENCE_START_TOLERANCE_RAD} rad (not yet calibrated "
          "for this candidate)")
    print("arm commands:         DISABLED BY DESIGN")
    print("gripper connection:   DISABLED BY DESIGN")
    print("gripper commands:     DISABLED BY DESIGN")
    input(
        "Press Enter to connect and begin read-only monitoring (Ctrl+C cancels): "
        if args.monitor
        else "Press Enter to connect and read one live arm state (Ctrl+C cancels): "
    )

    from franky import Robot  # noqa: PLC0415

    robot = Robot(args.robot_ip)
    if args.monitor:
        stop_requested = threading.Event()

        def wait_for_stop() -> None:
            input("Hand-guide while watching the errors below; press Enter when finished.\n")
            stop_requested.set()

        threading.Thread(target=wait_for_stop, daemon=True).start()
        state = robot.state
        while not stop_requested.is_set():
            state = robot.state
            live_q = np.asarray(state.q, dtype=np.float64)
            q_error = live_q - plan.expected_start_q
            maximum_error = float(np.max(np.abs(q_error)))
            print(
                "max|qerr|="
                f"{maximum_error:.5f} rad  qerr="
                f"{np.array2string(q_error, precision=4, suppress_small=True)}",
                end="\r",
                flush=True,
            )
            stop_requested.wait(0.25)
        print()
    else:
        state = robot.state

    live_q = np.asarray(state.q, dtype=np.float64)
    live_xyz = np.asarray(state.O_T_EE.translation, dtype=np.float64)
    q_error = live_q - plan.expected_start_q
    xyz_error = live_xyz - plan.xyz_targets[0]
    maximum_error = float(np.max(np.abs(q_error)))
    print(f"live q:               {np.array2string(live_q, precision=5)}")
    print(f"q error:              {np.array2string(q_error, precision=5)}")
    print(f"max |q error|:        {maximum_error:.6f} rad")
    print(f"live EE xyz:          {np.array2string(live_xyz, precision=5)}")
    print(f"EE xyz error:         {np.array2string(xyz_error, precision=5)}")
    if maximum_error <= REFERENCE_START_TOLERANCE_RAD:
        print("RESULT: WITHIN REFERENCE START TOLERANCE")
    else:
        print("RESULT: OUTSIDE REFERENCE START TOLERANCE")
    print("No motion or gripper command was issued.")


if __name__ == "__main__":
    main()
