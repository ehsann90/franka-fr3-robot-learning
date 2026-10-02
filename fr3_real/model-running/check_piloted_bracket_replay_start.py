#!/usr/bin/env python3
"""Read-only comparison of live FR3 state with a piloted replay stage start."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.replay_plan import ReplayPlanError, STAGE_EVENTS, build_replay_stage  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only", action="store_true", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_EVENTS), default="approach_open")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuously display joint error while hand-guiding; press Enter to stop.",
    )
    args = parser.parse_args()

    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        plan = build_replay_stage(
            args.trajectory_dir,
            args.stage,
            command_rate_hz=float(config["replay"]["command_rate_hz"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[start check] invalid input: {error}") from error

    tolerance = config["replay"].get("start_joint_tolerance_rad")
    print("=== PILOTED REPLAY START CHECK ===")
    print("mode:                 READ_ONLY")
    print(f"robot IP:             {config['robot_ip']}")
    print(f"trajectory:           {args.trajectory_dir.expanduser().resolve()}")
    print(f"stage:                {plan.name}")
    print(f"expected start q:     {np.array2string(plan.expected_start_q, precision=5)}")
    print(f"configured tolerance: {tolerance}")
    print("arm commands:         DISABLED BY DESIGN")
    print("gripper connection:   DISABLED BY DESIGN")
    print("gripper commands:     DISABLED BY DESIGN")
    input(
        "Press Enter to connect and begin read-only monitoring (Ctrl+C cancels): "
        if args.monitor
        else "Press Enter to connect and read one live arm state (Ctrl+C cancels): "
    )

    from franky import Robot  # noqa: PLC0415

    robot = Robot(config["robot_ip"])
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
    maximum_error = float(np.max(np.abs(q_error)))
    print(f"live q:               {np.array2string(live_q, precision=5)}")
    print(f"q error:              {np.array2string(q_error, precision=5)}")
    print(f"max |q error|:        {maximum_error:.6f} rad")
    print(f"live EE xyz:          {np.array2string(live_xyz, precision=5)}")
    if not isinstance(tolerance, (int, float)):
        print("RESULT: UNCOMMISSIONED — start tolerance is NOT_CALIBRATED")
    elif maximum_error <= float(tolerance):
        print("RESULT: WITHIN CONFIGURED START TOLERANCE")
    else:
        print("RESULT: OUTSIDE CONFIGURED START TOLERANCE")
    print("No motion or gripper command was issued.")


if __name__ == "__main__":
    main()
