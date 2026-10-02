#!/usr/bin/env python3
"""Runs the full manual-bracket pick-and-place sequence as one command.

This is a convenience wrapper only -- it does not remove or weaken any
individual guard. Each of the six steps below is exactly the same script
that would otherwise be run by hand, invoked as a subprocess with stdin left
attached to the terminal, so every existing "Press Enter" and typed
confirmation (RUN_RETIMED_2X_STAGE, CLOSE_ON_BRACKET, OPEN_AND_RELEASE)
still has to be answered live by whoever is at the robot. If any step exits
non-zero -- a blocker, a refused confirmation, an aborted step -- this stops
immediately and does not attempt the next one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODEL_RUNNING = Path(__file__).resolve().parent

ARM = [sys.executable, str(MODEL_RUNNING / "commission_manual_bracket_arm.py")]


def build_steps(arm_config: Path | None) -> list[tuple[str, list[str]]]:
    arm_config_args = ["--config", str(arm_config)] if arm_config else []
    return [
        ("approach_open (arm reset-to-start)",
         ARM + ["--prepare-controller-preview", "--stage", "approach_open", "--object-aware"] + arm_config_args),
        ("approach_open (arm)",
         ARM + ["--execute", "--stage", "approach_open", "--object-aware"] + arm_config_args),
        ("grasp (Hand)", [sys.executable, str(MODEL_RUNNING / "grasp_manual_bracket.py"), "--execute"] + arm_config_args),
        ("transport_place_closed (arm)",
         ARM + ["--execute", "--stage", "transport_place_closed", "--object-aware"] + arm_config_args),
        ("release (Hand)", [sys.executable, str(MODEL_RUNNING / "release_manual_bracket.py"), "--execute"] + arm_config_args),
        ("retreat_open (arm)",
         ARM + ["--execute", "--stage", "retreat_open", "--object-aware"] + arm_config_args),
    ]


def main() -> None:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path, default=None)
    args_partial, _ = probe.parse_known_args()
    steps = build_steps(args_partial.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None,
        help="commissioning config to use for every step (arm + grasp + release); "
             "defaults to each script's own default (the B1 config) if omitted",
    )
    parser.add_argument(
        "--start-at", type=int, default=1, choices=range(1, len(steps) + 1),
        help="resume from this step (1-indexed) instead of the beginning, "
             "so an already-completed step isn't redone after a later step failed",
    )
    args = parser.parse_args()
    STEPS = steps

    print("=== MANUAL BRACKET PICK-AND-PLACE (6 STEPS, ONE COMMAND) ===")
    print("Each step below is a full, independent script run with its own live checks")
    print("and typed confirmation. Nothing here skips or auto-answers those prompts.")
    if args.start_at > 1:
        print(f"Resuming from step {args.start_at}/{len(STEPS)}.")
    print()
    for index, (label, command) in enumerate(STEPS, start=1):
        if index < args.start_at:
            continue
        print(f"\n----- step {index}/{len(STEPS)}: {label} -----")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(
                f"[STOPPED] step {index}/{len(STEPS)} ({label}) exited "
                f"{completed.returncode}; not attempting the remaining steps. "
                f"Once resolved, resume with --start-at {index}."
            )
    print("\n=== ALL STEPS COMPLETED ===")
    print("Visually confirm the bracket is seated correctly on the alignment pins.")


if __name__ == "__main__":
    main()
