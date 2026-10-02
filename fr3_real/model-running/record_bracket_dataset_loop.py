#!/usr/bin/env python3
"""Loop manual bracket recording and bracket-only replay with incrementing slots.

Usage from ``fr3_real/``::

    python model-running/record_bracket_dataset_loop.py --start-slot B53 --start-episode 53 --count 10

Each cycle records one bracket with Pilot, prepares its path, replays only the
bracket pick/place at 90 percent speed, saves the accepted dataset episode,
then waits for Enter before starting the next incremented slot. The bolt stays
fixed in the scene and no bolt stages are executed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent


def run(slot: str, episode: int, speed: float) -> None:
    command = [
        sys.executable,
        "model-running/record_next_full_assembly_demo.py",
        "--slot", slot,
        "--episode-index", str(episode),
        "--speed", str(speed),
    ]
    print("[BRACKET-LOOP] $", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=FR3_ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-slot", required=True, help="For example B53.")
    parser.add_argument("--start-episode", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--speed", type=float, default=0.90)
    args = parser.parse_args()
    match = re.fullmatch(r"B(\d+)", args.start_slot.upper())
    if not match or args.start_episode < 1 or args.count < 1:
        raise SystemExit("Use a slot like B53, a positive episode, and a positive count")
    if not 0.0 < args.speed <= 1.0:
        raise SystemExit("--speed must be greater than 0 and no greater than 1")

    start_slot = int(match.group(1))
    for offset in range(args.count):
        slot_number = start_slot + offset
        episode = args.start_episode + offset
        slot = f"B{slot_number}"
        print(f"\n[BRACKET-LOOP] episode {episode} slot {slot}", flush=True)
        run(slot, episode, args.speed)
        if offset + 1 < args.count:
            input("Press Enter to begin the next bracket episode: ")


if __name__ == "__main__":
    main()
