#!/usr/bin/env python3
"""Repeat a single commissioned bracket cell's setup move and placement capture.

Every repetition uses an unrecorded wall-to-cell setup move followed by one
cell-to-wall bracket placement episode. A change of cell requires its own new
live start validation; this launcher never guesses a cross-cell transition.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--execute-supervised-trial", action="store_true")
    args = parser.parse_args()
    if args.start_episode < 1 or args.count < 1:
        raise SystemExit("start episode and count must be positive")
    config = args.config.expanduser().resolve()
    for offset in range(args.count):
        episode = args.start_episode + offset
        print(f"\n[BRACKET-CYCLE] {offset + 1}/{args.count}, episode {episode}", flush=True)
        command = [
            sys.executable, str(ROOT / "model-running" / "run_bracket_wall_to_cell.py"),
            "--config", str(config), "--episode-index", str(episode),
        ]
        if args.execute_supervised_trial:
            command.append("--execute-supervised-trial")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(f"[STOPPED] cycle {offset + 1} exited {completed.returncode}")
        if not args.execute_supervised_trial:
            break


if __name__ == "__main__":
    main()
