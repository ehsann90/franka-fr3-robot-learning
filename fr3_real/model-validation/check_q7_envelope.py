#!/usr/bin/env python3
"""Quick read-only check: does a manual bracket episode's q7 stay inside the
autonomous-replay-safe envelope ([-2.8946, 2.8946] rad)? Run this before the
expensive outer-wrist IK preview to catch an out-of-envelope capture early."""

import json
import sys
from pathlib import Path

SAFE_MIN = -2.8946
SAFE_MAX = 2.8946


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} EPISODE_DIR")
    episode_dir = Path(sys.argv[1])
    samples = [
        json.loads(line)
        for line in (episode_dir / "robot_state.jsonl").read_text().splitlines()
        if line.strip()
    ]
    q7 = [sample["q"][6] for sample in samples]
    q7_min, q7_max = min(q7), max(q7)
    within_envelope = q7_min >= SAFE_MIN and q7_max <= SAFE_MAX
    print(f"episode:      {episode_dir}")
    print(f"q7 min/max:   {q7_min:.4f} / {q7_max:.4f} rad")
    print(f"safe envelope: [{SAFE_MIN}, {SAFE_MAX}] rad")
    print(f"within safe envelope: {within_envelope}")


if __name__ == "__main__":
    main()
