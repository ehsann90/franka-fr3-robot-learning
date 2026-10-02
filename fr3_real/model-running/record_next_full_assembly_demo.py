#!/usr/bin/env python3
"""Record one manual bracket and immediately run its full bolt assembly demo.

Usage from ``fr3_real/``::

    python model-running/record_next_full_assembly_demo.py --slot B52 --episode-index 52

The script runs the read-only manual bracket recorder, prepares the recorded
path locally, creates the slot-specific 90%-speed replay configuration, and
starts the existing full bracket-plus-bolt recorder. The IK preparation and
configuration generation require no model or assistant interaction; the
existing safety prompts in both recording programs remain active.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = FR3_ROOT / "recordings"
CONFIGS = FR3_ROOT / "configs"
PYTHON = sys.executable
MANUAL_RAW = RECORDINGS / "bracket_manual_raw_v1"
IK_OUTPUT = RECORDINGS / "bracket_manual_replay_ik_preview_v1"
TEMPLATE = CONFIGS / "bracket_manual_replay_1x_highcap_b50.json"


def run(command: list[str]) -> None:
    print("[NEXT-DEMO] $", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=FR3_ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def latest_manual_episode(slot: str, episode_index: int) -> Path:
    prefix = f"*_{episode_index:04d}_{slot}_"
    candidates = sorted(MANUAL_RAW.glob(prefix + "*/"), key=lambda path: path.stat().st_mtime)
    for episode in reversed(candidates):
        try:
            metadata = json.loads((episode / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("success") is True and metadata.get("accepted_by_operator") is True:
            return episode
    raise SystemExit(f"No accepted successful manual bracket episode found for {slot}")


def latest_candidate(before: set[Path]) -> Path:
    candidates = set(IK_OUTPUT.glob("*_bracket_manual_exact_pose_rebalanced_ik_candidate.csv"))
    new_candidates = candidates - before
    if not new_candidates:
        raise SystemExit("IK preparation did not produce a new candidate CSV")
    return max(new_candidates, key=lambda path: path.stat().st_mtime)


def make_slot_config(slot: str, candidate: Path, speed: float) -> Path:
    config = json.loads(TEMPLATE.read_text())
    config["bracket_slot"] = slot
    config["candidate"] = str(candidate.relative_to(FR3_ROOT))
    config["playback_time_scale"] = 1.0 / speed
    config["note"] = (
        f"{slot} bracket replay at {speed:.0%} speed, generated from the accepted "
        "manual bracket episode by record_next_full_assembly_demo.py."
    )
    config["commissioning_basis"]["offline_validation_date"] = time.strftime("%Y-%m-%d")
    config["commissioning_basis"]["offline_result"] = (
        "Candidate generated locally; the full-demo recorder performs its existing "
        "offline trajectory and execution gates before motion."
    )
    output = CONFIGS / f"bracket_manual_replay_1x_highcap_{slot.lower()}.json"
    output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"[NEXT-DEMO] generated replay config: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True, help="Bracket slot, for example B52.")
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--grasp", default="CENTER", choices=("CENTER",))
    parser.add_argument("--wrist-side", default="COMMISSIONED", choices=("COMMISSIONED",))
    parser.add_argument("--speed", type=float, default=0.90, help="Bracket replay speed fraction.")
    parser.add_argument("--skip-manual", action="store_true", help="Use an already recorded bracket episode.")
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        raise SystemExit("--speed must be greater than 0 and no greater than 1")

    slot = args.slot.upper()
    if not args.skip_manual:
        run([
            PYTHON,
            "franka-tests/record_manual_bracket_demo.py",
            "--record",
            "--slot", slot,
            "--episode-index", str(args.episode_index),
            "--grasp", args.grasp,
            "--wrist-side", args.wrist_side,
        ])

    episode = latest_manual_episode(slot, args.episode_index)
    before = set(IK_OUTPUT.glob("*_bracket_manual_exact_pose_rebalanced_ik_candidate.csv"))
    run([
        PYTHON,
        "model-running/prepare_bracket_outer_ik_preview.py",
        "--read-only-preview",
        "--manual-episode", str(episode),
    ])
    candidate = latest_candidate(before)
    config = make_slot_config(slot, candidate, args.speed)
    run([
        PYTHON,
        "model-running/record_autonomous_bolt_engagement_demo.py",
        "--record-demo",
        "--full-assembly",
        "--bracket-only",
        "--four-x",
        "--bracket-source", "manual_2026_08_16",
        "--bracket-arm-config", str(config),
        "--episode-index", str(args.episode_index),
    ])


if __name__ == "__main__":
    main()
