#!/usr/bin/env python3
"""Build a read-only phase plan from a successful full Pilot bolt capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.full_bolt_replay_plan import build_phase_boundaries, summarize_phase  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bolt_full_replay_v1.json"


def path_from_root(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (FR3_ROOT / path).resolve()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text())
    episode = (
        args.episode_dir.expanduser().resolve()
        if args.episode_dir is not None
        else path_from_root(config["source_episode"])
    )
    metadata = json.loads((episode / "metadata.json").read_text())
    if metadata.get("success") is not True:
        raise SystemExit("source episode is not successful")
    states = read_jsonl(episode / "robot_state.jsonl")
    events = read_jsonl(episode / "events.jsonl")
    actions = read_jsonl(episode / "actions.jsonl")
    phases = [
        summarize_phase(phase, states)
        for phase in build_phase_boundaries(events, actions)
    ]
    limits = json.loads(path_from_root(config["ratchet_limits_calibration"]).read_text())
    q7_min = float(limits["summary"]["q7_prewind_rad"])
    q7_max = float(limits["summary"]["q7_forward_rad"])
    blockers = []
    over_limit = []
    for phase in phases:
        if phase["q7_min_rad"] < q7_min or phase["q7_max_rad"] > q7_max:
            over_limit.append(phase["name"])
    if over_limit:
        blockers.append(
            "source q7 leaves conservative cable-safe interval in: " + ", ".join(over_limit)
        )
    if any(phase["contact_critical"] for phase in phases):
        blockers.append("contact-critical phases require compliant control; literal joint replay disabled")
    if config.get("contact_controller_status") != "COMMISSIONED":
        blockers.append("contact controller is not commissioned for full replay")
    if config.get("jam_detector_status") != "CALIBRATED":
        blockers.append("jam detector is not calibrated")
    if config.get("seated_detector_status") != "CALIBRATED":
        blockers.append("seated detector is not calibrated")
    if metadata.get("gripper_state_source") != "measured":
        blockers.append("source gripper state is operator-marked rather than continuously measured")
    if config.get("execution_enabled") is not True:
        blockers.append("full replay execution is disabled")

    plan = {
        "schema_version": 1,
        "source_episode": str(episode),
        "task_name": metadata["task_name"],
        "language_prompt": metadata["language_prompt"],
        "q7_cable_safe_interval_rad": [q7_min, q7_max],
        "phase_count": len(phases),
        "phases": phases,
        "execution_enabled": False,
        "execution_blockers": blockers,
    }
    print("=== FULL BOLT REPLAY PLAN — OFFLINE ONLY ===")
    print(f"source:                    {episode}")
    print(f"phases:                    {len(phases)}")
    print(f"q7 cable-safe interval:    {q7_min:+.6f} .. {q7_max:+.6f} rad")
    for phase in phases:
        marker = "CONTACT" if phase["contact_critical"] else "free-space"
        print(
            f"  {phase['name']:<26} {phase['duration_s']:7.2f}s "
            f"q7={phase['q7_start_rad']:+.3f}->{phase['q7_stop_rad']:+.3f} "
            f"max|dq|={phase['maximum_abs_dq_rad_s']:.3f} {marker}"
        )
    print("\nExecution blockers:")
    for blocker in blockers:
        print(f"  - {blocker}")
    if args.write:
        output = episode / "full_bolt_replay_plan.json"
        output.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"\n[saved] {output}")
    print("No robot, Hand, camera, or control connection was attempted.")


if __name__ == "__main__":
    main()
