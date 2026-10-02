#!/usr/bin/env python3
"""Offline-only preview of one staged piloted bracket replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.replay_plan import (  # noqa: E402
    ReplayPlanError,
    STAGE_EVENTS,
    build_replay_stage,
    replay_readiness_blockers,
)


DEFAULT_CONFIG = FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_EVENTS), default="approach_open")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    try:
        config = json.loads(args.config.expanduser().resolve().read_text())
        command_rate_hz = float(config["replay"]["command_rate_hz"])
        plan = build_replay_stage(
            args.trajectory_dir,
            args.stage,
            command_rate_hz=command_rate_hz,
        )
        blockers = replay_readiness_blockers(config, plan)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ReplayPlanError) as error:
        raise SystemExit(f"[replay preview] invalid input: {error}") from error

    print("=== PILOTED BRACKET REPLAY PREVIEW ===")
    print("mode:                   DRY_RUN / OFFLINE_ONLY")
    print(f"trajectory:             {args.trajectory_dir.expanduser().resolve()}")
    print(f"stage:                  {plan.name}")
    print(f"event range:            {plan.start_event} -> {plan.stop_event}")
    print(f"source sample range:    {plan.source_start_index} .. {plan.source_stop_index}")
    print(f"source duration:        {plan.source_duration_s:.3f} s")
    print(f"command rate:           {plan.command_rate_hz:.1f} Hz")
    print(f"absolute q targets:     {len(plan.q_targets)} x 7")
    print(f"expected start q:       {np.array2string(plan.expected_start_q, precision=4)}")
    print(f"expected stop q:        {np.array2string(plan.expected_stop_q, precision=4)}")
    print(f"source EE xyz min:      {np.array2string(plan.source_xyz_min, precision=4)}")
    print(f"source EE xyz max:      {np.array2string(plan.source_xyz_max, precision=4)}")
    print(f"source peak |dq|:       {plan.max_source_abs_dq_rad_s:.4f} rad/s")
    print(f"15 Hz max target step:  {plan.max_target_step_rad:.4f} rad")
    print(f"implied peak velocity:  {plan.implied_peak_velocity_rad_s:.4f} rad/s")
    print("\nExecution blockers:")
    for blocker in blockers:
        print(f"  - {blocker}")
    print("\nPhysical replay: DISABLED. No Franky/libfranka import or hardware connection occurred.")


if __name__ == "__main__":
    main()
