#!/usr/bin/env python3
"""Plan a discrete HIGH/CENTER/LOW FR3 bracket expert episode.

Only ``--dry-run`` and ``--teach-only`` are commissioned. ``--execute`` is
present as an explicit safety boundary and refuses before importing Franky or
opening a hardware connection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bracket_expert import (  # noqa: E402
    all_stage1_pose_names,
    build_bracket_plan,
    load_bracket_config,
)
from expert_demo.calibration import CalibrationError, CalibrationStore  # noqa: E402
from expert_demo.episode_spec import EpisodeSpecError, generate_bracket_episode_spec  # noqa: E402


DEFAULT_CONFIG = FR3_ROOT / "configs" / "bracket_expert_v1.json"


def resolve_calibration_path(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return (FR3_ROOT / config["calibration_file"]).resolve()


def print_required_teaching(config: dict, calibration_path: Path) -> None:
    print("Required Stage-1 pose names:")
    for name in all_stage1_pose_names():
        print(f"  {name}")
    print("\nSnapshot-only teaching command:")
    print(
        "  python franka-tests/teach_expert_poses.py "
        f"--robot-ip {config['robot_ip']} --calibration {calibration_path} --teach-only"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--teach-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slot")
    parser.add_argument("--grasp")
    args = parser.parse_args()

    config = load_bracket_config(args.config)
    calibration_path = resolve_calibration_path(config, args.calibration)
    print("=== FR3 BRACKET EXPERT ===")
    print(f"mode:               {'DRY_RUN' if args.dry_run else 'TEACH_ONLY' if args.teach_only else 'EXECUTE'}")
    print(f"robot IP:           {config['robot_ip']}")
    print(f"task:               {config['task']}")
    print(f"prompt:             {config['language_prompt']}")
    print(f"config:             {Path(args.config).resolve()}")
    print(f"calibration:        {calibration_path}")
    print(f"execution enabled:  {config['execution_enabled']}")
    print(f"dynamics factor:    {config['motion']['dynamics_factor']}")
    print(f"workspace bounds:   {config['safety']['workspace_bounds_m']}")

    if args.teach_only:
        print_required_teaching(config, calibration_path)
        return
    if args.execute:
        raise SystemExit(
            "[SAFETY] Physical bracket execution is not commissioned. "
            "No Franky import or robot connection was attempted."
        )

    try:
        store = CalibrationStore(calibration_path, config["robot_ip"])
        spec = generate_bracket_episode_spec(
            config,
            episode_index=args.episode_index,
            random_seed=args.seed,
            bracket_slot=args.slot,
            bracket_grasp_style=args.grasp,
        )
        plan = build_bracket_plan(spec, config, store.poses)
    except (CalibrationError, EpisodeSpecError, ValueError) as error:
        raise SystemExit(f"[dry-run] invalid configuration: {error}") from error

    print(f"\nEPISODE {spec.episode_index:03d}")
    print(f"Place bracket in {spec.bracket_slot}")
    print(f"Selected grasp: {spec.bracket_grasp_style}")
    print(f"Random seed: {spec.random_seed}")
    print("Physical readiness confirmation: REQUIRED DURING EXECUTION; not requested in dry-run")
    print("\nState-machine plan:")
    for index, step in enumerate(plan.steps, start=1):
        target = "-"
        if step.calibration_name is not None:
            if step.target_T_base_ee is None:
                target = f"{step.calibration_name} [MISSING]"
            else:
                xyz = step.target_T_base_ee[:3, 3]
                target = f"{step.calibration_name} xyz=({xyz[0]:+.4f},{xyz[1]:+.4f},{xyz[2]:+.4f})"
        print(f"  {index:02d} {step.state.value:<10} {step.operation:<26} {target}")

    if plan.safety_blockers:
        print("\nExecution blockers:")
        for blocker in plan.safety_blockers:
            print(f"  - {blocker}")
    print(f"\nDry-run result: {'EXECUTABLE' if plan.executable else 'NOT EXECUTABLE'}")
    print("No robot/camera/gripper connection was attempted.")


if __name__ == "__main__":
    main()
