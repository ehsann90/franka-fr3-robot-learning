#!/usr/bin/env python3
"""Audit the unrecorded wall-to-cell setup move for bracket placement demos.

A reversed forward demonstration is not, by itself, authorization to execute
a contact motion.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_demo.retimed_trajectory import build_retimed_trajectory, write_retimed_csv  # noqa: E402
from prepare_manual_bracket_replay import build_manual_stage_plan  # noqa: E402

STAGES = ("approach_open", "transport_place_closed", "retreat_open")
INVERSE = (
    ("wall_approach_open", "retreat_open"),
    ("wall_to_cell_held", "transport_place_closed"),
    ("cell_retreat_open", "approach_open"),
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def reverse_plan(plan):
    """Reverse joint and recorded Cartesian samples, preserving zero-speed ends."""
    duration = float(plan.command_times_s[-1])
    return replace(
        plan,
        name=f"reverse_{plan.name}",
        command_times_s=duration - plan.command_times_s[::-1],
        q_targets=plan.q_targets[::-1].copy(),
        dq_targets=-plan.dq_targets[::-1].copy(),
        xyz_targets=plan.xyz_targets[::-1].copy(),
        expected_start_q=plan.expected_stop_q.copy(),
        expected_stop_q=plan.expected_start_q.copy(),
    )


def retime(plan, config, *, stage_name=None):
    trim = config.get("idle_edge_trim", {})
    setup = config.get("inverse_setup_motion", {})
    scale = float(setup.get("playback_time_scales", {}).get(
        stage_name, config["playback_time_scale"]
    ))
    if scale < float(config["playback_time_scale"]):
        raise ValueError("inverse setup may not run faster than the bracket replay")
    return build_retimed_trajectory(
        plan,
        playback_time_scale=scale,
        validation_rate_hz=float(config["offline_validation_rate_hz"]),
        idle_trim_velocity_threshold_rad_s=(
            float(trim["interval_velocity_threshold_rad_s"])
            if trim.get("enabled") is True else 0.0
        ),
        idle_trim_buffer_s=(
            float(trim["source_time_buffer_s"])
            if trim.get("enabled") is True else 0.0
        ),
    )


def audit(name, plan, trajectory, config):
    limits = {
        "velocity_rad_s": (trajectory.peak_abs_velocity_rad_s, float(config["velocity_cap_rad_s"])),
        "acceleration_rad_s2": (trajectory.peak_abs_acceleration_rad_s2, float(config["acceleration_cap_rad_s2"])),
        "jerk_rad_s3": (trajectory.peak_abs_jerk_rad_s3, float(config["jerk_cap_rad_s3"])),
    }
    workspace = config["stages"][name.removeprefix("reverse_")]["workspace_bounds_m"]
    xyz_min = np.asarray(workspace["min_xyz"], dtype=float)
    xyz_max = np.asarray(workspace["max_xyz"], dtype=float)
    blockers = [key for key, (peak, cap) in limits.items() if peak > cap]
    if np.any(plan.source_xyz_min < xyz_min) or np.any(plan.source_xyz_max > xyz_max):
        blockers.append("recorded TCP path outside stage workspace")
    if np.max(np.abs(trajectory.evaluate(0.0, 1))) > 1e-9 or np.max(
        np.abs(trajectory.evaluate(trajectory.duration_s, 1))
    ) > 1e-9:
        blockers.append("nonstationary stage endpoint")
    return {
        "duration_s": trajectory.duration_s,
        "start_q": trajectory.expected_start_q.tolist(),
        "stop_q": trajectory.expected_stop_q.tolist(),
        "recorded_xyz_min_m": plan.source_xyz_min.tolist(),
        "recorded_xyz_max_m": plan.source_xyz_max.tolist(),
        "peaks": {key: peak for key, (peak, _) in limits.items()},
        "blockers": blockers,
    }


def prepare(config_path: Path, output_dir: Path | None):
    config = json.loads(config_path.read_text())
    candidate = project_path(config["candidate"])
    plans = {
        name: build_manual_stage_plan(
            candidate, name, command_rate_hz=float(config["command_rate_hz"])
        ) for name in STAGES
    }
    # Recorded Hand transition boundaries may have a small pose discrepancy;
    # the live collector must check these before each grasp and motion.
    joins = {
        "cell_grasp_rad": float(np.max(np.abs(
            plans["approach_open"].expected_stop_q -
            plans["transport_place_closed"].expected_start_q
        ))),
        "wall_release_rad": float(np.max(np.abs(
            plans["transport_place_closed"].expected_stop_q -
            plans["retreat_open"].expected_start_q
        ))),
    }
    trajectories = {}
    stages = {}
    for name, source in INVERSE:
        reversed_plan = reverse_plan(plans[source])
        trajectory = retime(reversed_plan, config, stage_name=name)
        trajectories[name] = trajectory
        stages[name] = audit(source, reversed_plan, trajectory, config)
    blockers = [f"{name}: {item}" for name, info in stages.items() for item in info["blockers"]]
    tolerance = float(config["controller_start_tolerance_rad"])
    blockers += [f"{name} stage join exceeds {tolerance:.4f} rad" for name, error in joins.items() if error > tolerance]
    calibration_only = bool(config.get("calibration_only", False))
    sequence = [
        "wall_approach_open", "grasp_on_wall", "wall_to_cell_held",
        "release_at_recorded_cell_grasp_pose", "cell_retreat_open",
        "recorded_approach_start_as_home",
    ]
    if not calibration_only:
        sequence += ["start_placement_episode"]
    sequence += [
        "approach_open", "grasp_at_cell", "transport_place_closed",
        "release_on_wall", "retreat_open",
    ]
    if not calibration_only:
        sequence += ["stop_placement_episode"]
    manifest = {
        "schema_version": 1,
        "task": "remove_bracket_from_wall_and_place_at_cell",
        "bracket_slot": config["bracket_slot"],
        "candidate": str(candidate),
        "source_config": str(config_path),
        "inverse_setup_log_root": "diagnostics/bracket_wall_to_cell_setup_trials_v1",
        "inverse_dataset_episode": False,
        "placement_training_root": None if calibration_only else "recordings/full_bracket_bolt_autonomous_raw_v1",
        "calibration_only": calibration_only,
        "sequence": sequence,
        "stage_joins_rad": joins,
        "inverse_stages": stages,
        "blockers": blockers,
        "inverse_motion_execution_mode": "supervised_setup_move_only",
        "home_bank_commissioned": False,
        "placement_capture_reference": (
            "model-running/run_manual_bracket_pick_and_place.py"
            if calibration_only else "model-running/record_autonomous_bolt_engagement_demo.py --bracket-only"
        ),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, trajectory in trajectories.items():
            write_retimed_csv(output_dir / f"{name}.csv", trajectory)
        (output_dir / "cycle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Commissioned B-cell bracket config")
    parser.add_argument("--prepare", action="store_true", help="Save offline candidate plans and manifest")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text())
    default_dir = ROOT / "recordings" / "bracket_wall_cell_cycle_candidates_v1" / str(config["bracket_slot"])
    manifest = prepare(config_path, project_path(args.output_dir) if args.output_dir else default_dir if args.prepare else None)
    print(json.dumps(manifest, indent=2))
    if args.prepare:
        print(f"[saved] offline candidate: {project_path(args.output_dir) if args.output_dir else default_dir}")
    if manifest["blockers"]:
        raise SystemExit("[BLOCKED] inverse trajectory failed offline audit")
    print("[PREPARED] inverse path passes offline math; physical inverse and home transitions remain uncommissioned")


if __name__ == "__main__":
    main()
