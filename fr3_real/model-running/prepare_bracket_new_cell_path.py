#!/usr/bin/env python3
"""Prepare read-only bracket IK targets for a new tabletop cell.

Translate a nearby recorded cell-side path, then blend its wall placement to a
separately passed fixed wall pose. The result is not an executable trajectory;
continuous IK and physical commissioning are separate requirements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare_bracket_outer_ik_preview import manual_stage_offsets, selected_manual_samples


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = ROOT / "calibration" / "bracket_new_cell_p01_02_v1.json"
DEFAULT_WALL = ROOT / "calibration" / "bracket_wall_keepout_v1" / "probed_points.json"
DEFAULT_OUTPUT = ROOT / "diagnostics" / "bracket_new_cell_p01_02_v1"
WORKSPACE_MIN = np.array([0.21, -0.53, -0.04], dtype=float)
WORKSPACE_MAX = np.array([0.73, 0.40, 0.32], dtype=float)
WALL_MARGIN_XY_M = 0.05
FADE_START_ARC_FRACTION = 0.10
FADE_END_ARC_FRACTION = 0.50
RAMP_START_ARC_FRACTION = 0.15
RAMP_END_ARC_FRACTION = 0.60
WALL_BLEND_START_ARC_FRACTION = 0.55
WALL_BLEND_END_ARC_FRACTION = 0.85


def table_plane_height(points: np.ndarray, xy: np.ndarray) -> float:
    plane, _, _, _ = np.linalg.lstsq(
        np.column_stack((points[:, :2], np.ones(len(points)))), points[:, 2], rcond=None
    )
    return float(np.dot(np.r_[xy, 1.0], plane))


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert a small 3-D rotation matrix to an axis-angle vector."""
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vector = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ])
    if angle < 1e-8:
        return 0.5 * skew_vector
    if np.pi - angle < 1e-6:
        values, vectors = np.linalg.eigh(0.5 * (matrix + np.eye(3)))
        axis = vectors[:, int(np.argmax(values))]
        if np.dot(axis, skew_vector) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * skew_vector


def rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        x, y, z = rotvec
        return np.array([
            [1.0, -z, y], [z, 1.0, -x], [-y, x, 1.0],
        ])
    axis = rotvec / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def smooth_remaining(arclength_fraction: np.ndarray) -> np.ndarray:
    u = np.clip(
        (arclength_fraction - FADE_START_ARC_FRACTION)
        / (FADE_END_ARC_FRACTION - FADE_START_ARC_FRACTION),
        0.0, 1.0,
    )
    return 1.0 - (3.0 * u**2 - 2.0 * u**3)


def blend_weights(times: np.ndarray, positions: np.ndarray,
                  approach_start_s: float, approach_end_s: float,
                  transfer_start_s: float, transfer_end_s: float) -> np.ndarray:
    if not transfer_end_s > transfer_start_s:
        raise ValueError("transport stage has no positive duration")
    approach = np.flatnonzero((times >= approach_start_s) & (times <= approach_end_s))
    transfer = np.flatnonzero((times >= transfer_start_s) & (times <= transfer_end_s))
    if len(approach) < 3 or len(transfer) < 3:
        raise ValueError("approach or transport stage has too few samples")
    approach_positions = positions[approach]
    approach_arclength = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(approach_positions, axis=0), axis=1))
    ]
    if approach_arclength[-1] < 0.20:
        raise ValueError("recorded approach is unexpectedly short")
    segment = positions[transfer]
    arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
    if arclength[-1] < 0.20:
        raise ValueError("recorded transport is unexpectedly short")
    weights = np.ones(len(times), dtype=float)
    weights[times < approach_start_s] = 0.0
    approach_fraction = approach_arclength / approach_arclength[-1]
    ramp_u = np.clip(
        (approach_fraction - RAMP_START_ARC_FRACTION)
        / (RAMP_END_ARC_FRACTION - RAMP_START_ARC_FRACTION),
        0.0, 1.0,
    )
    weights[approach] = 3.0 * ramp_u**2 - 2.0 * ramp_u**3
    weights[times > transfer_end_s] = 0.0
    weights[transfer] = smooth_remaining(arclength / arclength[-1])
    return weights


def wall_blend_weights(times: np.ndarray, positions: np.ndarray,
                       transfer_start_s: float, transfer_end_s: float) -> np.ndarray:
    transfer = np.flatnonzero((times >= transfer_start_s) & (times <= transfer_end_s))
    segment = positions[transfer]
    arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
    if len(transfer) < 3 or arclength[-1] < 0.20:
        raise ValueError("transport stage too short for wall-pose blend")
    fraction = arclength / arclength[-1]
    u = np.clip(
        (fraction - WALL_BLEND_START_ARC_FRACTION)
        / (WALL_BLEND_END_ARC_FRACTION - WALL_BLEND_START_ARC_FRACTION),
        0.0, 1.0,
    )
    weights = np.zeros(len(times), dtype=float)
    weights[transfer] = 3.0 * u**2 - 2.0 * u**3
    weights[times > transfer_end_s] = 1.0
    return weights


def recorded_wall_pose(config_path: Path) -> np.ndarray:
    config = json.loads(config_path.read_text())
    sidecar = json.loads((ROOT / config["candidate"]).with_suffix(".metadata.json").read_text())
    source = Path(sidecar["source_episode"])
    samples = selected_manual_samples(source)
    start_ns = int(samples[0]["time_ns"])
    wall_time = manual_stage_offsets(source, start_ns)["transport_place_closed"][1]
    times = np.asarray([(int(row["time_ns"]) - start_ns) / 1e9 for row in samples])
    index = int(np.argmin(np.abs(times - wall_time)))
    return np.asarray(samples[index]["O_T_EE"], dtype=float).reshape(4, 4)


def build_target_rows(spec_path: Path, wall_path: Path) -> tuple[list[dict], dict]:
    spec = json.loads(spec_path.read_text())
    if spec.get("motion_approved") is not False or spec.get("replay_enabled") is not False:
        raise ValueError("source cell spec must explicitly disable motion and replay")
    config = json.loads((ROOT / spec["source_replay_config"]).read_text())
    if config["bracket_slot"] != spec["source_bracket_slot"]:
        raise ValueError("source bracket slot mismatch")
    candidate = ROOT / config["candidate"]
    sidecar = json.loads(candidate.with_suffix(".metadata.json").read_text())
    source = Path(sidecar["source_episode"])
    samples = selected_manual_samples(source)
    source_start_ns = int(samples[0]["time_ns"])
    stages = manual_stage_offsets(source, source_start_ns)
    times = np.asarray([(int(row["time_ns"]) - source_start_ns) / 1e9 for row in samples])
    transforms = np.asarray([np.asarray(row["O_T_EE"], dtype=float).reshape(4, 4) for row in samples])
    positions = transforms[:, :3, 3]
    source_q = np.asarray([row["q"] for row in samples], dtype=float)
    if source_q.shape != (len(samples), 7) or not np.isfinite(transforms).all():
        raise ValueError("invalid source bracket samples")

    source_grasp_time = stages["approach_open"][1]
    source_grasp_index = int(np.argmin(np.abs(times - source_grasp_time)))
    source_grasp = positions[source_grasp_index]
    desired_probe = np.asarray(spec["target_probed_ee_origin_xyz_m"], dtype=float)
    if desired_probe.shape != (3,) or not np.isfinite(desired_probe).all():
        raise ValueError("new cell XY must be finite")
    corner_raw = json.loads((ROOT / "calibration/bracket_workspace_corners_v2/probed_points.json").read_text())
    corners = np.asarray(list(corner_raw.values()), dtype=float)
    if corners.shape != (4, 3):
        raise ValueError("four tabletop corner probes required")
    table_delta_z = table_plane_height(corners, desired_probe[:2]) - table_plane_height(
        corners, source_grasp[:2]
    )
    shift = np.array([
        desired_probe[0] - source_grasp[0],
        desired_probe[1] - source_grasp[1],
        table_delta_z,
    ])
    weights = blend_weights(
        times, positions, *stages["approach_open"], *stages["transport_place_closed"]
    )
    wall_weights = wall_blend_weights(times, positions, *stages["transport_place_closed"])
    wall_place_index = int(np.argmin(np.abs(times - stages["transport_place_closed"][1])))
    source_wall_pose = transforms[wall_place_index]
    fixed_wall_pose = recorded_wall_pose(ROOT / spec["fixed_wall_replay_config"])
    wall_shift = fixed_wall_pose[:3, 3] - source_wall_pose[:3, 3]
    wall_rotation = rotation_matrix_to_rotvec(
        fixed_wall_pose[:3, :3] @ source_wall_pose[:3, :3].T
    )
    if np.linalg.norm(wall_shift) > 0.01 or np.linalg.norm(wall_rotation) > 0.12:
        raise ValueError("fixed wall pose differs too much from the source wall placement")
    target_transforms = transforms.copy()
    target_transforms[:, :3, 3] += weights[:, None] * shift
    target_transforms[:, :3, 3] += wall_weights[:, None] * wall_shift
    for index, weight in enumerate(wall_weights):
        if weight > 0:
            correction = rotvec_to_rotation_matrix(weight * wall_rotation)
            target_transforms[index, :3, :3] = correction @ transforms[index, :3, :3]
    target_positions = target_transforms[:, :3, 3]
    if np.any(target_positions < WORKSPACE_MIN) or np.any(target_positions > WORKSPACE_MAX):
        raise ValueError("translated TCP targets leave the configured workspace")

    wall_points = np.asarray(list(json.loads(wall_path.read_text()).values()), dtype=float)
    if wall_points.shape != (4, 3) or not np.isfinite(wall_points).all():
        raise ValueError("four finite wall keep-out probes required")
    wall_low = wall_points[:, :2].min(axis=0) - WALL_MARGIN_XY_M
    wall_high = wall_points[:, :2].max(axis=0) + WALL_MARGIN_XY_M
    changed = weights > 1e-6
    inside_wall = np.all(
        (target_positions[:, :2] >= wall_low) & (target_positions[:, :2] <= wall_high),
        axis=1,
    )
    if np.any(changed & inside_wall):
        raise ValueError("translated portion enters the measured wall keep-out")
    wall_position_error = float(np.linalg.norm(target_positions[wall_place_index] - fixed_wall_pose[:3, 3]))
    wall_rotation_error = float(np.linalg.norm(rotation_matrix_to_rotvec(
        target_transforms[wall_place_index, :3, :3] @ fixed_wall_pose[:3, :3].T
    )))
    if wall_position_error > 1e-9 or wall_rotation_error > 1e-9:
        raise ValueError("new path does not reach the fixed wall placement pose exactly")

    rows = []
    for time_s, q, transform in zip(times, source_q, target_transforms):
        rows.append({
            "time_s": float(time_s), "source_q": q.tolist(), "target_T": transform.tolist()
        })
    audit = {
        "kind": "read_only_translated_bracket_ik_targets",
        "source_cell": config["bracket_slot"], "new_cell": spec["cell_label"],
        "source_episode": str(source), "target_count": len(rows),
        "source_grasp_xyz_m": source_grasp.tolist(),
        "new_grasp_xyz_m": target_positions[source_grasp_index].tolist(),
        "translation_xyz_m": shift.tolist(),
        "table_plane_height_change_m": table_delta_z,
        "source_wall_xyz_m": source_wall_pose[:3, 3].tolist(),
        "fixed_wall_reference_cell": spec["fixed_wall_bracket_slot"],
        "wall_place_xyz_m": target_positions[wall_place_index].tolist(),
        "wall_reference_position_error_m": wall_position_error,
        "wall_reference_orientation_error_rad": wall_rotation_error,
        "source_to_fixed_wall_translation_m": wall_shift.tolist(),
        "source_to_fixed_wall_rotation_rad": float(np.linalg.norm(wall_rotation)),
        "translation_fades_by_transport_arc_fraction": FADE_END_ARC_FRACTION,
        "translation_ramps_by_approach_arc_fraction": RAMP_END_ARC_FRACTION,
        "fixed_wall_pose_blend_by_transport_arc_fraction": WALL_BLEND_END_ARC_FRACTION,
        "modified_targets_inside_wall_keepout": False,
        "stage_offsets_s": stages,
        "dataset_episode": False, "motion_approved": False, "replay_enabled": False,
        "note": "No robot connection, IK solution, collision proof, or physical motion from this file.",
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--wall-probes", type=Path, default=DEFAULT_WALL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, audit = build_target_rows(args.spec.resolve(), args.wall_probes.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "translated_bracket_ik_targets.csv"
    with targets_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s"] + [f"source_q{i}" for i in range(1, 8)]
                        + [f"T{row}{column}" for row in range(4) for column in range(4)])
        for row in rows:
            writer.writerow([row["time_s"], *row["source_q"], *np.asarray(row["target_T"]).reshape(-1)])
    (output_dir / "translated_bracket_ik_targets.json").write_text(json.dumps(audit, indent=2) + "\n")
    sidecar = {
        "schema_version": 1,
        "candidate_kind": "translated_bracket_cell_with_fixed_wall_pose",
        "source_episode": audit["source_episode"],
        "stage_offsets_s": audit["stage_offsets_s"],
        "target_poses_csv": str(targets_path),
        "sample_count": len(rows),
        "execution_enabled": False,
        "motion_approved": False,
    }
    (output_dir / "translated_bracket_ik_candidate.metadata.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )
    print("=== NEW BRACKET CELL PATH TARGETS — READ ONLY ===")
    print(f"{audit['source_cell']} source grasp: {np.round(audit['source_grasp_xyz_m'], 5)} m")
    print(f"{audit['new_cell']} target grasp: {np.round(audit['new_grasp_xyz_m'], 5)} m")
    print(f"translation: {np.round(audit['translation_xyz_m'], 5)} m")
    print(f"fixed {audit['fixed_wall_reference_cell']} wall placement: {np.round(audit['wall_place_xyz_m'], 5)} m")
    print(f"targets: {audit['target_count']}; saved: {targets_path}")
    print("No robot connection or motion. IK solution and physical clearance are still unverified.")


if __name__ == "__main__":
    main()
