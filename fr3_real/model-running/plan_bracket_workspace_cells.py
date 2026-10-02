#!/usr/bin/env python3
"""Build read-only bracket-cell proposals from four guided EE measurements.

The proposal is geometry, not a robot program: no trajectory, collision check,
bracket footprint, or cell release has been commissioned by this script.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from prepare_manual_bracket_replay import build_manual_stage_plan


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORNERS = ROOT / "calibration" / "bracket_workspace_corners_v1" / "probed_points.json"
CORNER_NAMES = (
    "bracket_xmin_ymin", "bracket_xmin_ymax",
    "bracket_xmax_ymin", "bracket_xmax_ymax",
)


def normalize_corners(raw: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    """Normalize Markdown-quoted labels and order each X pair by measured Y."""
    cleaned: dict[str, np.ndarray] = {}
    for label, value in raw.items():
        name = str(label).strip().strip("`")
        if name in cleaned:
            raise ValueError(f"duplicate corner label after normalization: {name}")
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"{label}: expected finite XYZ coordinates")
        cleaned[name] = point
    if set(cleaned) != set(CORNER_NAMES):
        raise ValueError(f"expected exactly {CORNER_NAMES}; found {tuple(cleaned)}")

    corners: dict[str, np.ndarray] = {}
    warnings = []
    for x_side in ("xmin", "xmax"):
        labelled_low = cleaned[f"bracket_{x_side}_ymin"]
        labelled_high = cleaned[f"bracket_{x_side}_ymax"]
        low, high = sorted((labelled_low, labelled_high), key=lambda point: point[1])
        corners[f"bracket_{x_side}_ymin"] = low
        corners[f"bracket_{x_side}_ymax"] = high
        if labelled_low[1] > labelled_high[1]:
            warnings.append(f"{x_side}: recorded ymin/ymax labels are reversed; corrected only in this preview")
    left = np.asarray([corners["bracket_xmin_ymin"], corners["bracket_xmin_ymax"]])
    right = np.asarray([corners["bracket_xmax_ymin"], corners["bracket_xmax_ymax"]])
    if left[:, 0].max() >= right[:, 0].min():
        raise ValueError("measured xmin and xmax sides overlap in base-frame X")
    if min(left[1, 1] - left[0, 1], right[1, 1] - right[0, 1]) < 0.05:
        raise ValueError("measured ymin/ymax corners are too close together")
    z_span = max(point[2] for point in corners.values()) - min(point[2] for point in corners.values())
    if z_span > 0.005:
        warnings.append(f"corner EE height varies by {z_span * 1000:.1f} mm; interpolated Z is not a calibrated release height")
    return corners, warnings


def bilinear(corners: dict[str, np.ndarray], x_fraction: float, y_fraction: float) -> np.ndarray:
    low = (1 - y_fraction) * corners["bracket_xmin_ymin"] + y_fraction * corners["bracket_xmin_ymax"]
    high = (1 - y_fraction) * corners["bracket_xmax_ymin"] + y_fraction * corners["bracket_xmax_ymax"]
    return (1 - x_fraction) * low + x_fraction * high


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain XY hull, counterclockwise; no geometry dependency."""
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique) < 3:
        raise ValueError("need at least three distinct recorded grasp positions")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def inside_convex_hull(point: np.ndarray, hull: np.ndarray) -> bool:
    edges = np.roll(hull, -1, axis=0) - hull
    offsets = np.asarray(point, dtype=float) - hull
    return bool(np.all(edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0] >= -1e-9))


def accepted_grasp_positions() -> list[dict]:
    result = []
    configs = ROOT / "configs"
    for path in sorted(configs.glob("bracket_manual_replay_1x_highcap_b*.json")):
        match = re.fullmatch(r"bracket_manual_replay_1x_highcap_b(\d+)\.json", path.name)
        if not match:
            continue
        config = json.loads(path.read_text())
        candidate = ROOT / config["candidate"]
        sidecar = json.loads(candidate.with_suffix(".metadata.json").read_text())
        metadata = json.loads((Path(sidecar["source_episode"]) / "metadata.json").read_text())
        if metadata.get("success") is not True or metadata.get("accepted_by_operator") is not True:
            continue
        plan = build_manual_stage_plan(candidate, "approach_open", command_rate_hz=float(config["command_rate_hz"]))
        result.append({"cell": config["bracket_slot"], "xyz_m": plan.xyz_targets[-1].tolist()})
    if len(result) < 3:
        raise ValueError("too few accepted manual bracket grasps to compare proposals")
    return result


def make_preview(raw: dict, *, x_count: int, y_count: int, inset: float) -> dict:
    if x_count < 1 or y_count < 1 or not 0.0 <= inset < 0.5:
        raise ValueError("counts must be positive and inset must be in [0, 0.5)")
    corners, warnings = normalize_corners(raw)
    existing = accepted_grasp_positions()
    xy = np.asarray([entry["xyz_m"][:2] for entry in existing])
    hull = convex_hull(xy)
    fractions_x = np.linspace(inset, 1 - inset, x_count)
    fractions_y = np.linspace(inset, 1 - inset, y_count)
    candidates = []
    for row, x_fraction in enumerate(fractions_x, start=1):
        for column, y_fraction in enumerate(fractions_y, start=1):
            position = bilinear(corners, float(x_fraction), float(y_fraction))
            distances = np.linalg.norm(xy - position[:2], axis=1)
            closest = int(np.argmin(distances))
            candidates.append({
                "proposal": f"P{row:02d}_{column:02d}",
                "x_fraction": round(float(x_fraction), 4),
                "y_fraction": round(float(y_fraction), 4),
                "ee_xyz_m": np.round(position, 5).tolist(),
                "inside_recorded_grasp_hull": inside_convex_hull(position[:2], hull),
                "nearest_accepted_cell": existing[closest]["cell"],
                "nearest_accepted_xy_distance_m": round(float(distances[closest]), 5),
                "motion_approved": False,
            })
    return {
        "schema_version": 1,
        "kind": "read_only_bracket_cell_geometry_preview",
        "coordinate_frame": "robot_base_end_effector_origin_m",
        "raw_corner_file": str(DEFAULT_CORNERS),
        "normalized_corners_xyz_m": {name: point.tolist() for name, point in corners.items()},
        "warnings": warnings,
        "accepted_grasp_count": len(existing),
        "accepted_grasp_xy_min_m": xy.min(axis=0).tolist(),
        "accepted_grasp_xy_max_m": xy.max(axis=0).tolist(),
        "accepted_grasp_hull_xy_m": hull.tolist(),
        "candidates": candidates,
        "execution_enabled": False,
        "required_before_execution": [
            "confirm normalized physical corner order",
            "measure bracket footprint and release/grasp offset",
            "check clearances and reachability for each candidate",
            "commission new wall-to-cell and cell-to-wall paths without reversing pin-contact motion",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corners", type=Path, default=DEFAULT_CORNERS)
    parser.add_argument("--x-count", type=int, default=3)
    parser.add_argument("--y-count", type=int, default=5)
    parser.add_argument("--inset", type=float, default=0.15)
    parser.add_argument("--output", type=Path, help="Optional JSON preview output; no robot connection")
    args = parser.parse_args()
    corners_path = args.corners.expanduser().resolve()
    preview = make_preview(json.loads(corners_path.read_text()), x_count=args.x_count, y_count=args.y_count, inset=args.inset)
    preview["raw_corner_file"] = str(corners_path)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(preview, indent=2) + "\n")
        print(f"[saved] {output}")
    print("=== BRACKET CELL GEOMETRY PREVIEW — NO ROBOT COMMANDS ===")
    for name, point in preview["normalized_corners_xyz_m"].items():
        print(f"{name}: {point}")
    for warning in preview["warnings"]:
        print(f"[warning] {warning}")
    print(f"accepted grasp references: {preview['accepted_grasp_count']}")
    print(f"candidate points: {len(preview['candidates'])}")
    print(f"outside recorded grasp hull: {sum(not row['inside_recorded_grasp_hull'] for row in preview['candidates'])}")
    print("No arm, Hand, camera, or collision-setting command was issued.")


if __name__ == "__main__":
    main()
