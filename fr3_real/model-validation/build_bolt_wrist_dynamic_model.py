#!/usr/bin/env python3
"""Build a q7-conditioned empty-wrist torque model and compare accepted Pilot drives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("engagement_dirs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-count", type=int, default=80)
    args = parser.parse_args()
    root = args.baseline_dir.expanduser().resolve()
    arrays = np.load(root / "dynamic_baseline.npz")
    trace = np.asarray(arrays["trace"], dtype=np.float64)
    columns = arrays["columns"].tolist()
    column = {name: columns.index(name) for name in ("phase", "q7", "dq7", "K_Tz", "tau_ext_7")}
    forward = trace[(trace[:, column["phase"]] == 1) & (trace[:, column["dq7"]] > 0.05)]
    if len(forward) < args.bin_count * 2:
        raise SystemExit("insufficient moving forward baseline samples")
    edges = np.linspace(np.min(forward[:, column["q7"]]), np.max(forward[:, column["q7"]]), args.bin_count + 1)
    knots = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = forward[(forward[:, column["q7"]] >= lo) & (forward[:, column["q7"]] <= hi)]
        if len(rows):
            knots.append([
                float(np.median(rows[:, column["q7"]])),
                float(np.median(rows[:, column["K_Tz"]])),
                float(np.median(rows[:, column["tau_ext_7"]])),
                len(rows),
            ])
    knots_array = np.asarray(knots, dtype=np.float64)
    trials = []
    all_residuals = []
    for directory in args.engagement_dirs:
        capture = directory.expanduser().resolve()
        metadata = json.loads((capture / "metadata.json").read_text())
        if metadata.get("accepted_by_operator") is not True:
            continue
        samples = read_jsonl(capture / "trajectory.jsonl")
        events = read_jsonl(capture / "events.jsonl")
        starts = [event for event in events if event["event"] == "ratchet_drive_started"]
        stops = [event for event in events if event["event"] == "ratchet_drive_complete"]
        residuals = []
        for start, stop in zip(starts, stops):
            rows = [
                sample for sample in samples
                if int(start["monotonic_time_ns"]) <= int(sample["monotonic_time_ns"]) <= int(stop["monotonic_time_ns"])
            ]
            q7 = np.asarray([row["q"][6] for row in rows])
            dq7 = np.asarray([row["dq"][6] for row in rows])
            tz = np.asarray([row["K_F_ext_hat_K"][5] for row in rows])
            mask = (
                (q7 >= knots_array[0, 0])
                & (q7 <= knots_array[-1, 0])
                & (dq7 > 0.05)
            )
            expected = np.interp(q7[mask], knots_array[:, 0], knots_array[:, 1])
            residuals.extend((tz[mask] - expected).tolist())
        if residuals:
            values = np.asarray(residuals)
            all_residuals.extend(residuals)
            trials.append({
                "capture": capture.name,
                "sample_count": len(values),
                "minimum_residual_K_Tz_Nm": float(np.min(values)),
                "p01_residual_K_Tz_Nm": float(np.percentile(values, 1)),
                "median_residual_K_Tz_Nm": float(np.median(values)),
                "p99_residual_K_Tz_Nm": float(np.percentile(values, 99)),
                "maximum_residual_K_Tz_Nm": float(np.max(values)),
            })
    combined = np.asarray(all_residuals, dtype=np.float64)
    document = {
        "schema_version": 1,
        "calibration_kind": "q7_conditioned_empty_wrist_forward_torque_model",
        "source_dynamic_baseline": str(root),
        "model_conditions": {
            "bolt_present": False,
            "Hand_empty": True,
            "relative_dynamics_factor": 0.05,
            "direction": "positive_q7_tightening",
            "minimum_abs_dq7_rad_s": 0.05,
        },
        "columns": ["q7_rad", "median_K_Tz_Nm", "median_tau_ext_joint7_Nm", "sample_count"],
        "knots": knots,
        "accepted_pilot_drive_residual_comparison": trials,
        "combined_accepted_pilot_residual": (
            {
                "sample_count": len(combined),
                "minimum_Nm": float(np.min(combined)),
                "p01_Nm": float(np.percentile(combined, 1)),
                "p05_Nm": float(np.percentile(combined, 5)),
                "median_Nm": float(np.median(combined)),
                "p95_Nm": float(np.percentile(combined, 95)),
                "p99_Nm": float(np.percentile(combined, 99)),
                "maximum_Nm": float(np.max(combined)),
            }
            if len(combined)
            else None
        ),
        "thresholds_derived": False,
        "candidate_hard_negative_residual_cutoff_Nm": (
            float(np.min(combined)) if len(combined) else "NOT_AVAILABLE"
        ),
        "candidate_basis": "most-negative q7-conditioned residual observed in operator-accepted successful Pilot engagement drives; human-confounded and requires explicit user approval",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing calibration: {output}")
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"[model] q7 knots:         {len(knots)}")
    print(f"[model] accepted residual samples: {len(combined)}")
    print(f"[model] candidate hard negative residual cutoff: {document['candidate_hard_negative_residual_cutoff_Nm']} Nm")
    print(f"[model] wrote {output}")
    print("[model] no threshold was activated; explicit approval is required")


if __name__ == "__main__":
    main()
