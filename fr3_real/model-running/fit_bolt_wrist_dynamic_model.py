#!/usr/bin/env python3
"""Fit a reviewable q7-conditioned wrench model from an empty-Hand baseline.

Usage from ``fr3_real/``::

    python model-running/fit_bolt_wrist_dynamic_model.py \
        recordings/bolt_wrist_dynamic_baselines_extended_100deg_v1/<episode>

The forward phase is binned into robust median knots. The existing production
model is never overwritten; the candidate must be reviewed before use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--knots", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=FR3_ROOT / "calibration/bolt_wrist_dynamic_model_extended_95deg_candidate.json",
    )
    args = parser.parse_args()
    episode = args.episode.expanduser().resolve()
    trace = np.load(episode / "dynamic_baseline.npz", allow_pickle=True)["trace"]
    forward = trace[trace[:, 1] == 1.0]
    if len(forward) < args.knots * 10:
        raise SystemExit("forward phase does not contain enough samples for the requested knots")

    q7 = forward[:, 8]
    ktz = forward[:, 21]
    edges = np.linspace(float(q7.min()), float(q7.max()), args.knots + 1)
    knots = []
    for index in range(args.knots):
        mask = (q7 >= edges[index]) & (q7 <= edges[index + 1] if index == args.knots - 1 else q7 < edges[index + 1])
        if not np.any(mask):
            continue
        knots.append([
            float(np.median(q7[mask])),
            float(np.median(ktz[mask])),
            float(np.median(forward[mask, 22])),
            int(np.count_nonzero(mask)),
        ])
    if len(knots) < args.knots - 2:
        raise SystemExit("too many empty q7 bins; collect a denser baseline")

    document = {
        "schema_version": 1,
        "calibration_kind": "q7_conditioned_empty_wrist_forward_torque_model_candidate",
        "source_dynamic_baseline": str(episode),
        "model_conditions": {
            "bolt_present": False,
            "Hand_empty": True,
            "direction": "positive_q7_tightening",
            "minimum_abs_dq7_rad_s": 0.05,
        },
        "columns": ["q7_rad", "median_K_Tz_Nm", "median_tau_ext_joint7_Nm", "sample_count"],
        "knots": knots,
        "candidate_status": "REVIEW_REQUIRED_NOT_PRODUCTION",
        "forward_sample_count": int(len(forward)),
        "forward_q7_range_rad": [float(q7.min()), float(q7.max())],
        "forward_K_Tz_range_Nm": [float(ktz.min()), float(ktz.max())],
        "basis": "median per q7 bin from the extended empty-Hand forward sweep; no threshold was derived automatically",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"[saved] candidate model: {output}")
    print(f"[candidate] knots={len(knots)} q7={q7.min():+.6f}..{q7.max():+.6f} rad samples={len(forward)}")
    print("[candidate] production model was not modified")


if __name__ == "__main__":
    main()
