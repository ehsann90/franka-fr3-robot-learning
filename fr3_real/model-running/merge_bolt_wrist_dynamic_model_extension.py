#!/usr/bin/env python3
"""Merge a validated q7 model with an offset-corrected upper-range extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=FR3_ROOT / "calibration/bolt_wrist_dynamic_model_v1.json")
    parser.add_argument("--extension", type=Path, default=FR3_ROOT / "calibration/bolt_wrist_dynamic_model_extended_95deg_candidate.json")
    parser.add_argument("--output", type=Path, default=FR3_ROOT / "calibration/bolt_wrist_dynamic_model_hybrid_95deg_candidate.json")
    parser.add_argument("--overlap-min", type=float, default=0.10)
    parser.add_argument("--overlap-max", type=float, default=1.50)
    args = parser.parse_args()
    base = json.loads(args.base.expanduser().resolve().read_text())
    extension = json.loads(args.extension.expanduser().resolve().read_text())
    b = np.asarray(base["knots"], dtype=np.float64)
    e = np.asarray(extension["knots"], dtype=np.float64)
    overlap = (e[:, 0] >= args.overlap_min) & (e[:, 0] <= args.overlap_max)
    if overlap.sum() < 10:
        raise SystemExit("extension has too few knots in the requested overlap")
    base_interp = np.interp(e[overlap, 0], b[:, 0], b[:, 1])
    offset = float(np.median(e[overlap, 1] - base_interp))
    spread = float(np.median(np.abs((e[overlap, 1] - base_interp) - offset)))
    extension_corrected = e.copy()
    extension_corrected[:, 1] -= offset
    keep_extension = extension_corrected[:, 0] > b[-1, 0]
    merged = np.vstack((b, extension_corrected[keep_extension]))
    merged = merged[np.argsort(merged[:, 0])]
    document = dict(base)
    document["knots"] = merged.tolist()
    document["calibration_kind"] = "q7_conditioned_empty_wrist_forward_torque_model_hybrid_candidate"
    document["candidate_status"] = "REVIEW_REQUIRED_NOT_PRODUCTION"
    document["extension_source"] = str(args.extension.expanduser().resolve())
    document["base_model_source"] = str(args.base.expanduser().resolve())
    document["overlap_offset_Nm"] = offset
    document["overlap_median_absolute_error_Nm"] = spread
    document["extension_q7_range_rad"] = [float(extension_corrected[:, 0].min()), float(extension_corrected[:, 0].max())]
    document["basis"] = "validated base knots preserved; extension shifted by robust median overlap offset; production model not modified"
    output = args.output.expanduser().resolve()
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"[saved] {output}")
    print(f"[hybrid] overlap offset={offset:+.6f} Nm median_abs_error={spread:.6f} Nm")
    print(f"[hybrid] q7 range={merged[0,0]:+.6f}..{merged[-1,0]:+.6f} rad knots={len(merged)}")
    print("[hybrid] production model was not modified")


if __name__ == "__main__":
    main()
