#!/usr/bin/env python3
"""Offline validation of a piloted bolt drop-and-head-regrasp capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_drop_regrasp import (  # noqa: E402
    EXPECTED_BOLT_DROP_REGRASP_EVENTS,
    bolt_drop_regrasp_metadata_blockers,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    root = args.capture_dir.expanduser().resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text())
        samples = read_jsonl(root / "trajectory.jsonl")
        events = read_jsonl(root / "events.jsonl")
        errors = read_jsonl(root / "sample_errors.jsonl")
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read capture {root}: {error}") from error
    failures = bolt_drop_regrasp_metadata_blockers(metadata)
    if not metadata.get("accepted_by_operator"):
        failures.append("capture was not accepted by operator")
    if errors:
        failures.append(f"{len(errors)} state sampling error(s)")
    if len(samples) < 2:
        failures.append("fewer than two robot samples")
    event_names = tuple(event.get("event") for event in events)
    if event_names != EXPECTED_BOLT_DROP_REGRASP_EVENTS:
        failures.append(
            f"event sequence is {event_names}; expected {EXPECTED_BOLT_DROP_REGRASP_EVENTS}"
        )
    print(f"trajectory:             {metadata.get('trajectory_name', root.name)}")
    print(f"normalization:          {metadata.get('mechanical_normalization_method')}")
    print(f"accepted/upright/held:  {metadata.get('accepted_by_operator')} / "
          f"{metadata.get('bolt_upright_at_endpoint')} / {metadata.get('bolt_regrasped_at_endpoint')}")
    print(f"events:                 {list(event_names)}")
    print(f"sample errors:          {len(errors)}")
    if samples:
        times = np.asarray([sample["monotonic_time_ns"] for sample in samples], dtype=np.int64)
        dq = np.asarray([sample["dq"] for sample in samples], dtype=np.float64)
        xyz = np.asarray([sample["translation_xyz_m"] for sample in samples], dtype=np.float64)
        duration_s = float((times[-1] - times[0]) / 1e9)
        rate = (len(samples) - 1) / duration_s if duration_s > 0 else 0.0
        print(f"samples/duration/rate:  {len(samples)} / {duration_s:.3f} s / {rate:.2f} Hz")
        print(f"maximum measured |dq|: {float(np.max(np.abs(dq))):.4f} rad/s")
        print(f"EE xyz min/max:         {xyz.min(axis=0)} / {xyz.max(axis=0)}")
        for signal in ("tau_J", "tau_ext_hat_filtered", "O_F_ext_hat_K", "K_F_ext_hat_K"):
            print(f"{signal} samples:       {sum(signal in sample for sample in samples)}/{len(samples)}")
    print("rotation/threading:     excluded and uncommissioned")
    if failures:
        print("\nRAW DROP/REGRASP CAPTURE INVALID:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nRAW DROP/REGRASP CAPTURE VALID")
    print("REPLAY NOT COMMISSIONED: no automatic drop, regrasp, or threading command is enabled.")


if __name__ == "__main__":
    main()
