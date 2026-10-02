#!/usr/bin/env python3
"""Offline validation of one raw piloted bolt pickup-through-PREINSERT capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_preinsert import (  # noqa: E402
    EXPECTED_BOLT_PREINSERT_EVENTS,
    bolt_capture_metadata_blockers,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_dir", type=Path)
    args = parser.parse_args()
    root = args.trajectory_dir.expanduser().resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text())
        samples = read_jsonl(root / "trajectory.jsonl")
        errors = read_jsonl(root / "sample_errors.jsonl")
        events = read_jsonl(root / "events.jsonl")
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read raw bolt capture: {error}") from error

    failures = bolt_capture_metadata_blockers(metadata)
    if not metadata.get("accepted_by_operator", False):
        failures.append("capture was not accepted by the operator")
    if errors:
        failures.append(f"{len(errors)} state sampling error(s)")
    if len(samples) < 2:
        failures.append("fewer than two arm samples")
    event_names = tuple(event.get("event") for event in events)
    if event_names != EXPECTED_BOLT_PREINSERT_EVENTS:
        failures.append(
            f"event sequence is {event_names}; expected {EXPECTED_BOLT_PREINSERT_EVENTS}"
        )

    print(f"trajectory:             {metadata.get('trajectory_name', root.name)}")
    print(f"task/prompt:            {metadata.get('task_name')} / {metadata.get('language_prompt')}")
    print(f"variation:              {metadata.get('screw_slot')} / {metadata.get('screw_grasp_style')}")
    print(f"operator accepted:      {metadata.get('accepted_by_operator')}")
    print(f"events:                 {list(event_names)}")
    print(f"sample errors:          {len(errors)}")
    if samples:
        times = np.asarray([row["monotonic_time_ns"] for row in samples], dtype=np.int64)
        q = np.asarray([row["q"] for row in samples], dtype=np.float64)
        dq = np.asarray([row["dq"] for row in samples], dtype=np.float64)
        xyz = np.asarray([row["translation_xyz_m"] for row in samples], dtype=np.float64)
        if q.shape != (len(samples), 7) or dq.shape != q.shape or xyz.shape != (len(samples), 3):
            failures.append("robot state arrays have invalid dimensions")
        if not all(np.isfinite(array).all() for array in (q, dq, xyz)):
            failures.append("robot state arrays contain NaN/Inf")
        gaps = np.diff(times) / 1e9
        if len(gaps) and np.any(gaps <= 0):
            failures.append("sample timestamps are not strictly increasing")
        duration = (times[-1] - times[0]) / 1e9
        rate = (len(samples) - 1) / duration if duration > 0 else 0.0
        print(f"samples/duration/rate:  {len(samples)} / {duration:.3f} s / {rate:.2f} Hz")
        print(f"maximum timestamp gap: {float(gaps.max()) if len(gaps) else 0.0:.4f} s")
        print(f"maximum measured |dq|: {float(np.max(np.abs(dq))):.4f} rad/s")
        print(f"EE xyz min:            {np.array2string(xyz.min(axis=0), precision=4)}")
        print(f"EE xyz max:            {np.array2string(xyz.max(axis=0), precision=4)}")
        for signal in ("tau_J", "tau_ext_hat_filtered", "O_F_ext_hat_K", "K_F_ext_hat_K"):
            present = sum(signal in sample for sample in samples)
            print(f"{signal} samples:       {present}/{len(samples)}")
        if events:
            elapsed = [
                (event["event"], (int(event["monotonic_time_ns"]) - int(times[0])) / 1e9)
                for event in events
            ]
            print("event elapsed seconds: " + ", ".join(f"{name}={value:.2f}" for name, value in elapsed))

    print("contact/threading:      excluded and uncommissioned")
    if failures:
        print("\nRAW BOLT PREINSERT CAPTURE INVALID:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nRAW BOLT PREINSERT CAPTURE VALID")
    print("REPLAY NOT COMMISSIONED: do not run contact or threading from this capture.")


if __name__ == "__main__":
    main()
