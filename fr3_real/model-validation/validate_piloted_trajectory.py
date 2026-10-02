#!/usr/bin/env python3
"""Validate and summarize one raw piloted FR3 trajectory without hardware access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_dir", type=Path)
    args = parser.parse_args()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    metadata_path = trajectory_dir / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    samples = read_jsonl(trajectory_dir / "trajectory.jsonl")
    errors = read_jsonl(trajectory_dir / "sample_errors.jsonl")
    events = read_jsonl(trajectory_dir / "events.jsonl")
    failures: list[str] = []
    if metadata.get("trajectory_schema_version") != 1:
        failures.append("unsupported trajectory schema")
    if not metadata.get("accepted_by_operator", False):
        failures.append("trajectory was not accepted by operator")
    if len(samples) < 2:
        failures.append("fewer than two robot samples")
    if errors:
        failures.append(f"{len(errors)} state sampling error(s)")

    if samples:
        times = np.asarray([row["monotonic_time_ns"] for row in samples], dtype=np.int64)
        q = np.asarray([row["q"] for row in samples], dtype=np.float64)
        dq = np.asarray([row["dq"] for row in samples], dtype=np.float64)
        xyz = np.asarray([row["translation_xyz_m"] for row in samples], dtype=np.float64)
        width = np.asarray(
            [row.get("gripper", {}).get("width_m", np.nan) for row in samples], dtype=np.float64
        )
        for name, array, shape_tail in (
            ("q", q, (7,)),
            ("dq", dq, (7,)),
            ("translation", xyz, (3,)),
        ):
            if array.ndim != 2 or array.shape[1:] != shape_tail or not np.isfinite(array).all():
                failures.append(f"{name} has invalid shape or non-finite values")
        gaps_s = np.diff(times) / 1e9 if len(times) > 1 else np.asarray([], dtype=float)
        if len(gaps_s) and np.any(gaps_s <= 0):
            failures.append("sample timestamps are not strictly increasing")
        duration_s = (times[-1] - times[0]) / 1e9 if len(times) > 1 else 0.0
        effective_hz = (len(samples) - 1) / duration_s if duration_s > 0 else 0.0

        print(f"trajectory:             {metadata.get('trajectory_name', trajectory_dir.name)}")
        print(f"task/prompt:            {metadata.get('task_name')} / {metadata.get('language_prompt')}")
        print(f"variation:              {metadata.get('bracket_slot')} / {metadata.get('bracket_grasp_style')}")
        print(f"operator accepted:      {metadata.get('accepted_by_operator')}")
        print(f"samples:                {len(samples)}")
        print(f"sample duration:        {duration_s:.3f} s")
        print(f"effective sample rate:  {effective_hz:.2f} Hz")
        print(f"maximum timestamp gap:  {float(gaps_s.max()) if len(gaps_s) else 0.0:.4f} s")
        print(f"q min:                  {np.array2string(q.min(axis=0), precision=4)}")
        print(f"q max:                  {np.array2string(q.max(axis=0), precision=4)}")
        print(f"maximum measured |dq|:  {float(np.max(np.abs(dq))):.4f} rad/s")
        print(f"EE xyz min:             {np.array2string(xyz.min(axis=0), precision=4)}")
        print(f"EE xyz max:             {np.array2string(xyz.max(axis=0), precision=4)}")
        if np.isfinite(width).any():
            print(f"gripper width range:    {float(np.nanmin(width)):.4f} .. {float(np.nanmax(width)):.4f} m")
        elif metadata.get("gripper_event_mode") == "ordered_semantic_markers":
            print("gripper state:          not sampled by design; using semantic phase markers")
        else:
            failures.append("no finite gripper width samples")
    else:
        print(f"trajectory:             {metadata.get('trajectory_name', trajectory_dir.name)}")
        print("samples:                0")

    print(f"sample errors:          {len(errors)}")
    event_names = [row.get("event") for row in events]
    required_events = [
        "trajectory_started",
        "grasp_pose_reached",
        "gripper_closed",
        "place_pose_reached",
        "gripper_opened",
        "retreat_complete",
    ]
    print(f"events:                 {event_names}")
    if event_names != required_events:
        failures.append(f"event sequence mismatch; expected {required_events}")
    elif samples:
        origin = int(samples[0]["monotonic_time_ns"])
        event_elapsed = [
            (row["event"], (int(row["monotonic_time_ns"]) - origin) / 1e9) for row in events
        ]
        print("event elapsed seconds:  " + ", ".join(f"{name}={elapsed:.2f}" for name, elapsed in event_elapsed))
    print(f"replay commissioned:    {metadata.get('replay_commissioned', False)}")
    if failures:
        print("\nRAW CAPTURE INVALID:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nRAW CAPTURE VALID")
    print("REPLAY NOT COMMISSIONED: no motion command should be run from this result.")


if __name__ == "__main__":
    main()
