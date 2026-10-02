#!/usr/bin/env python3
"""Trim idle tails from accepted recorded episodes after a named event.

Usage from ``fr3_real/``::

    python data-pipeline/trim_episode_idle_tails.py --episode B60_DIR B61_DIR --apply

The default is a dry run. With ``--apply``, robot state, actions, camera
timestamps, camera videos, and ``episode_arrays.npz`` are cut at
``training_motion_completed``. The accepted-success metadata is retained and
the original cut event remains the final trajectory boundary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve()


def event_time(episode: Path, name: str) -> int:
    for line in (episode / "events.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == name:
            return int(row["time_ns"])
    raise ValueError(f"{episode.name}: missing event {name!r}")


def trim_jsonl(path: Path, cutoff: int, key: str = "time_ns") -> int:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row[key]) <= cutoff:
            rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return len(rows)


def trim_npz(path: Path, cutoff: int) -> None:
    source = np.load(path, allow_pickle=True)
    state_mask = source["state_time_ns"] <= cutoff
    action_mask = source["action_time_ns"] <= cutoff
    arrays = {}
    for key in source.files:
        value = source[key]
        if key.startswith("state_") and len(value) == len(state_mask):
            arrays[key] = value[state_mask]
        elif key.startswith("action_") and len(value) == len(action_mask):
            arrays[key] = value[action_mask]
        else:
            arrays[key] = value
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def trim_video(path: Path, first_camera_ns: int, cutoff: int) -> None:
    duration = (cutoff - first_camera_ns) / 1e9
    if duration <= 0:
        raise ValueError(f"{path.name}: cutoff precedes first camera frame")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temporary = Path(handle.name)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
        "-t", f"{duration:.6f}", "-c", "copy", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def trim_episode(episode: Path, *, apply: bool) -> None:
    cutoff = event_time(episode, "training_motion_completed")
    state_rows = [json.loads(line) for line in (episode / "robot_state.jsonl").read_text().splitlines() if line.strip()]
    camera_files = sorted(episode.glob("camera_*_timestamps.jsonl"))
    tail_s = (int(state_rows[-1]["time_ns"]) - cutoff) / 1e9
    print(f"{episode.name}: cut={cutoff}, state_tail={tail_s:.3f}s, apply={apply}")
    if not apply:
        return

    trim_jsonl(episode / "robot_state.jsonl", cutoff)
    if (episode / "actions.jsonl").exists():
        trim_jsonl(episode / "actions.jsonl", cutoff)
    for timestamps in camera_files:
        rows = [json.loads(line) for line in timestamps.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        first_camera_ns = int(rows[0]["time_ns"])
        trim_jsonl(timestamps, cutoff)
        video = episode / timestamps.name.replace("_timestamps.jsonl", ".mp4")
        if video.exists():
            trim_video(video, first_camera_ns, cutoff)
    if (episode / "episode_arrays.npz").exists():
        trim_npz(episode / "episode_arrays.npz", cutoff)

    events = []
    for line in (episode / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["time_ns"]) <= cutoff:
            events.append(row)
    (episode / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "success": True,
            "accepted_by_operator": True,
            "bc_training_eligible": True,
            "idle_tail_trimmed": True,
            "trimmed_at_event": "training_motion_completed",
            "trimmed_at_time_ns": cutoff,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", nargs="+", required=True, type=resolve)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for episode in args.episode:
        if not episode.is_dir():
            raise SystemExit(f"missing episode directory: {episode}")
        trim_episode(episode, apply=args.apply)


if __name__ == "__main__":
    main()
