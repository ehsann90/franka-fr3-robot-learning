#!/usr/bin/env python3
"""Solve a read-only IK preview of the bracket path with a 180-degree tool roll."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAJECTORY = (
    FR3_ROOT
    / "recordings"
    / "piloted_bracket_trajectories_v1"
    / "20260812_115247_bracket_pilot_0004_B1_CENTER"
)
DEFAULT_BINARY = (
    FR3_ROOT / "model-running" / "streaming" / "build" / "fr3_bracket_outer_ik_preview"
)
DEFAULT_OUTPUT_ROOT = FR3_ROOT / "recordings" / "bracket_outer_ik_preview_v1"
DEFAULT_MANUAL_OUTPUT_ROOT = FR3_ROOT / "recordings" / "bracket_manual_replay_ik_preview_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nearest_index(times: np.ndarray, query: int) -> int:
    return int(np.argmin(np.abs(times - int(query))))


def selected_samples(trajectory_dir: Path) -> list[dict]:
    samples = read_jsonl(trajectory_dir / "trajectory.jsonl")
    events = read_jsonl(trajectory_dir / "events.jsonl")
    event_times = {row["event"]: int(row["monotonic_time_ns"]) for row in events}
    if "trajectory_started" not in event_times or "retreat_complete" not in event_times:
        raise ValueError("source trajectory is missing complete bracket event boundaries")
    times = np.asarray([row["monotonic_time_ns"] for row in samples], dtype=np.int64)
    start = nearest_index(times, event_times["trajectory_started"])
    stop = nearest_index(times, event_times["retreat_complete"])
    if stop <= start:
        raise ValueError("source bracket event range is not time ordered")
    return samples[start : stop + 1]


def selected_manual_samples(episode_dir: Path) -> list[dict]:
    metadata = json.loads((episode_dir / "metadata.json").read_text())
    if metadata.get("task_name") != "bracket" or metadata.get("success") is not True:
        raise ValueError("manual source must be a successful bracket episode")
    if metadata.get("accepted_by_operator") is not True:
        raise ValueError("manual source was not accepted by the operator")
    samples = read_jsonl(episode_dir / "robot_state.jsonl")
    events = read_jsonl(episode_dir / "events.jsonl")
    event_times = {row["event"]: int(row["time_ns"]) for row in events}
    required = ("manual_demo_started", "training_motion_completed")
    if any(name not in event_times for name in required):
        raise ValueError("manual source is missing replay boundaries")
    times = np.asarray([row["time_ns"] for row in samples], dtype=np.int64)
    start = nearest_index(times, event_times[required[0]])
    stop = nearest_index(times, event_times[required[1]])
    if stop <= start:
        raise ValueError("manual source replay range is not time ordered")
    return samples[start : stop + 1]


def write_targets(path: Path, samples: list[dict]) -> None:
    # +pi and -pi are the same target orientation. The numerical solver is
    # seeded toward the reverse/opposite-tightening joint branch.
    reverse_roll = np.asarray(
        [
            [math.cos(math.pi), -math.sin(math.pi), 0.0],
            [math.sin(math.pi), math.cos(math.pi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    start_ns = int(samples[0]["monotonic_time_ns"])
    header = ["time_s"] + [f"source_q{joint + 1}" for joint in range(7)]
    header += [f"T{row}{column}" for row in range(4) for column in range(4)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for sample in samples:
            transform = np.asarray(sample["T_base_ee"], dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("source trajectory contains an invalid T_base_ee")
            target = transform.copy()
            target[:3, :3] = transform[:3, :3] @ reverse_roll
            writer.writerow(
                [(int(sample["monotonic_time_ns"]) - start_ns) / 1e9]
                + list(map(float, sample["q"]))
                + target.reshape(-1).tolist()
            )


def write_manual_targets(path: Path, samples: list[dict]) -> None:
    start_ns = int(samples[0]["time_ns"])
    header = ["time_s"] + [f"source_q{joint + 1}" for joint in range(7)]
    header += [f"T{row}{column}" for row in range(4) for column in range(4)]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for sample in samples:
            transform = np.asarray(sample["O_T_EE"], dtype=np.float64).reshape(4, 4)
            if not np.isfinite(transform).all():
                raise ValueError("manual source contains an invalid O_T_EE")
            writer.writerow(
                [(int(sample["time_ns"]) - start_ns) / 1e9]
                + list(map(float, sample["q"]))
                + transform.reshape(-1).tolist()
            )


def load_candidate(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True)
    time_s = np.asarray(values["time_s"], dtype=np.float64)
    q = np.column_stack([values[f"q{joint + 1}"] for joint in range(7)])
    position_error = np.asarray(values["position_error_m"], dtype=np.float64)
    orientation_error = np.asarray(values["orientation_error_rad"], dtype=np.float64)
    return time_s, q, position_error, orientation_error


def manual_stage_offsets(episode_dir: Path, source_start_ns: int) -> dict[str, list[float]]:
    events = read_jsonl(episode_dir / "events.jsonl")
    actions = read_jsonl(episode_dir / "actions.jsonl")
    timestamps = {row["event"]: int(row["time_ns"]) for row in events}
    timestamps.update({row["action"]: int(row["time_ns"]) for row in actions})
    boundaries = {
        "approach_open": ("manual_demo_started", "bracket_grasp_pose_reached"),
        "transport_place_closed": ("bracket_gripper_closed", "bracket_place_pose_reached"),
        "retreat_open": ("bracket_gripper_opened", "training_motion_completed"),
    }
    missing = sorted({name for pair in boundaries.values() for name in pair} - timestamps.keys())
    if missing:
        raise ValueError(f"manual source is missing stage markers: {missing}")
    return {
        stage: [
            (timestamps[start] - source_start_ns) / 1e9,
            (timestamps[stop] - source_start_ns) / 1e9,
        ]
        for stage, (start, stop) in boundaries.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only-preview", action="store_true", required=True)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument(
        "--manual-episode",
        type=Path,
        help="successful manual bracket episode whose exact recorded TCP poses are preserved",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    manual_episode = (
        args.manual_episode.expanduser().resolve() if args.manual_episode is not None else None
    )
    binary = args.binary.expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"missing read-only IK binary: {binary}")
    try:
        samples = (
            selected_manual_samples(manual_episode)
            if manual_episode is not None
            else selected_samples(trajectory_dir)
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid source bracket trajectory: {error}") from error

    print("=== BRACKET OUTER-WRIST PATH PREPARATION ===")
    print("mode:                     READ_ONLY_IK_PREVIEW")
    print(f"robot IP:                 {args.robot_ip}")
    print(
        f"source {'manual episode' if manual_episode is not None else 'trajectory'}:        "
        f"{manual_episode if manual_episode is not None else trajectory_dir}"
    )
    print(f"source samples:           {len(samples)}")
    print("Cartesian positions:      EXACTLY PRESERVED TARGETS")
    print(
        "tool orientation:         EXACTLY RECORDED; redundant joints rebalanced"
        if manual_episode is not None
        else "tool orientation:         180 deg roll, reverse branch preferred"
    )
    print("arm/Hand commands:        DISABLED BY DESIGN")
    print("collision settings:       UNCHANGED")
    print("execution output:         DISABLED; candidate is not commissioned")

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root != DEFAULT_OUTPUT_ROOT or manual_episode is None
        else DEFAULT_MANUAL_OUTPUT_ROOT.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate_label = (
        "bracket_manual_exact_pose_rebalanced_ik_candidate"
        if manual_episode is not None
        else "bracket_outer_reverse_ik_candidate"
    )
    candidate_path = output_root / f"{stamp}_{candidate_label}.csv"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
        target_path = Path(temporary.name)
    try:
        if manual_episode is not None:
            write_manual_targets(target_path, samples)
        else:
            write_targets(target_path, samples)
        result = subprocess.run(
            [str(binary), args.robot_ip, str(target_path), str(candidate_path)],
            check=False,
        )
    finally:
        target_path.unlink(missing_ok=True)
    if result.returncode != 0:
        candidate_path.unlink(missing_ok=True)
        raise SystemExit("[BLOCKED] no usable continuous outer-wrist IK candidate was produced")

    time_s, q, position_error, orientation_error = load_candidate(candidate_path)
    dt = np.diff(time_s)
    dq = np.diff(q, axis=0) / dt[:, None]
    print("\nCandidate audit:")
    print(f"  duration:                {time_s[-1]:.3f} s")
    print(f"  q7 range:                {q[:, 6].min():+.6f} .. {q[:, 6].max():+.6f} rad")
    print(f"  maximum sampled |dq|:    {np.max(np.abs(dq)):.6f} rad/s")
    print(f"  maximum position error:  {position_error.max():.9f} m")
    print(f"  maximum rotation error:  {orientation_error.max():.9f} rad")
    print(f"[saved] {candidate_path}")
    if manual_episode is not None:
        sidecar = candidate_path.with_suffix(".metadata.json")
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate_kind": "manual_bracket_exact_pose_joint_rebalance",
                    "source_episode": str(manual_episode),
                    "source_start_time_ns": int(samples[0]["time_ns"]),
                    "stage_offsets_s": manual_stage_offsets(
                        manual_episode, int(samples[0]["time_ns"])
                    ),
                    "sample_count": int(len(time_s)),
                    "duration_s": float(time_s[-1]),
                    "q_min": q.min(axis=0).tolist(),
                    "q_max": q.max(axis=0).tolist(),
                    "maximum_position_error_m": float(position_error.max()),
                    "maximum_orientation_error_rad": float(orientation_error.max()),
                    "execution_enabled": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"[saved] {sidecar}")
    print(
        "Preview complete. This CSV is an IK candidate only; no executable trajectory "
        "or robot command was created."
    )


if __name__ == "__main__":
    main()
