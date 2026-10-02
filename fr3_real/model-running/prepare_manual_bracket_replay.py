#!/usr/bin/env python3
"""Offline-only staging of a manually recorded bracket episode's IK candidate.

The existing ``expert_demo/replay_plan.py`` stage builder only understands the
camera-free piloted trajectory format (``trajectory.jsonl`` + the
``trajectory_started``/``grasp_pose_reached``/... event names). Manually
recorded bracket episodes use a different raw layout (``robot_state.jsonl`` +
``manual_demo_started``/``bracket_grasp_pose_reached``/... event names) and,
for an outer/rebalanced capture, an IK candidate CSV produced separately by
``prepare_bracket_outer_ik_preview.py``. This script bridges that gap: it
slices the candidate's joint path into the three replay stages using the
stage offsets already computed by the IK preview's metadata sidecar, then
reuses the same offline clamped-cubic retiming and velocity/acceleration/jerk
audit that the piloted pipeline uses, so both sources are held to the same
safety bar before anything is ever sent to the robot.

Purely offline math and file reads. No franky/libfranka import, no robot
connection, no control/Hand/collision command of any kind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.replay_plan import ReplayStagePlan  # noqa: E402
from expert_demo.retimed_trajectory import build_retimed_trajectory  # noqa: E402

STAGE_MANUAL_EVENTS = {
    "approach_open": ("manual_demo_started", "bracket_grasp_pose_reached"),
    "transport_place_closed": ("bracket_gripper_closed", "bracket_place_pose_reached"),
    "retreat_open": ("bracket_gripper_opened", "training_motion_completed"),
}

DEFAULT_CAPS_CONFIG = FR3_ROOT / "configs" / "full_assembly_2x_commissioning_v1.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_candidate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.genfromtxt(path, delimiter=",", names=True)
    time_s = np.asarray(values["time_s"], dtype=np.float64)
    q = np.column_stack([values[f"q{joint + 1}"] for joint in range(7)])
    return time_s, q


def source_trimmed_samples(episode_dir: Path) -> list[dict]:
    """Mirrors selected_manual_samples() in prepare_bracket_outer_ik_preview.py."""
    events = read_jsonl(episode_dir / "events.jsonl")
    event_times = {row["event"]: int(row["time_ns"]) for row in events}
    samples = read_jsonl(episode_dir / "robot_state.jsonl")
    times = np.asarray([row["time_ns"] for row in samples], dtype=np.int64)
    start = int(np.argmin(np.abs(times - event_times["manual_demo_started"])))
    stop = int(np.argmin(np.abs(times - event_times["training_motion_completed"])))
    return samples[start : stop + 1]


def build_manual_stage_plan(
    candidate_path: Path,
    stage: str,
    *,
    command_rate_hz: float,
) -> ReplayStagePlan:
    sidecar_path = candidate_path.with_suffix(".metadata.json")
    sidecar = json.loads(sidecar_path.read_text())
    source_episode = Path(sidecar["source_episode"])
    stage_offsets_s = sidecar["stage_offsets_s"]
    if stage not in stage_offsets_s:
        raise ValueError(f"unknown stage {stage!r}; choices={sorted(stage_offsets_s)}")

    time_s, q = load_candidate(candidate_path)
    samples = source_trimmed_samples(source_episode)
    if len(samples) != len(time_s):
        raise ValueError(
            f"candidate row count {len(time_s)} does not match the source trimmed "
            f"sample count {len(samples)}; candidate and episode are out of sync"
        )
    if "target_poses_csv" in sidecar:
        target_poses_path = Path(sidecar["target_poses_csv"])
        target_poses = np.genfromtxt(target_poses_path, delimiter=",", names=True)
        target_times = np.asarray(target_poses["time_s"], dtype=np.float64)
        if len(target_times) != len(time_s) or np.max(np.abs(target_times - time_s)) > 1e-6:
            raise ValueError("candidate target poses and solved joint samples are out of sync")
        xyz = np.column_stack([target_poses[f"T{axis}3"] for axis in range(3)])
        dq = np.gradient(q, time_s, axis=0)
    else:
        dq = np.asarray([sample["dq"] for sample in samples], dtype=np.float64)
        xyz = np.asarray([sample["ee_translation"] for sample in samples], dtype=np.float64)

    start_s, stop_s = stage_offsets_s[stage]
    start_index = int(np.searchsorted(time_s, start_s, side="left"))
    stop_index = int(np.searchsorted(time_s, stop_s, side="right")) - 1
    if stop_index <= start_index:
        raise ValueError(f"stage {stage} has a non-positive sample range")

    segment_time_s = time_s[start_index : stop_index + 1] - time_s[start_index]
    duration_s = float(segment_time_s[-1])
    command_times_s = np.arange(0.0, duration_s, 1.0 / command_rate_hz)
    if not len(command_times_s) or command_times_s[-1] < duration_s:
        command_times_s = np.append(command_times_s, duration_s)
    segment_q = q[start_index : stop_index + 1]
    segment_dq = dq[start_index : stop_index + 1]
    segment_xyz = xyz[start_index : stop_index + 1]
    q_targets = np.column_stack(
        [np.interp(command_times_s, segment_time_s, segment_q[:, joint]) for joint in range(7)]
    )
    dq_targets = np.column_stack(
        [np.interp(command_times_s, segment_time_s, segment_dq[:, joint]) for joint in range(7)]
    )
    xyz_targets = np.column_stack(
        [np.interp(command_times_s, segment_time_s, segment_xyz[:, axis]) for axis in range(3)]
    )
    target_steps = np.diff(q_targets, axis=0)
    max_target_step = float(np.max(np.abs(target_steps))) if len(target_steps) else 0.0
    start_event, stop_event = STAGE_MANUAL_EVENTS[stage]

    return ReplayStagePlan(
        trajectory_name=source_episode.name,
        name=stage,
        start_event=start_event,
        stop_event=stop_event,
        source_start_index=start_index,
        source_stop_index=stop_index,
        source_duration_s=duration_s,
        command_rate_hz=float(command_rate_hz),
        command_times_s=command_times_s,
        q_targets=q_targets,
        dq_targets=dq_targets,
        xyz_targets=xyz_targets,
        expected_start_q=q_targets[0].copy(),
        expected_stop_q=q_targets[-1].copy(),
        source_xyz_min=segment_xyz.min(axis=0),
        source_xyz_max=segment_xyz.max(axis=0),
        max_source_abs_dq_rad_s=float(np.max(np.abs(segment_dq))),
        max_target_step_rad=max_target_step,
        implied_peak_velocity_rad_s=max_target_step * command_rate_hz,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGE_MANUAL_EVENTS), default="approach_open")
    parser.add_argument("--command-rate-hz", type=float, default=15.0)
    parser.add_argument("--playback-time-scale", type=float, default=2.0)
    parser.add_argument("--caps-config", type=Path, default=DEFAULT_CAPS_CONFIG)
    args = parser.parse_args()

    candidate_path = args.candidate.expanduser().resolve()
    caps = json.loads(args.caps_config.expanduser().resolve().read_text())

    plan = build_manual_stage_plan(
        candidate_path, args.stage, command_rate_hz=args.command_rate_hz
    )
    retimed = build_retimed_trajectory(plan, playback_time_scale=args.playback_time_scale)

    print("=== MANUAL BRACKET REPLAY STAGE PREVIEW ===")
    print("mode:                   DRY_RUN / OFFLINE_ONLY")
    print(f"candidate:              {candidate_path}")
    print(f"stage:                  {plan.name}")
    print(f"event range:            {plan.start_event} -> {plan.stop_event}")
    print(f"source duration:        {plan.source_duration_s:.3f} s")
    print(f"playback time scale:    {args.playback_time_scale:.2f}x")
    print(f"retimed duration:       {retimed.duration_s:.3f} s")
    print(f"expected start q:       {np.array2string(plan.expected_start_q, precision=4)}")
    print(f"expected stop q:        {np.array2string(plan.expected_stop_q, precision=4)}")
    print(f"source EE xyz min:      {np.array2string(plan.source_xyz_min, precision=4)}")
    print(f"source EE xyz max:      {np.array2string(plan.source_xyz_max, precision=4)}")
    print(f"peak velocity:          {retimed.peak_abs_velocity_rad_s:.4f} rad/s")
    print(f"peak acceleration:      {retimed.peak_abs_acceleration_rad_s2:.4f} rad/s^2")
    print(f"peak jerk:              {retimed.peak_abs_jerk_rad_s3:.4f} rad/s^3")

    print("\nCap comparison (reference caps from "
          f"{args.caps_config.name}, not yet a calibrated bound for this candidate):")
    checks = (
        ("velocity", retimed.peak_abs_velocity_rad_s, caps["velocity_cap_rad_s"]),
        ("acceleration", retimed.peak_abs_acceleration_rad_s2, caps["acceleration_cap_rad_s2"]),
        ("jerk", retimed.peak_abs_jerk_rad_s3, caps["jerk_cap_rad_s3"]),
    )
    blockers = []
    for label, measured, cap in checks:
        status = "OK" if measured <= float(cap) else "EXCEEDS CAP"
        print(f"  {label:12s} {measured:10.4f}  <=  {float(cap):10.4f}   [{status}]")
        if measured > float(cap):
            blockers.append(f"{label} exceeds reference cap")

    workspace = caps.get("reset_workspace_bounds_m")
    if isinstance(workspace, dict):
        minimum = np.asarray(workspace["min_xyz"], dtype=np.float64)
        maximum = np.asarray(workspace["max_xyz"], dtype=np.float64)
        within = bool(np.all(plan.source_xyz_min >= minimum) and np.all(plan.source_xyz_max <= maximum))
        print(f"\nworkspace envelope [{minimum.tolist()}, {maximum.tolist()}]: "
              f"{'within' if within else 'OUTSIDE'} (reference bound, not yet stage-specific)")
        if not within:
            blockers.append("stage EE path leaves the reference workspace envelope")

    print("\nExecution blockers:")
    if blockers:
        for blocker in blockers:
            print(f"  - {blocker}")
    else:
        print("  (none against the reference caps above)")
    print(
        "\nThis is still only an offline preview: no execution-ready stage config, "
        "controller binary, or live-start check exists for this candidate yet. "
        "No Franky/libfranka import or hardware connection occurred."
    )


if __name__ == "__main__":
    main()
