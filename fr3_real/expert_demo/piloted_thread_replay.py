"""Extract deterministic joint replay targets from a successful Pilot bolt drive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class PilotedThreadDrive:
    cycle: int
    q_targets: np.ndarray
    dq_targets: np.ndarray
    xyz_targets: np.ndarray
    pose_targets: np.ndarray
    source_duration_s: float
    command_rate_hz: float
    source_start_time_ns: int
    source_stop_time_ns: int

    @property
    def expected_start_q(self) -> np.ndarray:
        return self.q_targets[0]

    @property
    def expected_stop_q(self) -> np.ndarray:
        return self.q_targets[-1]

    @property
    def maximum_target_step_rad(self) -> float:
        return float(np.max(np.abs(np.diff(self.q_targets, axis=0))))


def _read_events(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_piloted_thread_drive(
    episode_dir: str | Path,
    *,
    cycle: int,
    command_rate_hz: float,
    maximum_q7_rad: float,
) -> PilotedThreadDrive:
    if cycle < 1:
        raise ValueError("cycle must be positive")
    if not np.isfinite(command_rate_hz) or command_rate_hz <= 0.0:
        raise ValueError("command rate must be finite and positive")
    root = Path(episode_dir).expanduser().resolve()
    events = _read_events(root / "events.jsonl")
    starts = [
        row for row in events
        if row.get("event") == "thread_drive_started"
        and int(row.get("payload", {}).get("cycle", -1)) == cycle
    ]
    stops = [
        row for row in events
        if row.get("event") == "thread_drive_complete"
        and int(row.get("payload", {}).get("cycle", -1)) == cycle
    ]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError(f"cycle {cycle} does not have exactly one start and stop event")
    start_ns = int(starts[0]["time_ns"])
    event_stop_ns = int(stops[0]["time_ns"])
    if event_stop_ns <= start_ns:
        raise ValueError("drive event interval is invalid")

    arrays = np.load(root / "episode_arrays.npz")
    times = np.asarray(arrays["state_time_ns"], dtype=np.int64)
    q = np.asarray(arrays["state_q"], dtype=np.float64)
    dq = np.asarray(arrays["state_dq"], dtype=np.float64)
    xyz = np.asarray(arrays["state_ee_translation"], dtype=np.float64)
    poses = np.asarray(arrays["state_O_T_EE"], dtype=np.float64).reshape(-1, 4, 4)
    if q.shape != (len(times), 7) or dq.shape != q.shape or xyz.shape != (len(times), 3) or poses.shape != (len(times), 4, 4):
        raise ValueError("episode arrays have incompatible dimensions")
    indices = np.flatnonzero((times >= start_ns) & (times <= event_stop_ns))
    if len(indices) < 2:
        raise ValueError("drive has too few samples")
    source_times = times[indices]
    source_q = q[indices]
    source_dq = dq[indices]
    source_xyz = xyz[indices]
    source_poses = poses[indices]
    # Preserve the complete recorded phase, including its stationary ending.
    # Uniformly scale only q7's rotation so its maximum remains inside the
    # calibrated cable-safe target. Hard trimming previously ended the replay
    # while dq7 was still ~0.54 rad/s and caused a reflex at motion handoff.
    q7_start = float(source_q[0, 6])
    q7_peak = float(np.max(source_q[:, 6]))
    if q7_peak <= q7_start or maximum_q7_rad <= q7_start:
        raise ValueError("drive q7 interval cannot be safely scaled")
    q7_scale = min(1.0, (maximum_q7_rad - q7_start) / (q7_peak - q7_start))
    source_q = source_q.copy()
    source_dq = source_dq.copy()
    source_q[:, 6] = q7_start + (source_q[:, 6] - q7_start) * q7_scale
    source_dq[:, 6] *= q7_scale
    if not all(np.all(np.isfinite(value)) for value in (source_q, source_dq, source_xyz)):
        raise ValueError("drive contains NaN/Inf")
    if np.any(np.diff(source_times) <= 0):
        raise ValueError("drive timestamps are not strictly increasing")

    period_ns = 1e9 / command_rate_hz
    target_times = np.arange(source_times[0], source_times[-1] + 1.0, period_ns)
    if target_times[-1] < source_times[-1] - 0.25 * period_ns:
        target_times = np.append(target_times, float(source_times[-1]))
    target_times[-1] = float(source_times[-1])

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.interp(target_times, source_times, values[:, column])
             for column in range(values.shape[1])]
        )

    q_targets = interpolate(source_q)
    dq_targets = interpolate(source_dq)
    xyz_targets = interpolate(source_xyz)
    pose_flat_targets = interpolate(source_poses.reshape(len(source_poses), 16))
    pose_targets = pose_flat_targets.reshape(-1, 4, 4)
    # Element-wise interpolation introduces tiny rotation drift. Project each
    # 3x3 block back onto SO(3); translation remains the independently
    # interpolated source trajectory.
    for index, pose in enumerate(pose_targets):
        u, _, vt = np.linalg.svd(pose[:3, :3])
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        pose[:3, :3] = rotation
        pose[:3, 3] = xyz_targets[index]
        pose[3] = [0.0, 0.0, 0.0, 1.0]
    # The phase includes stationary samples at both boundaries. Make those
    # boundary conditions explicit for the motion generator so the handoff
    # cannot request a nonzero terminal velocity due to measurement noise.
    dq_targets[0] = 0.0
    dq_targets[-1] = 0.0
    return PilotedThreadDrive(
        cycle=cycle,
        q_targets=q_targets,
        dq_targets=dq_targets,
        xyz_targets=xyz_targets,
        pose_targets=pose_targets,
        source_duration_s=float((source_times[-1] - source_times[0]) / 1e9),
        command_rate_hz=float(command_rate_hz),
        source_start_time_ns=int(source_times[0]),
        source_stop_time_ns=int(source_times[-1]),
    )
