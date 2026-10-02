"""Smooth, deterministic retiming for commissioned joint-space paths.

The piloted recordings are sampled at 50 Hz and the legacy replay path is
reduced to 15 Hz waypoints.  Passing those sparse waypoints to a separate
Ruckig motion for every sample creates velocity discontinuities when the path
is sped up.  This module instead fits one clamped cubic spline through the
same joint positions.  The spline has zero velocity at both stage boundaries
and continuous velocity and acceleration everywhere inside the stage.

Only offline math and file serialization live here.  Robot execution is done
by the direct libfranka controller in ``model-running/streaming``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .replay_plan import ReplayStagePlan


@dataclass(frozen=True)
class RetimedTrajectory:
    """Piecewise cubic joint trajectory in local segment time."""

    playback_time_scale: float
    untrimmed_source_duration_s: float
    source_trim_start_s: float
    source_trim_end_s: float
    trimmed_source_duration_s: float
    source_start_knot_index: int
    source_stop_knot_index: int
    knot_times_s: np.ndarray
    # Shape: (segments, joints, [position, velocity, acceleration/2, jerk/6]).
    coefficients: np.ndarray
    duration_s: float
    expected_start_q: np.ndarray
    expected_stop_q: np.ndarray
    peak_abs_velocity_rad_s: float
    peak_abs_acceleration_rad_s2: float
    peak_abs_jerk_rad_s3: float

    def evaluate(self, times_s: np.ndarray | float, derivative: int = 0) -> np.ndarray:
        times = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
        if derivative not in (0, 1, 2, 3):
            raise ValueError("derivative must be 0, 1, 2, or 3")
        clipped = np.clip(times, 0.0, self.duration_s)
        indices = np.searchsorted(self.knot_times_s, clipped, side="right") - 1
        indices = np.clip(indices, 0, len(self.knot_times_s) - 2)
        local = clipped - self.knot_times_s[indices]
        c = self.coefficients[indices]
        if derivative == 0:
            values = c[:, :, 0] + local[:, None] * (
                c[:, :, 1] + local[:, None] * (
                    c[:, :, 2] + local[:, None] * c[:, :, 3]
                )
            )
        elif derivative == 1:
            values = c[:, :, 1] + local[:, None] * (
                2.0 * c[:, :, 2] + 3.0 * local[:, None] * c[:, :, 3]
            )
        elif derivative == 2:
            values = 2.0 * c[:, :, 2] + 6.0 * local[:, None] * c[:, :, 3]
        else:
            values = 6.0 * c[:, :, 3]
        return values[0] if np.asarray(times_s).ndim == 0 else values


def _clamped_second_derivatives(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve a clamped cubic spline with zero endpoint velocity.

    This is a vector-valued Thomas solve.  Avoiding a SciPy dependency keeps
    the production recorder usable in the existing ``fr3-recording`` env.
    """

    count, joints = values.shape
    h = np.diff(times)
    slopes = np.diff(values, axis=0) / h[:, None]
    lower = np.zeros(count, dtype=np.float64)
    diagonal = np.zeros(count, dtype=np.float64)
    upper = np.zeros(count, dtype=np.float64)
    rhs = np.zeros((count, joints), dtype=np.float64)

    diagonal[0] = 2.0 * h[0]
    upper[0] = h[0]
    rhs[0] = 6.0 * slopes[0]  # start velocity is exactly zero
    for index in range(1, count - 1):
        lower[index] = h[index - 1]
        diagonal[index] = 2.0 * (h[index - 1] + h[index])
        upper[index] = h[index]
        rhs[index] = 6.0 * (slopes[index] - slopes[index - 1])
    lower[-1] = h[-1]
    diagonal[-1] = 2.0 * h[-1]
    rhs[-1] = -6.0 * slopes[-1]  # end velocity is exactly zero

    for index in range(1, count):
        factor = lower[index] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        rhs[index] -= factor * rhs[index - 1]
    result = np.empty_like(rhs)
    result[-1] = rhs[-1] / diagonal[-1]
    for index in range(count - 2, -1, -1):
        result[index] = (
            rhs[index] - upper[index] * result[index + 1]
        ) / diagonal[index]
    return result


def build_retimed_trajectory(
    plan: ReplayStagePlan,
    *,
    playback_time_scale: float,
    validation_rate_hz: float = 1000.0,
    idle_trim_velocity_threshold_rad_s: float = 0.0,
    idle_trim_buffer_s: float = 0.0,
) -> RetimedTrajectory:
    """Retimes a commissioned stage, optionally removing stationary edge dwell.

    Edge trimming uses the maximum joint displacement between adjacent source
    command knots divided by their time interval.  Every knot between the first
    and last interval above the configured threshold is preserved, including
    all interior pauses.  A source-time buffer can retain a short stationary
    lead-in and tail around the detected motion.
    """

    if playback_time_scale <= 0.0 or validation_rate_hz <= 0.0:
        raise ValueError("playback scale and validation rate must be positive")
    if idle_trim_velocity_threshold_rad_s < 0.0 or idle_trim_buffer_s < 0.0:
        raise ValueError("idle trim threshold and buffer must be non-negative")
    source_times = np.asarray(plan.command_times_s, dtype=np.float64)
    source_values = np.asarray(plan.q_targets, dtype=np.float64)
    values = source_values
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise ValueError("joint targets must have shape (N, 7), N >= 2")
    if len(source_times) != len(values) or np.any(np.diff(source_times) <= 0.0):
        raise ValueError("retimed knot times must be strictly increasing")
    if not np.isfinite(source_times).all() or not np.isfinite(values).all():
        raise ValueError("retimed trajectory contains non-finite values")

    start_index = 0
    stop_index = len(source_times) - 1
    if idle_trim_velocity_threshold_rad_s > 0.0:
        interval_speeds = np.max(
            np.abs(np.diff(source_values, axis=0)) / np.diff(source_times)[:, None],
            axis=1,
        )
        moving = np.flatnonzero(interval_speeds > idle_trim_velocity_threshold_rad_s)
        if not len(moving):
            raise ValueError("idle trim found no moving source interval")
        buffer_intervals = int(np.ceil(idle_trim_buffer_s * plan.command_rate_hz))
        start_index = max(0, int(moving[0]) - buffer_intervals)
        stop_index = min(
            len(source_times) - 1,
            int(moving[-1]) + 1 + buffer_intervals,
        )
        source_times = source_times[start_index : stop_index + 1]
        values = source_values[start_index : stop_index + 1]

    source_trim_start_s = float(source_times[0])
    source_trim_end_s = float(plan.command_times_s[-1] - source_times[-1])
    trimmed_source_duration_s = float(source_times[-1] - source_times[0])
    times = (source_times - source_times[0]) * playback_time_scale

    h = np.diff(times)
    second = _clamped_second_derivatives(times, values)
    slopes = np.diff(values, axis=0) / h[:, None]
    coefficients = np.empty((len(h), 7, 4), dtype=np.float64)
    coefficients[:, :, 0] = values[:-1]
    coefficients[:, :, 1] = slopes - h[:, None] * (
        2.0 * second[:-1] + second[1:]
    ) / 6.0
    coefficients[:, :, 2] = second[:-1] / 2.0
    coefficients[:, :, 3] = (second[1:] - second[:-1]) / (6.0 * h[:, None])

    provisional = RetimedTrajectory(
        playback_time_scale=float(playback_time_scale),
        untrimmed_source_duration_s=float(plan.command_times_s[-1]),
        source_trim_start_s=source_trim_start_s,
        source_trim_end_s=source_trim_end_s,
        trimmed_source_duration_s=trimmed_source_duration_s,
        source_start_knot_index=start_index,
        source_stop_knot_index=stop_index,
        knot_times_s=times,
        coefficients=coefficients,
        duration_s=float(times[-1]),
        expected_start_q=values[0].copy(),
        expected_stop_q=values[-1].copy(),
        peak_abs_velocity_rad_s=0.0,
        peak_abs_acceleration_rad_s2=0.0,
        peak_abs_jerk_rad_s3=0.0,
    )
    sample_count = max(2, int(np.ceil(times[-1] * validation_rate_hz)) + 1)
    dense_times = np.linspace(0.0, times[-1], sample_count)
    velocity = provisional.evaluate(dense_times, 1)
    acceleration = provisional.evaluate(dense_times, 2)
    jerk = provisional.evaluate(dense_times, 3)
    return RetimedTrajectory(
        playback_time_scale=provisional.playback_time_scale,
        untrimmed_source_duration_s=provisional.untrimmed_source_duration_s,
        source_trim_start_s=provisional.source_trim_start_s,
        source_trim_end_s=provisional.source_trim_end_s,
        trimmed_source_duration_s=provisional.trimmed_source_duration_s,
        source_start_knot_index=provisional.source_start_knot_index,
        source_stop_knot_index=provisional.source_stop_knot_index,
        knot_times_s=provisional.knot_times_s,
        coefficients=provisional.coefficients,
        duration_s=provisional.duration_s,
        expected_start_q=provisional.expected_start_q,
        expected_stop_q=provisional.expected_stop_q,
        peak_abs_velocity_rad_s=float(np.max(np.abs(velocity))),
        peak_abs_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
        peak_abs_jerk_rad_s3=float(np.max(np.abs(jerk))),
    )


def write_retimed_csv(path: str | Path, trajectory: RetimedTrajectory) -> None:
    """Serialize cubic coefficients for the allocation-free 1 kHz controller."""

    path = Path(path)
    header = ["t0_s", "t1_s"]
    for joint in range(7):
        header.extend(
            [f"q{joint}_a", f"q{joint}_b", f"q{joint}_c", f"q{joint}_d"]
        )
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for index, coefficients in enumerate(trajectory.coefficients):
            row = [
                float(trajectory.knot_times_s[index]),
                float(trajectory.knot_times_s[index + 1]),
            ]
            row.extend(float(value) for value in coefficients.reshape(-1))
            writer.writerow(row)
