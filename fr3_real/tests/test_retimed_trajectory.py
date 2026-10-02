from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_demo.replay_plan import ReplayStagePlan
from expert_demo.retimed_trajectory import (
    build_retimed_trajectory,
    write_retimed_csv,
)


def synthetic_plan() -> ReplayStagePlan:
    times = np.asarray([0.0, 0.4, 1.0, 1.7])
    q = np.zeros((4, 7))
    q[:, 0] = [0.0, 0.2, -0.1, 0.3]
    q[:, 6] = [0.4, 0.5, 0.7, 0.6]
    return ReplayStagePlan(
        trajectory_name="synthetic",
        name="stage",
        start_event="start",
        stop_event="stop",
        source_start_index=0,
        source_stop_index=3,
        source_duration_s=1.7,
        command_rate_hz=10.0,
        command_times_s=times,
        q_targets=q,
        dq_targets=np.zeros_like(q),
        xyz_targets=np.zeros((4, 3)),
        expected_start_q=q[0],
        expected_stop_q=q[-1],
        source_xyz_min=np.zeros(3),
        source_xyz_max=np.zeros(3),
        max_source_abs_dq_rad_s=0.0,
        max_target_step_rad=0.4,
        implied_peak_velocity_rad_s=1.0,
    )


def idle_padded_plan() -> ReplayStagePlan:
    times = np.arange(0.0, 1.8, 0.2)
    q = np.zeros((len(times), 7))
    q[:, 0] = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3]
    return ReplayStagePlan(
        trajectory_name="idle_padded",
        name="stage",
        start_event="start",
        stop_event="stop",
        source_start_index=0,
        source_stop_index=len(times) - 1,
        source_duration_s=float(times[-1]),
        command_rate_hz=5.0,
        command_times_s=times,
        q_targets=q,
        dq_targets=np.zeros_like(q),
        xyz_targets=np.zeros((len(times), 3)),
        expected_start_q=q[0],
        expected_stop_q=q[-1],
        source_xyz_min=np.zeros(3),
        source_xyz_max=np.zeros(3),
        max_source_abs_dq_rad_s=0.0,
        max_target_step_rad=0.1,
        implied_peak_velocity_rad_s=0.5,
    )


class RetimedTrajectoryTests(unittest.TestCase):
    def test_clamped_spline_preserves_knots_and_zero_endpoint_velocity(self) -> None:
        source = synthetic_plan()
        trajectory = build_retimed_trajectory(
            source, playback_time_scale=2.0, validation_rate_hz=1000.0
        )
        np.testing.assert_allclose(
            trajectory.evaluate(source.command_times_s * 2.0), source.q_targets,
            atol=1e-12,
        )
        np.testing.assert_allclose(trajectory.evaluate(0.0, 1), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            trajectory.evaluate(trajectory.duration_s, 1), 0.0, atol=1e-12
        )
        self.assertAlmostEqual(trajectory.duration_s, 3.4)

    def test_position_velocity_and_acceleration_are_continuous(self) -> None:
        trajectory = build_retimed_trajectory(
            synthetic_plan(), playback_time_scale=2.0
        )
        for knot in trajectory.knot_times_s[1:-1]:
            epsilon = 1e-8
            for derivative in (0, 1, 2):
                np.testing.assert_allclose(
                    trajectory.evaluate(knot - epsilon, derivative),
                    trajectory.evaluate(knot + epsilon, derivative),
                    atol=1e-6,
                )

    def test_csv_has_one_row_per_segment(self) -> None:
        trajectory = build_retimed_trajectory(
            synthetic_plan(), playback_time_scale=2.0
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.csv"
            write_retimed_csv(path, trajectory)
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(len(rows), len(trajectory.coefficients) + 1)
        self.assertEqual(len(rows[0]), 30)
        self.assertEqual(rows[0][:2], ["t0_s", "t1_s"])

    def test_idle_trim_removes_only_stationary_edges_and_keeps_buffer(self) -> None:
        source = idle_padded_plan()
        trajectory = build_retimed_trajectory(
            source,
            playback_time_scale=2.0,
            idle_trim_velocity_threshold_rad_s=0.01,
            idle_trim_buffer_s=0.2,
        )
        self.assertEqual(trajectory.source_start_knot_index, 1)
        self.assertEqual(trajectory.source_stop_knot_index, 6)
        self.assertAlmostEqual(trajectory.source_trim_start_s, 0.2)
        self.assertAlmostEqual(trajectory.source_trim_end_s, 0.4)
        self.assertAlmostEqual(trajectory.trimmed_source_duration_s, 1.0)
        self.assertAlmostEqual(trajectory.duration_s, 2.0)
        np.testing.assert_allclose(
            trajectory.evaluate(trajectory.knot_times_s),
            source.q_targets[1:7],
            atol=1e-12,
        )
        np.testing.assert_allclose(trajectory.evaluate(0.0, 1), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            trajectory.evaluate(trajectory.duration_s, 1), 0.0, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
