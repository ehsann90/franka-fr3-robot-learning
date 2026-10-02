"""Synthetic tests for offline piloted trajectory replay staging."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.replay_plan import (  # noqa: E402
    EXPECTED_EVENTS,
    ReplayPlanError,
    build_replay_stage,
    replay_readiness_blockers,
)


def write_synthetic_trajectory(root: Path, accepted: bool = True) -> Path:
    root.mkdir()
    metadata = {"accepted_by_operator": accepted, "sample_error_count": 0}
    (root / "metadata.json").write_text(json.dumps(metadata))
    samples = []
    for index in range(61):
        samples.append(
            {
                "monotonic_time_ns": index * 20_000_000,
                "q": (np.ones(7) * index * 0.001).tolist(),
                "dq": (np.ones(7) * 0.05).tolist(),
                "translation_xyz_m": [0.4 + index * 0.001, -0.1, 0.2],
            }
        )
    (root / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in samples) + "\n")
    event_indices = [0, 10, 20, 40, 50, 60]
    events = [
        {"event": name, "monotonic_time_ns": samples[index]["monotonic_time_ns"]}
        for name, index in zip(EXPECTED_EVENTS, event_indices, strict=True)
    ]
    (root / "events.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    return root


class ReplayPlanTests(unittest.TestCase):
    def test_full_stage_spans_every_recorded_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory")
            plan = build_replay_stage(path, "full_arm_no_gripper", command_rate_hz=15.0)
            self.assertEqual(plan.source_start_index, 0)
            self.assertEqual(plan.source_stop_index, 60)
            np.testing.assert_allclose(plan.expected_start_q, np.zeros(7))
            np.testing.assert_allclose(plan.expected_stop_q, np.ones(7) * 0.06)

    def test_approach_stage_is_resampled_as_absolute_joint_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory")
            plan = build_replay_stage(path, "approach_open", command_rate_hz=15.0)
            np.testing.assert_allclose(plan.expected_start_q, np.zeros(7))
            np.testing.assert_allclose(plan.expected_stop_q, np.ones(7) * 0.01)
            self.assertEqual(plan.q_targets.shape[1], 7)
            self.assertEqual(plan.dq_targets.shape, plan.q_targets.shape)
            self.assertEqual(plan.xyz_targets.shape, (len(plan.q_targets), 3))
            self.assertEqual(plan.command_times_s.shape, (len(plan.q_targets),))
            self.assertGreaterEqual(len(plan.q_targets), 2)

    def test_rejected_trajectory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory", accepted=False)
            with self.assertRaises(ReplayPlanError):
                build_replay_stage(path, "approach_open", command_rate_hz=15.0)

    def test_uncalibrated_replay_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory")
            plan = build_replay_stage(path, "retreat_open", command_rate_hz=15.0)
            config = {
                "replay": {
                    "enabled": False,
                    "workspace_bounds_m": "NOT_CALIBRATED",
                    "joint_velocity_cap_rad_s": "NOT_CALIBRATED",
                    "start_joint_tolerance_rad": "NOT_CALIBRATED",
                }
            }
            blockers = replay_readiness_blockers(config, plan)
            self.assertIn("replay.enabled is false", blockers)
            self.assertEqual(len(blockers), 4)


if __name__ == "__main__":
    unittest.main()
