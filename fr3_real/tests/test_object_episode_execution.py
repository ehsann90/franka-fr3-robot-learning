"""Offline commissioning tests for live bracket episode execution."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

import numpy as np

from expert_demo.object_episode_execution import (  # noqa: E402
    REQUIRED_STAGES,
    automatic_reset_blockers,
    object_episode_blockers,
)
from expert_demo.replay_plan import build_replay_stage  # noqa: E402


class ObjectEpisodeExecutionTests(unittest.TestCase):
    def test_automatic_reset_accepts_valid_computed_path(self) -> None:
        reset = {
            "enabled": True,
            "relative_dynamics_factor": 0.05,
            "joint_margin_rad": 0.12,
            "maximum_joint_delta_rad": 0.5,
            "workspace_bounds_m": {"min_xyz": [0.5, -0.2, 0.1], "max_xyz": [0.7, 0.1, 0.3]},
            "target_fk_error_limit_m": 0.005,
            "terminal_joint_tolerance_rad": 0.005,
            "requires_empty_gripper": True,
            "requires_bracket_removed": True,
        }
        blockers = automatic_reset_blockers(
            reset,
            np.zeros(7),
            np.ones(7) * 0.1,
            np.asarray([[0.6, -0.1, 0.2], [0.62, -0.05, 0.21]]),
            0.0001,
        )
        self.assertEqual(blockers, [])

    def test_automatic_reset_blocks_path_outside_workspace(self) -> None:
        reset = {
            "enabled": True,
            "relative_dynamics_factor": 0.05,
            "joint_margin_rad": 0.12,
            "maximum_joint_delta_rad": 0.5,
            "workspace_bounds_m": {"min_xyz": [0.5, -0.2, 0.1], "max_xyz": [0.7, 0.1, 0.3]},
            "target_fk_error_limit_m": 0.005,
            "terminal_joint_tolerance_rad": 0.005,
            "requires_empty_gripper": True,
            "requires_bracket_removed": True,
        }
        blockers = automatic_reset_blockers(
            reset,
            np.zeros(7),
            np.ones(7) * 0.1,
            np.asarray([[0.8, -0.1, 0.2]]),
            0.0001,
        )
        self.assertIn("computed automatic reset path leaves its configured workspace", blockers)

    def test_real_object_episode_has_no_static_blockers(self) -> None:
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bracket_trajectories_v1"
            / "20260812_115247_bracket_pilot_0004_B1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted hardware trajectory is not present")
        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json").read_text()
        )
        plans = {
            name: build_replay_stage(
                trajectory,
                name,
                command_rate_hz=float(config["replay"]["command_rate_hz"]),
            )
            for name in REQUIRED_STAGES
        }
        blockers = object_episode_blockers(config["replay"]["object_episode_execution"], plans)
        self.assertEqual(blockers, [])


if __name__ == "__main__":
    unittest.main()
