"""Offline tests for guarded piloted-stage execution gates."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.replay_execution import (  # noqa: E402
    smoothed_position_velocities,
    stage_config_blockers,
)
from expert_demo.replay_plan import build_replay_stage  # noqa: E402
from test_replay_plan import write_synthetic_trajectory  # noqa: E402


def commissioned_stage(trajectory: str) -> dict:
    return {
        "enabled": True,
        "trajectory": trajectory,
        "empty_workspace_only": True,
        "gripper_commands_enabled": False,
        "playback_time_scale": 4.0,
        "relative_dynamics_factor": 0.3,
        "joint_margin_rad": 0.12,
        "start_joint_tolerance_rad": 0.005,
        "terminal_joint_tolerance_rad": 0.02,
        "maximum_scaled_target_velocity_rad_s": 0.5,
        "maximum_target_step_rad": 0.12,
        "workspace_bounds_m": {"min_xyz": [0.3, -0.2, 0.1], "max_xyz": [0.6, 0.0, 0.3]},
        "maximum_fk_path_error_m": 0.005,
        "state_timeout_s": 90.0,
    }


class ReplayExecutionTests(unittest.TestCase):
    def test_smoothed_position_velocities_have_zero_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory")
            plan = build_replay_stage(path, "approach_open", command_rate_hz=15.0)
            velocities = smoothed_position_velocities(plan, 2.0, 3)
            self.assertEqual(velocities.shape, plan.q_targets.shape)
            self.assertTrue((velocities[0] == 0).all())
            self.assertTrue((velocities[-1] == 0).all())

    def test_real_full_arm_stage_commissioning_has_no_static_blockers(self) -> None:
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bracket_trajectories_v1"
            / "20260812_115247_bracket_pilot_0004_B1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted hardware trajectory is not present")
        import json

        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json").read_text()
        )
        plan = build_replay_stage(
            trajectory,
            "full_arm_no_gripper",
            command_rate_hz=float(config["replay"]["command_rate_hz"]),
        )
        stage = config["replay"]["stage_execution"]["full_arm_no_gripper"]
        self.assertIn("stage execution is disabled", stage_config_blockers(stage, plan))

    def test_commissioned_synthetic_stage_has_no_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory")
            plan = build_replay_stage(path, "approach_open", command_rate_hz=15.0)
            self.assertEqual(stage_config_blockers(commissioned_stage(path.name), plan), [])

    def test_wrong_trajectory_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_synthetic_trajectory(Path(temporary) / "trajectory")
            plan = build_replay_stage(path, "approach_open", command_rate_hz=15.0)
            blockers = stage_config_blockers(commissioned_stage("different"), plan)
            self.assertTrue(any("commissioned for trajectory" in blocker for blocker in blockers))

    def test_real_transport_stage_commissioning_has_no_static_blockers(self) -> None:
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bracket_trajectories_v1"
            / "20260812_115247_bracket_pilot_0004_B1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted hardware trajectory is not present")
        import json

        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json").read_text()
        )
        plan = build_replay_stage(
            trajectory,
            "transport_place_closed",
            command_rate_hz=float(config["replay"]["command_rate_hz"]),
        )
        stage = config["replay"]["stage_execution"]["transport_place_closed"]
        self.assertEqual(stage_config_blockers(stage, plan), [])

    def test_real_retreat_stage_commissioning_has_no_static_blockers(self) -> None:
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bracket_trajectories_v1"
            / "20260812_115247_bracket_pilot_0004_B1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted hardware trajectory is not present")
        import json

        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bracket_trajectory_v1.json").read_text()
        )
        plan = build_replay_stage(
            trajectory,
            "retreat_open",
            command_rate_hz=float(config["replay"]["command_rate_hz"]),
        )
        stage = config["replay"]["stage_execution"]["retreat_open"]
        self.assertEqual(stage_config_blockers(stage, plan), [])


if __name__ == "__main__":
    unittest.main()
