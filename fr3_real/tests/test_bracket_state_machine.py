"""Synthetic tests for discrete bracket expert planning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bracket_expert import (  # noqa: E402
    BracketState,
    build_bracket_plan,
    load_bracket_config,
    required_pose_names,
)
from expert_demo.episode_spec import generate_bracket_episode_spec  # noqa: E402
from expert_demo.transforms import make_transform  # noqa: E402


CONFIG_PATH = FR3_ROOT / "configs" / "bracket_expert_v1.json"


def synthetic_pose(xyz: list[float]) -> dict:
    return {"T_base_ee": make_transform(xyz).tolist()}


class BracketStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_bracket_config(CONFIG_PATH)
        self.config["execution_enabled"] = True
        self.config["motion"] = {
            "dynamics_factor": 0.01,
            "max_cartesian_travel_per_state_m": 0.2,
            "state_timeout_s": 10.0,
            "grasp_width_m": 0.03,
            "grasp_speed_m_s": 0.01,
            "grasp_force_n": 1.0,
            "open_speed_m_s": 0.01,
        }
        self.config["safety"]["workspace_bounds_m"] = {
            "min_xyz": [0.0, -1.0, 0.0],
            "max_xyz": [1.0, 1.0, 1.0],
        }

    def test_complete_low_plan_has_explicit_state_sequence(self) -> None:
        spec = generate_bracket_episode_spec(
            self.config,
            episode_index=3,
            random_seed=12,
            bracket_slot="B1",
            bracket_grasp_style="LOW",
        )
        poses = {
            name: synthetic_pose([0.3 + index * 0.01, -0.1, 0.2])
            for index, name in enumerate(required_pose_names("LOW"))
        }
        plan = build_bracket_plan(spec, self.config, poses)
        states = [step.state for step in plan.steps]
        for expected in (
            BracketState.IDLE,
            BracketState.PREGRASP,
            BracketState.GRASP,
            BracketState.LIFT,
            BracketState.TRANSFER,
            BracketState.PREPLACE,
            BracketState.ALIGN,
            BracketState.PLACE,
            BracketState.RELEASE,
            BracketState.RETREAT,
            BracketState.SUCCESS,
        ):
            self.assertIn(expected, states)
        self.assertTrue(plan.executable)
        self.assertFalse(plan.missing_calibrations)

    def test_missing_calibrations_are_reported_without_fabricating_targets(self) -> None:
        spec = generate_bracket_episode_spec(
            self.config,
            episode_index=1,
            random_seed=0,
            bracket_slot="B1",
            bracket_grasp_style="HIGH",
        )
        plan = build_bracket_plan(spec, self.config, {})
        self.assertFalse(plan.executable)
        self.assertEqual(set(plan.missing_calibrations), set(required_pose_names("HIGH")))
        self.assertTrue(all(step.target_T_base_ee is None for step in plan.steps if step.calibration_name))

    def test_slot_offset_applies_only_to_pickup_poses(self) -> None:
        self.config["slots"]["B2"] = {"enabled": True, "offset_xyz_m": [0.05, -0.02, 0.0]}
        spec = generate_bracket_episode_spec(
            self.config,
            episode_index=1,
            random_seed=0,
            bracket_slot="B2",
            bracket_grasp_style="CENTER",
        )
        poses = {name: synthetic_pose([0.4, 0.1, 0.3]) for name in required_pose_names("CENTER")}
        plan = build_bracket_plan(spec, self.config, poses)
        targets = {step.calibration_name: step.target_T_base_ee for step in plan.steps if step.calibration_name}
        np.testing.assert_allclose(targets["BRACKET_CENTER_GRASP"][:3, 3], [0.45, 0.08, 0.3])
        np.testing.assert_allclose(targets["BRACKET_CENTER_TRANSFER_SAFE"][:3, 3], [0.4, 0.1, 0.3])
        np.testing.assert_allclose(targets["WALL_PLACE_CENTER"][:3, 3], [0.4, 0.1, 0.3])


if __name__ == "__main__":
    unittest.main()
