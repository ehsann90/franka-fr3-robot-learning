"""Offline checks for the bracket inverse collector's automatic start reset."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "model-running" / "run_bracket_wall_to_cell.py"


class FakeState:
    def __init__(self, q):
        self.q = q
        self.F_T_EE = np.eye(4)
        self.EE_T_K = np.eye(4)
        self.O_T_EE = types.SimpleNamespace(matrix=self.matrix(q))

    @staticmethod
    def matrix(q):
        value = np.eye(4)
        value[:3, 3] = [0.5 + 0.01 * q[0], -0.1, 0.2]
        return value


class FakeRobot:
    def __init__(self):
        self.model = self

    def pose(self, _frame, q_or_state, *_args):
        q = q_or_state.q if isinstance(q_or_state, FakeState) else q_or_state
        return types.SimpleNamespace(matrix=FakeState.matrix(q))


class ResetPlanningTests(unittest.TestCase):
    def test_far_start_becomes_two_guarded_segments(self):
        spec = importlib.util.spec_from_file_location("bracket_inverse", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        current = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
        target = current.copy()
        target[0] = 0.823
        state = FakeState(current)
        config = {
            "reset_maximum_joint_delta_rad": 0.5,
            "reset_maximum_segments": 3,
            "reset_workspace_bounds_m": {
                "min_xyz": [0.21, -0.53, -0.04],
                "max_xyz": [0.73, 0.4, 0.32],
            },
        }
        with patch.dict(sys.modules, {"franky": types.SimpleNamespace(Frame=types.SimpleNamespace(Joint7=object()))}):
            path, segments, blockers = module.plan_empty_hand_reset(
                FakeRobot(), state, current, target, np.array([0.50823, -0.1, 0.2]), config
            )
        self.assertEqual(segments, 2)
        self.assertEqual(blockers, [])
        self.assertEqual(path.shape, (201, 3))

    def test_reset_refuses_workspace_escape(self):
        spec = importlib.util.spec_from_file_location("bracket_inverse", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        current = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
        target = current.copy()
        target[0] = 0.823
        config = {
            "reset_maximum_joint_delta_rad": 0.5,
            "reset_maximum_segments": 3,
            "reset_workspace_bounds_m": {
                "min_xyz": [0.51, -0.53, -0.04],
                "max_xyz": [0.73, 0.4, 0.32],
            },
        }
        with patch.dict(sys.modules, {"franky": types.SimpleNamespace(Frame=types.SimpleNamespace(Joint7=object()))}):
            _, _, blockers = module.plan_empty_hand_reset(
                FakeRobot(), FakeState(current), current, target,
                np.array([0.50823, -0.1, 0.2]), config,
            )
        self.assertIn("reset path leaves the configured workspace", blockers)


if __name__ == "__main__":
    unittest.main()
