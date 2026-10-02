"""Offline tests for piloted trajectory start-reset commissioning gates."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = FR3_ROOT / "model-running" / "reset_to_piloted_trajectory_start.py"
SPEC = importlib.util.spec_from_file_location("reset_to_piloted_start", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResetStartTests(unittest.TestCase):
    def test_uncalibrated_reset_is_blocked(self) -> None:
        reset = {
            "enabled": False,
            "relative_dynamics_factor": "NOT_CALIBRATED",
            "joint_margin_rad": "NOT_CALIBRATED",
            "maximum_joint_delta_rad": "NOT_CALIBRATED",
            "workspace_bounds_m": "NOT_CALIBRATED",
            "target_fk_error_limit_m": "NOT_CALIBRATED",
        }
        blockers = MODULE.reset_config_blockers(reset, np.zeros(7), np.zeros(7))
        self.assertEqual(len(blockers), 6)

    def test_maximum_joint_delta_gate(self) -> None:
        reset = {
            "enabled": True,
            "relative_dynamics_factor": 0.1,
            "joint_margin_rad": 0.1,
            "maximum_joint_delta_rad": 0.2,
            "workspace_bounds_m": {"min_xyz": [0, 0, 0], "max_xyz": [1, 1, 1]},
            "target_fk_error_limit_m": 0.01,
        }
        blockers = MODULE.reset_config_blockers(reset, np.ones(7) * 0.3, np.zeros(7))
        self.assertEqual(len(blockers), 1)
        self.assertIn("exceeds configured maximum", blockers[0])


if __name__ == "__main__":
    unittest.main()
