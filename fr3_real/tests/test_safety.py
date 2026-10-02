"""Unit tests for Cartesian workspace enforcement."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.safety import SafetyError, WorkspaceBounds  # noqa: E402
from expert_demo.transforms import make_transform  # noqa: E402


class SafetyTests(unittest.TestCase):
    def test_pose_inside_workspace_passes(self) -> None:
        bounds = WorkspaceBounds((0.2, -0.5, 0.05), (0.8, 0.5, 0.8))
        bounds.validate_pose(make_transform([0.5, 0.0, 0.3]), label="inside")

    def test_pose_outside_any_axis_is_rejected(self) -> None:
        bounds = WorkspaceBounds((0.2, -0.5, 0.05), (0.8, 0.5, 0.8))
        with self.assertRaisesRegex(SafetyError, "outside workspace"):
            bounds.validate_pose(make_transform([0.81, 0.0, 0.3]), label="outside")

    def test_inverted_bounds_are_rejected(self) -> None:
        with self.assertRaises(SafetyError):
            WorkspaceBounds((0.8, -0.5, 0.05), (0.2, 0.5, 0.8))


if __name__ == "__main__":
    unittest.main()
