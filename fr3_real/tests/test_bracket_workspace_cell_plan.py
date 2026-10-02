"""The workspace planner never silently trusts malformed probe labels."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "model-running" / "plan_bracket_workspace_cells.py"


class WorkspaceCellPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location("bracket_workspace_plan", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.planner = module

    def test_current_probe_reorders_swapped_far_x_y_labels(self):
        raw = json.loads((ROOT / "calibration" / "bracket_workspace_corners_v1" / "probed_points.json").read_text())
        corners, warnings = self.planner.normalize_corners(raw)
        self.assertLess(corners["bracket_xmax_ymin"][1], corners["bracket_xmax_ymax"][1])
        self.assertAlmostEqual(corners["bracket_xmax_ymin"][1], -0.4645)
        self.assertAlmostEqual(corners["bracket_xmax_ymax"][1], 0.4255)
        self.assertTrue(any("reversed" in warning for warning in warnings))

    def test_bilinear_midpoint_is_average_of_corners(self):
        raw = {
            "bracket_xmin_ymin": [0.0, 0.0, 0.0],
            "bracket_xmin_ymax": [0.0, 1.0, 0.0],
            "bracket_xmax_ymin": [1.0, 0.0, 0.0],
            "bracket_xmax_ymax": [1.0, 1.0, 0.0],
        }
        corners, warnings = self.planner.normalize_corners(raw)
        self.assertEqual(warnings, [])
        np.testing.assert_allclose(self.planner.bilinear(corners, 0.5, 0.5), [0.5, 0.5, 0.0])

    def test_current_preview_is_read_only_and_has_reference_comparison(self):
        raw = json.loads((ROOT / "calibration" / "bracket_workspace_corners_v1" / "probed_points.json").read_text())
        preview = self.planner.make_preview(raw, x_count=3, y_count=5, inset=0.15)
        self.assertFalse(preview["execution_enabled"])
        self.assertEqual(len(preview["candidates"]), 15)
        self.assertGreater(preview["accepted_grasp_count"], 50)
        self.assertTrue(all(not candidate["motion_approved"] for candidate in preview["candidates"]))

    def test_first_new_cell_calibration_is_not_a_motion_or_dataset_episode(self):
        spec = json.loads((ROOT / "calibration" / "bracket_new_cell_p01_02_v1.json").read_text())
        preview = json.loads((ROOT / spec["source_preview"]).read_text())
        candidate = next(row for row in preview["candidates"] if row["proposal"] == spec["proposal"])
        self.assertEqual(spec["target_probed_ee_origin_xyz_m"], candidate["ee_xyz_m"])
        self.assertEqual(spec["nearest_accepted_cell"], candidate["nearest_accepted_cell"])
        self.assertFalse(spec["dataset_episode"])
        self.assertFalse(spec["motion_approved"])
        self.assertFalse(spec["replay_enabled"])


if __name__ == "__main__":
    unittest.main()
