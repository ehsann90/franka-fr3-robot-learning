"""Offline checks for the unrecorded B44 wall-to-cell setup move."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "model-running" / "prepare_bracket_wall_cell_cycle.py"
CONFIGS = (
    ROOT / "configs" / "bracket_manual_replay_1x_highcap_b44.json",
    ROOT / "configs" / "bracket_manual_replay_1x_highcap_b93.json",
)


class SetupPlanningTests(unittest.TestCase):
    def test_slower_setups_are_not_dataset_episodes(self):
        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location("bracket_setup_planner", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for config in CONFIGS:
            with self.subTest(config=config.name):
                manifest = module.prepare(config, None)
                self.assertEqual(manifest["blockers"], [])
                self.assertFalse(manifest["inverse_dataset_episode"])
                self.assertNotIn("stop_inverse_episode", manifest["sequence"])
                durations = manifest["inverse_stages"]
                self.assertGreater(durations["wall_to_cell_held"]["duration_s"], 20.0)
                self.assertGreater(durations["cell_retreat_open"]["duration_s"], 9.0)
                self.assertLess(durations["wall_to_cell_held"]["peaks"]["velocity_rad_s"], 0.6)


if __name__ == "__main__":
    unittest.main()
