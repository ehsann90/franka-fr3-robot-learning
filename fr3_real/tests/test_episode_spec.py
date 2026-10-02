"""Tests for deterministic Stage-1 episode selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bracket_expert import load_bracket_config  # noqa: E402
from expert_demo.episode_spec import EpisodeSpecError, generate_bracket_episode_spec  # noqa: E402


CONFIG = FR3_ROOT / "configs" / "bracket_expert_v1.json"


class EpisodeSpecTests(unittest.TestCase):
    def test_seeded_selection_is_reproducible(self) -> None:
        config = load_bracket_config(CONFIG)
        first = generate_bracket_episode_spec(config, episode_index=17, random_seed=1234)
        second = generate_bracket_episode_spec(config, episode_index=17, random_seed=1234)
        self.assertEqual(first, second)
        self.assertEqual(first.bracket_slot, "B1")
        self.assertIn(first.bracket_grasp_style, {"HIGH", "CENTER", "LOW"})
        self.assertEqual(first.to_metadata()["random_seed"], 1234)

    def test_explicit_enabled_variation(self) -> None:
        config = load_bracket_config(CONFIG)
        spec = generate_bracket_episode_spec(
            config,
            episode_index=1,
            random_seed=9,
            bracket_slot="b1",
            bracket_grasp_style="low",
        )
        self.assertEqual(spec.bracket_slot, "B1")
        self.assertEqual(spec.bracket_grasp_style, "LOW")

    def test_disabled_slot_is_rejected(self) -> None:
        config = load_bracket_config(CONFIG)
        with self.assertRaises(EpisodeSpecError):
            generate_bracket_episode_spec(config, episode_index=1, random_seed=1, bracket_slot="B4")


if __name__ == "__main__":
    unittest.main()
