"""Calibration-only bracket cycles use Enter at the checked Hand action."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-running"))

from manual_bracket_confirmation import confirm_hand_action  # noqa: E402


class ManualBracketHandConfirmationTests(unittest.TestCase):
    def test_calibration_cycle_accepts_empty_enter(self):
        self.assertTrue(confirm_hand_action("grasp", {"calibration_only": True}, lambda _: ""))
        self.assertTrue(confirm_hand_action("release", {"calibration_only": True}, lambda _: ""))

    def test_calibration_cycle_rejects_other_input(self):
        self.assertFalse(confirm_hand_action("grasp", {"calibration_only": True}, lambda _: "CLOSE_ON_BRACKET"))
        self.assertFalse(confirm_hand_action("release", {"calibration_only": True}, lambda _: "no"))

    def test_existing_recording_config_keeps_phrase(self):
        self.assertFalse(confirm_hand_action("grasp", {}, lambda _: ""))
        self.assertTrue(confirm_hand_action("grasp", {}, lambda _: "CLOSE_ON_BRACKET"))
        self.assertTrue(confirm_hand_action("release", {}, lambda _: "OPEN_AND_RELEASE"))


if __name__ == "__main__":
    unittest.main()
