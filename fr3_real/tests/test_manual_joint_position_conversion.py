"""Offline tests for manual Pilot demonstration action semantics."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = FR3_ROOT / "data-pipeline" / "convert_raw_recordings_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("fr3_raw_converter", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class ManualJointPositionConversionTests(unittest.TestCase):
    def test_action_is_absolute_next_q_and_measured_next_closedness(self) -> None:
        next_sample = {
            "q": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
            "gripper_width": 0.02,
            "gripper_command_width": None,
        }
        action = CONVERTER.droid_joint_position_action_vector(next_sample, 0.08)
        np.testing.assert_allclose(action[:7], next_sample["q"])
        self.assertAlmostEqual(float(action[7]), 0.75)
        self.assertEqual(action.dtype, np.float32)

    def test_measured_gripper_is_clamped_to_closedness_range(self) -> None:
        too_wide = {"q": [0.0] * 7, "gripper_width": 0.10}
        below_zero = {"q": [0.0] * 7, "gripper_width": -0.01}
        self.assertEqual(float(CONVERTER.droid_joint_position_action_vector(too_wide, 0.08)[7]), 0.0)
        self.assertEqual(float(CONVERTER.droid_joint_position_action_vector(below_zero, 0.08)[7]), 1.0)

    def test_full_policy_view_splits_at_bracket_retreat(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                {"time_ns": 200, "event": "bracket_retreat_open_completed"},
                {"time_ns": 500, "event": "thread_ready_completed"},
                {"time_ns": 850, "event": "training_motion_completed"},
                {"time_ns": 900, "event": "autonomous_bolt_episode_finished"},
            ]
            (root / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            metadata = {
                "task_name": "full_bracket_bolt_assembly",
                "record_start_time_ns": 100,
                "success": True,
                "bc_training_eligible": True,
                "setup_validation_only": False,
            }
            views = CONVERTER.derived_episode_views(
                root, metadata, "two_task_full_policy", "unused"
            )
        self.assertEqual([view.name for view in views], ["bracket", "bolt_full"])
        self.assertEqual((views[0].start_time_ns, views[0].end_time_ns), (100, 200))
        self.assertEqual((views[1].start_time_ns, views[1].end_time_ns), (200, 850))
        self.assertEqual(views[0].task, CONVERTER.BRACKET_TASK_PROMPT)
        self.assertEqual(views[1].task, CONVERTER.FULL_BOLT_TASK_PROMPT)

    def test_hybrid_view_ends_at_thread_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                {"time_ns": 200, "event": "bracket_retreat_open_completed"},
                {"time_ns": 500, "event": "thread_ready_completed"},
                {"time_ns": 850, "event": "training_motion_completed"},
                {"time_ns": 900, "event": "autonomous_bolt_episode_finished"},
            ]
            (root / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            metadata = {
                "task_name": "full_bracket_bolt_assembly",
                "record_start_time_ns": 100,
                "success": True,
                "bc_training_eligible": True,
                "setup_validation_only": False,
            }
            views = CONVERTER.derived_episode_views(
                root, metadata, "two_task_hybrid", "unused"
            )
        self.assertEqual([view.name for view in views], ["bracket", "bolt_thread_ready"])
        self.assertEqual((views[1].start_time_ns, views[1].end_time_ns), (200, 500))
        self.assertEqual(views[1].task, CONVERTER.HYBRID_BOLT_TASK_PROMPT)

    def test_legacy_training_capture_uses_release_plus_terminal_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [
                {"time_ns": 200, "event": "bracket_retreat_open_completed"},
                {"time_ns": 500, "event": "thread_ready_completed"},
                {"time_ns": 2_000_000_000, "event": "autonomous_bolt_episode_finished"},
            ]
            actions = [
                {"time_ns": 800_000_000, "action": "engaged_bolt_released"},
            ]
            (root / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            (root / "actions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in actions)
            )
            metadata = {
                "task_name": "full_bracket_bolt_assembly",
                "record_start_time_ns": 100,
                "success": True,
                "bc_training_eligible": True,
                "setup_validation_only": False,
            }
            views = CONVERTER.derived_episode_views(
                root, metadata, "two_task_full_policy", "unused"
            )
        self.assertEqual(views[1].end_time_ns, 1_800_000_000)

    def test_setup_validation_cannot_be_derived_as_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = {
                "task_name": "full_bracket_bolt_assembly",
                "record_start_time_ns": 100,
                "success": False,
                "bc_training_eligible": False,
                "setup_validation_only": True,
            }
            with self.assertRaisesRegex(RuntimeError, "setup-validation"):
                CONVERTER.derived_episode_views(
                    root, metadata, "two_task_full_policy", "unused"
                )


if __name__ == "__main__":
    unittest.main()
