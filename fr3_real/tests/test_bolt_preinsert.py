"""Tests for the bolt pickup-through-PREINSERT capture contract."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.bolt_preinsert import (  # noqa: E402
    BOLT_PREINSERT_STAGE_EVENTS,
    EXPECTED_BOLT_PREINSERT_EVENTS,
    bolt_capture_metadata_blockers,
    build_bolt_preinsert_stage,
)


class BoltPreinsertTests(unittest.TestCase):
    def test_config_keeps_replay_and_contact_disabled(self) -> None:
        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bolt_preinsert_v1.json").read_text()
        )
        self.assertEqual(config["task_name"], "bolt")
        self.assertFalse(config["replay"]["enabled"])
        self.assertFalse(config["contact_threading"]["enabled"])
        self.assertEqual(config["preinsert_requirements"]["safe_clearance_m"], "NOT_CALIBRATED")
        object_episode = config["replay"]["object_preinsert_execution"]
        self.assertFalse(object_episode["contact_threading_enabled"])
        self.assertTrue(object_episode["gripper"]["operator_visual_confirmation_required"])
        self.assertTrue(object_episode["gripper"]["width_cannot_distinguish_empty_hand"])

    def test_expected_events_stop_at_preinsert(self) -> None:
        self.assertEqual(EXPECTED_BOLT_PREINSERT_EVENTS[-1], "preinsert_pose_reached")
        self.assertNotIn("contact", " ".join(EXPECTED_BOLT_PREINSERT_EVENTS))
        self.assertEqual(
            BOLT_PREINSERT_STAGE_EVENTS["transport_preinsert_closed"][1],
            "preinsert_pose_reached",
        )

    def test_real_capture_builds_two_replay_stages(self) -> None:
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bolt_preinsert_trajectories_v1"
            / "20260812_130029_bolt_preinsert_pilot_0001_S1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted bolt trajectory is not present")
        approach = build_bolt_preinsert_stage(
            str(trajectory), "approach_open", command_rate_hz=15.0
        )
        transport = build_bolt_preinsert_stage(
            str(trajectory), "transport_preinsert_closed", command_rate_hz=15.0
        )
        self.assertEqual(approach.q_targets.shape[1], 7)
        self.assertEqual(transport.q_targets.shape[1], 7)
        self.assertEqual(transport.stop_event, "preinsert_pose_reached")

    def test_valid_metadata(self) -> None:
        metadata = {
            "task_name": "bolt",
            "capture_scope": "pickup_through_preinsert_only",
            "contact_or_threading_included": False,
            "cameras_recorded": False,
        }
        self.assertEqual(bolt_capture_metadata_blockers(metadata), [])


if __name__ == "__main__":
    unittest.main()
