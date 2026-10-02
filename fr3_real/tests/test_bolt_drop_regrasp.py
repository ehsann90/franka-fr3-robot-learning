import json
import importlib.util
import unittest
from pathlib import Path

from expert_demo.bolt_drop_regrasp import (
    BOLT_DROP_REGRASP_STAGE_EVENTS,
    EXPECTED_BOLT_DROP_REGRASP_EVENTS,
    bolt_drop_regrasp_metadata_blockers,
    build_bolt_drop_regrasp_stage,
)


FR3_ROOT = Path(__file__).resolve().parents[1]


class BoltDropRegraspTests(unittest.TestCase):
    def test_config_encodes_release_then_regrasp_and_disables_threading(self):
        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bolt_drop_regrasp_v1.json").read_text()
        )
        self.assertEqual(
            config["mechanical_normalization"]["method"],
            "release_into_guide_then_regrasp_head",
        )
        self.assertTrue(config["mechanical_normalization"]["guide_holds_bolt_upright"])
        self.assertFalse(config["replay"]["enabled"])
        self.assertFalse(config["contact_threading"]["enabled"])
        episode = config["replay"]["object_normalization_execution"]
        self.assertTrue(episode["enabled"])
        self.assertFalse(episode["contact_threading_enabled"])
        self.assertTrue(episode["gripper"]["operator_visual_confirmation_required"])

    def test_events_separate_closed_and_open_arm_paths(self):
        self.assertEqual(EXPECTED_BOLT_DROP_REGRASP_EVENTS[0], "trajectory_started_preinsert_held")
        self.assertEqual(BOLT_DROP_REGRASP_STAGE_EVENTS["lower_held"][1], "drop_pose_reached_held")
        self.assertEqual(
            BOLT_DROP_REGRASP_STAGE_EVENTS["reposition_regrasp_open"][0],
            "bolt_released_upright",
        )
        self.assertEqual(EXPECTED_BOLT_DROP_REGRASP_EVENTS[-1], "thread_ready_regrasped")

    def test_valid_metadata(self):
        metadata = {
            "task_name": "bolt",
            "capture_scope": "preinsert_through_thread_ready_only",
            "mechanical_normalization_method": "release_into_guide_then_regrasp_head",
            "bolt_upright_at_endpoint": True,
            "bolt_regrasped_at_endpoint": True,
            "rotation_or_threading_included": False,
            "cameras_recorded": False,
        }
        self.assertEqual(bolt_drop_regrasp_metadata_blockers(metadata), [])

    def test_real_capture_builds_two_arm_only_stages(self):
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bolt_drop_regrasp_trajectories_v1"
            / "20260812_133722_bolt_drop_regrasp_pilot_0001_S1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted drop/regrasp capture is not present")
        lower = build_bolt_drop_regrasp_stage(
            str(trajectory), "lower_held", command_rate_hz=15.0
        )
        reposition = build_bolt_drop_regrasp_stage(
            str(trajectory), "reposition_regrasp_open", command_rate_hz=15.0
        )
        self.assertEqual(lower.stop_event, "drop_pose_reached_held")
        self.assertEqual(reposition.start_event, "bolt_released_upright")
        self.assertEqual(lower.q_targets.shape[1], 7)
        self.assertEqual(reposition.q_targets.shape[1], 7)

    def test_stationary_trim_removes_prompt_dwell_but_preserves_endpoints(self):
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bolt_drop_regrasp_trajectories_v1"
            / "20260812_133722_bolt_drop_regrasp_pilot_0001_S1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted drop/regrasp capture is not present")
        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bolt_drop_regrasp_v1.json").read_text()
        )
        trims = config["replay"]["stage_stationary_trim_s"]
        full_lower = build_bolt_drop_regrasp_stage(
            str(trajectory), "lower_held", command_rate_hz=15.0
        )
        lower = build_bolt_drop_regrasp_stage(
            str(trajectory),
            "lower_held",
            command_rate_hz=15.0,
            trim_start_s=trims["lower_held"]["start_s"],
            trim_end_s=trims["lower_held"]["end_s"],
        )
        self.assertLess(lower.source_duration_s, 6.0)
        self.assertLess(
            max(abs(lower.expected_start_q - full_lower.expected_start_q)), 0.0005
        )
        self.assertLess(max(abs(lower.expected_stop_q - full_lower.expected_stop_q)), 0.0005)

    def test_real_object_profile_has_no_static_blockers(self):
        trajectory = (
            FR3_ROOT
            / "recordings"
            / "piloted_bolt_drop_regrasp_trajectories_v1"
            / "20260812_133722_bolt_drop_regrasp_pilot_0001_S1_CENTER"
        )
        if not trajectory.exists():
            self.skipTest("accepted drop/regrasp capture is not present")
        script = FR3_ROOT / "model-running" / "replay_piloted_bolt_drop_regrasp_episode.py"
        spec = importlib.util.spec_from_file_location("drop_regrasp_episode", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = json.loads(
            (FR3_ROOT / "configs" / "piloted_bolt_drop_regrasp_v1.json").read_text()
        )
        plans = {
            name: build_bolt_drop_regrasp_stage(
                str(trajectory),
                name,
                command_rate_hz=15.0,
                trim_start_s=config["replay"]["stage_stationary_trim_s"][name]["start_s"],
                trim_end_s=config["replay"]["stage_stationary_trim_s"][name]["end_s"],
            )
            for name in BOLT_DROP_REGRASP_STAGE_EVENTS
        }
        width = json.loads((FR3_ROOT / "calibration" / "bolt_grasp_width_v2.json").read_text())
        blockers = module.static_blockers(
            config["replay"]["object_normalization_execution"], plans, width
        )
        self.assertEqual(blockers, [])


if __name__ == "__main__":
    unittest.main()
