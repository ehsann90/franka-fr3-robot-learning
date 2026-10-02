import sys
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT / "franka-tests"))

from realsense_recorder import RealSenseEpisodeRecorder


class ManualGripperMarkerTests(unittest.TestCase):
    def recorder(self):
        recorder = RealSenseEpisodeRecorder.__new__(RealSenseEpisodeRecorder)
        recorder.active = True
        recorder.action_records = []
        recorder.gripper_command_width = None
        recorder.latest_gripper_state = {}
        recorder.manual_gripper_state_enabled = False
        recorder.state_samples = []
        return recorder

    def test_marker_updates_piecewise_constant_state_and_labels_source(self):
        recorder = self.recorder()
        recorder.set_manual_gripper_state(0.015746, True, "head_gripper_closed")
        self.assertTrue(recorder.manual_gripper_state_enabled)
        self.assertEqual(recorder.gripper_command_width, 0.015746)
        self.assertEqual(recorder.latest_gripper_state["gripper_width"], 0.015746)
        self.assertTrue(recorder.latest_gripper_state["gripper_is_grasped"])
        self.assertEqual(recorder.action_records[0]["action"], "head_gripper_closed")
        self.assertEqual(
            recorder.action_records[0]["payload"]["source"],
            "operator_marked_discrete_state",
        )

    def test_automatic_marker_keeps_command_and_achieved_width_distinct(self):
        recorder = self.recorder()
        recorder.set_achieved_gripper_state(
            0.0162,
            True,
            "head_grasped",
            command_width=0.0,
        )
        self.assertEqual(recorder.gripper_command_width, 0.0)
        self.assertEqual(recorder.latest_gripper_state["gripper_width"], 0.0162)
        self.assertEqual(recorder.action_records[0]["payload"]["target_width"], 0.0)
        self.assertEqual(recorder.action_records[0]["payload"]["achieved_width"], 0.0162)

    def test_external_direct_states_are_merged_in_timestamp_order(self):
        recorder = self.recorder()
        recorder.state_samples = [{"time_ns": 20, "q": [0.0] * 7}]
        recorder.append_external_state_samples(
            [
                {"time_ns": 30, "q": [0.2] * 7},
                {"time_ns": 10, "q": [0.1] * 7},
            ]
        )
        self.assertEqual([row["time_ns"] for row in recorder.state_samples], [10, 20, 30])

    def test_external_direct_state_requires_seven_joints(self):
        recorder = self.recorder()
        with self.assertRaises(ValueError):
            recorder.append_external_state_samples([{"time_ns": 10, "q": [0.0] * 6}])


if __name__ == "__main__":
    unittest.main()
