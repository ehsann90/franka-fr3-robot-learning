"""Offline tests for camera-free piloted trajectory capture."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.trajectory_recording import (  # noqa: E402
    PilotedTrajectoryRecorder,
    gripper_state_record,
    robot_state_record,
)


class FakeState:
    def __init__(self) -> None:
        self.time = 1.25
        self.q = np.arange(7, dtype=float) * 0.1
        self.dq = np.arange(7, dtype=float) * 0.01
        self.q_d = self.q.copy()
        self.dq_d = self.dq.copy()
        self.ddq_d = np.zeros(7)
        self.tau_J = np.ones(7)
        self.tau_J_d = np.ones(7) * 2
        self.tau_ext_hat_filtered = np.ones(7) * 3
        self.O_F_ext_hat_K = np.arange(6, dtype=float)
        self.K_F_ext_hat_K = np.arange(6, dtype=float) + 10
        self.O_T_EE = np.eye(4)
        self.O_T_EE[:3, 3] = [0.4, -0.1, 0.2]
        self.F_T_EE = np.eye(4)
        self.F_T_EE[:3, 3] = [0.0, 0.0, 0.1]
        self.EE_T_K = np.eye(4)
        self.EE_T_K[:3, 3] = [0.0, 0.0, 0.02]


class FakeGripperState:
    width = 0.035
    max_width = 0.08
    is_grasped = True
    temperature = 31
    time = 2.0


class FakeRobot:
    state = FakeState()


class FakeGripper:
    state = FakeGripperState()


class TrajectoryRecordingTests(unittest.TestCase):
    def test_state_record_contains_authoritative_transform_and_signals(self) -> None:
        gripper = gripper_state_record(FakeGripperState())
        record = robot_state_record(FakeState(), gripper)
        self.assertEqual(len(record["q"]), 7)
        self.assertEqual(len(record["O_F_ext_hat_K"]), 6)
        self.assertEqual(len(record["K_F_ext_hat_K"]), 6)
        np.testing.assert_allclose(record["translation_xyz_m"], [0.4, -0.1, 0.2])
        np.testing.assert_allclose(np.asarray(record["T_base_ee"]), FakeState().O_T_EE)
        np.testing.assert_allclose(np.asarray(record["F_T_EE"]), FakeState().F_T_EE)
        np.testing.assert_allclose(np.asarray(record["EE_T_K"]), FakeState().EE_T_K)
        self.assertEqual(record["gripper"]["width_m"], 0.035)

    def test_write_preserves_raw_capture_and_keeps_replay_disabled(self) -> None:
        recorder = PilotedTrajectoryRecorder(FakeRobot(), FakeGripper(), 50.0, 10.0)
        grip = gripper_state_record(FakeGripperState())
        recorder.samples = [robot_state_record(FakeState(), grip), robot_state_record(FakeState(), grip)]
        recorder.add_event("trajectory_started")
        recorder.started_host_time_ns = 1_000_000_000
        recorder.stopped_host_time_ns = 2_000_000_000
        with tempfile.TemporaryDirectory() as temporary:
            output = recorder.write(Path(temporary) / "trajectory", {"trajectory_name": "TEST"}, True)
            metadata = __import__("json").loads((output / "metadata.json").read_text())
            self.assertTrue(metadata["accepted_by_operator"])
            self.assertFalse(metadata["replay_commissioned"])
            self.assertEqual(metadata["robot_sample_count"], 2)
            self.assertEqual(metadata["event_count"], 1)
            self.assertTrue((output / "trajectory_arrays.npz").exists())
            self.assertTrue((output / "events.jsonl").exists())

    def test_main_thread_gripper_poll_updates_latest_state(self) -> None:
        recorder = PilotedTrajectoryRecorder(FakeRobot(), FakeGripper(), 50.0, 10.0)
        self.assertTrue(recorder.poll_gripper_once())
        self.assertIsNotNone(recorder._latest_gripper)
        self.assertEqual(recorder._latest_gripper["width_m"], 0.035)


if __name__ == "__main__":
    unittest.main()
