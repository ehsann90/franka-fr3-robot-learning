"""Offline tests for calibration persistence and read-only state snapshots."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


FR3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.calibration import CalibrationError, CalibrationStore, normalize_pose_name  # noqa: E402
from expert_demo.snapshots import SnapshotReadError, capture_robot_snapshot  # noqa: E402
from expert_demo.transforms import make_transform, rotation_matrix_from_axis_angle  # noqa: E402


class FakeDuration:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def to_sec(self) -> float:
        return self.seconds


class FakeAffine:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix


class FakeRobotState:
    def __init__(self) -> None:
        rotation = rotation_matrix_from_axis_angle([0.0, 0.0, 1.0], 0.2)
        self.O_T_EE = FakeAffine(make_transform([0.4, -0.2, 0.3], rotation))
        self.q = np.arange(7, dtype=float) / 10.0
        self.dq = np.arange(7, dtype=float) / 100.0
        self.q_d = self.q + 0.01
        self.dq_d = self.dq + 0.001
        self.tau_J = np.arange(7, dtype=float)
        self.tau_J_d = np.arange(7, dtype=float) + 0.1
        self.tau_ext_hat_filtered = np.arange(7, dtype=float) + 0.2
        self.O_F_ext_hat_K = np.arange(6, dtype=float)
        self.K_F_ext_hat_K = np.arange(6, dtype=float) + 1.0
        self.time = FakeDuration(12.5)


class FakeRobot:
    def __init__(self) -> None:
        self.state_reads = 0
        self._state = FakeRobotState()

    @property
    def state(self) -> FakeRobotState:
        self.state_reads += 1
        return self._state


class FakeGripperState:
    width = 0.031
    max_width = 0.08
    is_grasped = True
    temperature = 27
    time = FakeDuration(4.25)


class FakeGripper:
    @property
    def state(self) -> FakeGripperState:
        return FakeGripperState()


class FlakyGripper:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.reads = 0

    @property
    def state(self) -> FakeGripperState:
        self.reads += 1
        if self.reads <= self.failures:
            raise RuntimeError("simulated Hand timeout")
        return FakeGripperState()


def make_snapshot(notes: str = "test") -> dict:
    return capture_robot_snapshot(
        FakeRobot(),
        FakeGripper(),
        snapshot_kind="ee_pose",
        notes=notes,
    )


class SnapshotTests(unittest.TestCase):
    def test_capture_reads_robot_state_once_and_contains_required_data(self) -> None:
        robot = FakeRobot()
        snapshot = capture_robot_snapshot(robot, FakeGripper(), snapshot_kind="ee_pose", notes="center")
        self.assertEqual(robot.state_reads, 1)
        self.assertEqual(len(snapshot["q"]), 7)
        self.assertEqual(len(snapshot["dq"]), 7)
        self.assertEqual(np.asarray(snapshot["T_base_ee"]).shape, (4, 4))
        self.assertEqual(snapshot["translation_xyz_m"], [0.4, -0.2, 0.3])
        self.assertEqual(snapshot["orientation"]["quaternion_order"], "xyzw")
        self.assertAlmostEqual(np.linalg.norm(snapshot["orientation"]["quaternion_xyzw"]), 1.0)
        self.assertTrue(snapshot["gripper"]["is_grasped"])
        self.assertEqual(snapshot["notes"], "center")

    def test_capture_can_explicitly_omit_gripper(self) -> None:
        snapshot = capture_robot_snapshot(FakeRobot(), None, snapshot_kind="joint_pose")
        self.assertEqual(snapshot["gripper"], {"available": False})

    def test_transient_gripper_timeout_is_retried(self) -> None:
        gripper = FlakyGripper(failures=2)
        snapshot = capture_robot_snapshot(FakeRobot(), gripper, snapshot_kind="ee_pose")
        self.assertEqual(gripper.reads, 3)
        self.assertTrue(snapshot["gripper"]["available"])

    def test_persistent_gripper_timeout_refuses_incomplete_snapshot(self) -> None:
        gripper = FlakyGripper(failures=3)
        with self.assertRaisesRegex(SnapshotReadError, "no calibration pose was saved"):
            capture_robot_snapshot(FakeRobot(), gripper, snapshot_kind="ee_pose")


class CalibrationStoreTests(unittest.TestCase):
    def test_add_reload_and_explicit_overwrite_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            store = CalibrationStore(path, "172.16.0.2")
            store.add("bracket center grasp", make_snapshot("first"))
            self.assertTrue(path.exists())
            self.assertEqual(store.list_pose_names(), ["BRACKET_CENTER_GRASP"])

            reloaded = CalibrationStore(path, "172.16.0.2")
            self.assertEqual(reloaded.calibration_hash, store.calibration_hash)
            self.assertEqual(reloaded.poses["BRACKET_CENTER_GRASP"]["notes"], "first")

            with self.assertRaises(CalibrationError):
                reloaded.add("BRACKET_CENTER_GRASP", make_snapshot("silent overwrite"))

            old_hash = reloaded.calibration_hash
            backup = reloaded.overwrite("BRACKET_CENTER_GRASP", make_snapshot("replacement"))
            self.assertTrue(backup.exists())
            self.assertNotEqual(reloaded.calibration_hash, old_hash)
            self.assertEqual(reloaded.poses["BRACKET_CENTER_GRASP"]["notes"], "replacement")
            backup_document = json.loads(backup.read_text())
            self.assertEqual(backup_document["poses"]["BRACKET_CENTER_GRASP"]["notes"], "first")

    def test_hash_detects_manual_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            store = CalibrationStore(path, "172.16.0.2")
            store.add("WALL_PLACE_LOW", make_snapshot())
            document = json.loads(path.read_text())
            document["poses"]["WALL_PLACE_LOW"]["q"][0] += 1.0
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(CalibrationError, "hash mismatch"):
                CalibrationStore(path, "172.16.0.2")

    def test_robot_ip_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            store = CalibrationStore(path, "172.16.0.2")
            store.add("WALL_PLACE_HIGH", make_snapshot())
            with self.assertRaisesRegex(CalibrationError, "robot_ip"):
                CalibrationStore(path, "192.0.2.10")

    def test_pose_name_normalization_and_rejection(self) -> None:
        self.assertEqual(normalize_pose_name("wall place low"), "WALL_PLACE_LOW")
        with self.assertRaises(CalibrationError):
            normalize_pose_name("../wall")


if __name__ == "__main__":
    unittest.main()
