import unittest

import numpy as np

from expert_demo.hybrid_thread_plan import build_hybrid_thread_plan
from expert_demo.piloted_thread_replay import PilotedThreadDrive


class HybridThreadPlanTests(unittest.TestCase):
    def test_removes_axial_translation_but_preserves_lateral_and_rotation(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, 3] = [0.1, 0.2, 0.3]
        poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        drive = PilotedThreadDrive(
            cycle=1,
            q_targets=np.zeros((2, 7)),
            dq_targets=np.zeros((2, 7)),
            xyz_targets=poses[:, :3, 3],
            pose_targets=poses,
            source_duration_s=1.0,
            command_rate_hz=15.0,
            source_start_time_ns=0,
            source_stop_time_ns=1_000_000_000,
        )
        plan = build_hybrid_thread_plan(drive, playback_time_scale=2.0)
        self.assertTrue(np.allclose(plan.local_lateral_m[-1], [0.1, 0.2, 0.0]))
        self.assertAlmostEqual(plan.time_s[-1], 2.0)
        self.assertTrue(np.allclose(np.linalg.norm(plan.relative_quaternion_xyzw, axis=1), 1.0))

    def test_stationary_prefix_is_trimmed_with_smooth_lead_in(self):
        poses = np.tile(np.eye(4), (8, 1, 1))
        poses[6, 0, 3] = 0.0002
        poses[7, 0, 3] = 0.0004
        q = np.zeros((8, 7))
        q[6:, 6] = [0.002, 0.004]
        drive = PilotedThreadDrive(
            cycle=1,
            q_targets=q,
            dq_targets=np.zeros_like(q),
            xyz_targets=poses[:, :3, 3],
            pose_targets=poses,
            source_duration_s=7.0,
            command_rate_hz=1.0,
            source_start_time_ns=0,
            source_stop_time_ns=7_000_000_000,
        )
        plan = build_hybrid_thread_plan(
            drive,
            playback_time_scale=1.0,
            trim_stationary_prefix=True,
            stationary_lead_in_s=0.25,
        )
        self.assertAlmostEqual(plan.source_trim_start_s, 6.0)
        self.assertTrue(np.allclose(plan.time_s, [0.0, 0.25, 1.25]))
        self.assertTrue(np.allclose(plan.local_lateral_m[0], 0.0))
        self.assertAlmostEqual(plan.local_lateral_m[1, 0], 0.0002)

    def test_threading_mode_holds_translation_and_removes_roll_pitch(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        poses[1, :3, 3] = [0.01, -0.02, 0.03]
        roll = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64
        )
        yaw = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
        poses[1, :3, :3] = yaw @ roll
        q = np.zeros((2, 7))
        drive = PilotedThreadDrive(
            cycle=1,
            q_targets=q,
            dq_targets=q,
            xyz_targets=poses[:, :3, 3],
            pose_targets=poses,
            source_duration_s=1.0,
            command_rate_hz=15.0,
            source_start_time_ns=0,
            source_stop_time_ns=1_000_000_000,
        )
        plan = build_hybrid_thread_plan(
            drive,
            playback_time_scale=1.0,
            replay_lateral_translation=False,
            replay_yaw_only=True,
        )
        self.assertTrue(np.allclose(plan.local_lateral_m, 0.0))
        quaternion = plan.relative_quaternion_xyzw[-1]
        self.assertTrue(
            np.allclose(np.abs(quaternion), [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
        )


if __name__ == "__main__":
    unittest.main()
