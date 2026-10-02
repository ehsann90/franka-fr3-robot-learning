import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_demo.piloted_thread_replay import build_piloted_thread_drive


class PilotedThreadReplayTests(unittest.TestCase):
    def test_extracts_and_trims_drive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {"time_ns": 0, "event": "thread_drive_started", "payload": {"cycle": 1}},
                {"time_ns": 5_000_000_000, "event": "thread_drive_complete", "payload": {"cycle": 1}},
            ]
            (root / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events)
            )
            times = np.arange(6, dtype=np.int64) * 1_000_000_000
            q = np.zeros((6, 7)); q[:, 6] = np.arange(6)
            np.savez(
                root / "episode_arrays.npz",
                state_time_ns=times,
                state_q=q,
                state_dq=np.ones((6, 7)),
                state_ee_translation=np.zeros((6, 3)),
                state_O_T_EE=np.tile(np.eye(4).reshape(1, 16), (6, 1)),
            )
            drive = build_piloted_thread_drive(
                root, cycle=1, command_rate_hz=2.0, maximum_q7_rad=3.1
            )
            self.assertEqual(drive.expected_start_q[6], 0.0)
            self.assertAlmostEqual(drive.expected_stop_q[6], 3.1)
            self.assertEqual(drive.source_duration_s, 5.0)
            self.assertTrue(np.all(drive.q_targets[:, 6] <= 3.1))
            self.assertEqual(drive.pose_targets.shape[1:], (4, 4))
            self.assertTrue(np.allclose(drive.pose_targets[:, :3, :3], np.eye(3)))

    def test_rejects_missing_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events.jsonl").write_text("")
            with self.assertRaises(ValueError):
                build_piloted_thread_drive(
                    root, cycle=2, command_rate_hz=15.0, maximum_q7_rad=2.98
                )


if __name__ == "__main__":
    unittest.main()
