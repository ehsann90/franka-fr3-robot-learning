import unittest

from expert_demo.full_bolt_replay_plan import build_phase_boundaries


class FullBoltReplayPlanTests(unittest.TestCase):
    def test_builds_drive_and_ratchet_phases(self):
        events = [
            {"time_ns": 0, "event": "full_bolt_demo_started", "payload": {}},
            {"time_ns": 10, "event": "pickup_grasp_pose_reached", "payload": {}},
            {"time_ns": 30, "event": "bolt_self_supported", "payload": {}},
            {"time_ns": 50, "event": "head_regrasp_pose_reached", "payload": {}},
            {"time_ns": 70, "event": "thread_drive_started", "payload": {"cycle": 1}},
            {"time_ns": 80, "event": "thread_drive_complete", "payload": {"cycle": 1}},
            {"time_ns": 100, "event": "thread_drive_started", "payload": {"cycle": 2}},
            {"time_ns": 110, "event": "thread_drive_complete", "payload": {"cycle": 2}},
            {"time_ns": 140, "event": "full_bolt_demo_finished", "payload": {"success": True}},
        ]
        actions = [
            {"time_ns": 11, "action": "pickup_gripper_closed"},
            {"time_ns": 31, "action": "drop_gripper_opened"},
            {"time_ns": 51, "action": "head_gripper_closed"},
            {"time_ns": 81, "action": "ratchet_gripper_opened"},
            {"time_ns": 99, "action": "ratchet_head_regrasped"},
            {"time_ns": 120, "action": "seated_release_opened"},
        ]
        phases = build_phase_boundaries(events, actions)
        self.assertEqual(
            [phase.name for phase in phases],
            [
                "approach_open",
                "pickup_to_support_closed",
                "reposition_head_open",
                "thread_drive_1",
                "ratchet_reset_1",
                "thread_drive_2",
                "retreat_open",
            ],
        )
        self.assertTrue(phases[3].contact_critical)
        self.assertFalse(phases[-1].contact_critical)


if __name__ == "__main__":
    unittest.main()
