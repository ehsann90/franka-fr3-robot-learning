import unittest

import numpy as np

from expert_demo.thread_engagement import (
    _accumulated_signed_kz_rotation,
    summarize_thread_engagement,
)


def row(index, yaw, z, fz, tz, q7=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[2, 3] = z
    return {
        "monotonic_time_ns": index * 1_000_000_000,
        "T_base_ee": T.tolist(),
        "K_F_ext_hat_K": [0, 0, fz, 0, 0, tz],
        "dq": [0] * 7,
        "q": [0, 0, 0, 0, 0, 0, q7],
    }


class ThreadEngagementTests(unittest.TestCase):
    def test_rotation_accumulation_does_not_wrap_at_pi(self):
        samples = [row(i, yaw, 0, 0, 0) for i, yaw in enumerate((0.0, 1.0, 2.0, 3.0, 3.2))]
        self.assertAlmostEqual(_accumulated_signed_kz_rotation(samples), 3.2)

    def test_summary_recovers_rotation_progress_and_wrench_delta(self):
        samples = [row(i, 0, 0, 1, 0.1, 0.0) for i in range(15)]
        # Prewind, then two +0.1 rad / +1 mm tightening drives separated by
        # an empty-wrist reset. The final phase is stationary and engaged.
        samples[1] = row(1, -1.0, 0, 1, 0.1, -1.0)
        samples[2] = row(2, -1.0, 0, 1, 0.1, -1.0)
        samples[3] = row(3, -1.0, 0, 1, 0.1, -1.0)
        samples[4] = row(4, -1.0, 0, 1, 0.1, -1.0)
        samples[5] = row(5, -0.9, 0.001, 2, 0.2, -0.9)
        samples[6] = row(6, -0.9, 0.001, 2, 0.2, -0.9)
        samples[7] = row(7, -1.0, 0.001, 1, 0.1, -1.0)
        samples[8] = row(8, -1.0, 0.001, 1, 0.1, -1.0)
        samples[9] = row(9, -0.9, 0.002, 2, 0.2, -0.9)
        for i in range(10, 15):
            samples[i] = row(i, -0.9, 0.002, 2, 0.3, -0.9)
        specs = [
            ("wrist_prewind_started", 0, {}),
            ("wrist_prewind_complete", 1, {}),
            ("unloaded_baseline_started", 2, {}),
            ("unloaded_baseline_complete", 3, {}),
            ("ratchet_drive_started", 4, {"cycle": 1}),
            ("ratchet_drive_complete", 5, {"cycle": 1, "outcome": "continue"}),
            ("ratchet_reset_started", 6, {"cycle": 1}),
            ("ratchet_reset_complete", 7, {"cycle": 1}),
            ("ratchet_drive_started", 8, {"cycle": 2}),
            ("ratchet_drive_complete", 9, {"cycle": 2, "outcome": "engaged"}),
            ("engaged_stationary_started", 10, {}),
            ("engaged_stationary_complete", 14, {}),
        ]
        events = [
            {"event": name, "monotonic_time_ns": second * 1_000_000_000, "payload": payload}
            for name, second, payload in specs
        ]
        summary = summarize_thread_engagement(samples, events)
        self.assertAlmostEqual(summary["signed_rotation_about_positive_Kz_rad"], 0.2)
        self.assertAlmostEqual(summary["wrist_prewind_rotation_about_positive_Kz_rad"], -1.0)
        self.assertAlmostEqual(summary["wrist_prewind_q7_delta_rad"], -1.0)
        self.assertAlmostEqual(summary["axial_progress_along_positive_Kz_m"], 0.002)
        self.assertEqual(summary["ratchet_cycle_count"], 2)
        self.assertAlmostEqual(summary["engaged_axial_force_delta_N"], 1.0)
        self.assertAlmostEqual(summary["engaged_axial_torque_delta_Nm"], 0.2)
        self.assertFalse(summary["thresholds_derived"])


if __name__ == "__main__":
    unittest.main()
