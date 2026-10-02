import unittest

from expert_demo.axial_load_calibration import summarize_axial_load_phases


def sample(index, fz):
    return {
        "monotonic_time_ns": index * 1_000_000_000,
        "dq": [0.0] * 7,
        "K_F_ext_hat_K": [0.0, 0.0, fz, 0.0, 0.0, 0.1 * fz],
    }


class AxialLoadCalibrationTests(unittest.TestCase):
    def test_phase_delta_uses_kz_and_derives_no_threshold(self):
        samples = [sample(i, value) for i, value in enumerate([1, 1, 1, 4, 4, 4, 1, 1, 1])]
        events = [
            {"event": "unloaded_baseline_started", "monotonic_time_ns": 0},
            {"event": "unloaded_baseline_complete", "monotonic_time_ns": 2_000_000_000},
            {"event": "axial_load_window_started", "monotonic_time_ns": 3_000_000_000},
            {"event": "axial_load_window_complete", "monotonic_time_ns": 5_000_000_000},
            {"event": "unloaded_recovery_started", "monotonic_time_ns": 6_000_000_000},
            {"event": "unloaded_recovery_complete", "monotonic_time_ns": 8_000_000_000},
        ]
        summary = summarize_axial_load_phases(samples, events)
        self.assertAlmostEqual(summary["axial_force_loaded_minus_unloaded_N"], 3.0)
        self.assertAlmostEqual(summary["axial_torque_loaded_minus_unloaded_Nm"], 0.3)
        self.assertEqual(summary["dominant_force_component_index"], 2)
        self.assertFalse(summary["thresholds_derived"])


if __name__ == "__main__":
    unittest.main()
