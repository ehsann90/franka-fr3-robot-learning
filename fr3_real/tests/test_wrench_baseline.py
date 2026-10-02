import unittest

import numpy as np

from expert_demo.wrench_baseline import summarize_wrench_baseline


def sample(index: int) -> dict:
    return {
        "monotonic_time_ns": 1_000_000_000 + index * 20_000_000,
        "dq": [0.01 * index] * 7,
        "O_F_ext_hat_K": (np.arange(6) + index).tolist(),
        "K_F_ext_hat_K": (np.arange(6) - index).tolist(),
    }


class WrenchBaselineTests(unittest.TestCase):
    def test_summary_preserves_uncommissioned_axis_and_threshold_status(self):
        summary = summarize_wrench_baseline([sample(index) for index in range(5)])
        self.assertEqual(summary["sample_count"], 5)
        self.assertAlmostEqual(summary["achieved_sample_hz"], 50.0)
        self.assertAlmostEqual(summary["maximum_measured_abs_dq_rad_s"], 0.04)
        self.assertFalse(summary["thresholds_derived"])
        self.assertFalse(summary["axis_convention_experimentally_verified"])
        self.assertEqual(
            len(summary["wrench_statistics"]["K_F_ext_hat_K"]["components"]), 6
        )

    def test_summary_rejects_missing_wrench_signal(self):
        samples = [sample(0), sample(1)]
        del samples[1]["K_F_ext_hat_K"]
        with self.assertRaisesRegex(ValueError, "K_F_ext_hat_K"):
            summarize_wrench_baseline(samples)

    def test_summary_rejects_non_monotonic_timestamps(self):
        samples = [sample(0), sample(0)]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            summarize_wrench_baseline(samples)
