import unittest

import numpy as np

from expert_demo.ratchet_limits import summarize_ratchet_limits


def snapshot(q7, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return {"q": [0, 0, 0, 0, 0, 0, q7], "T_base_ee": T.tolist()}


class RatchetLimitTests(unittest.TestCase):
    def test_positive_q7_interval_is_reported(self):
        result = summarize_ratchet_limits(snapshot(-0.2, -0.2), snapshot(2.8, 2.8))
        self.assertAlmostEqual(result["q7_available_forward_range_rad"], 3.0)
        self.assertAlmostEqual(result["maximum_other_joint_change_rad"], 0.0)

    def test_reversed_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_ratchet_limits(snapshot(1.0, 1.0), snapshot(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
