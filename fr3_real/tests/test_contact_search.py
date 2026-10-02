import unittest

import numpy as np

from expert_demo.contact_search import ContactSearchGuard, rotation_angle_rad


def guard():
    return ContactSearchGuard(
        baseline_force_n=-2.0,
        contact_delta_n=0.5,
        contact_persistence_s=0.1,
        hard_delta_n=2.0,
        maximum_travel_m=0.003,
        maximum_duration_s=8.0,
        start_xyz_m=np.zeros(3),
        axis_base=np.array([0.0, 0.0, -1.0]),
    )


class ContactSearchTests(unittest.TestCase):
    def test_contact_requires_continuous_persistence(self):
        item = guard()
        self.assertIsNone(item.update(0.0, -1.4, np.zeros(3)))
        self.assertIsNone(item.update(0.05, -2.0, np.zeros(3)))
        self.assertIsNone(item.update(0.10, -1.4, np.zeros(3)))
        self.assertIsNone(item.update(0.19, -1.4, np.zeros(3)))
        self.assertEqual(item.update(0.20, -1.4, np.zeros(3)), "contact_detected")

    def test_hard_force_has_priority(self):
        item = guard()
        self.assertEqual(item.update(0.0, 0.1, np.zeros(3)), "hard_axial_force_cutoff")

    def test_maximum_travel_and_timeout_stop(self):
        item = guard()
        self.assertEqual(
            item.update(0.5, -2.0, np.array([0.0, 0.0, -0.0031])),
            "maximum_travel",
        )
        timed = guard()
        self.assertEqual(timed.update(8.0, -2.0, np.zeros(3)), "timeout")

    def test_rotation_angle(self):
        self.assertAlmostEqual(rotation_angle_rad(np.eye(3), np.eye(3)), 0.0)
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(rotation_angle_rad(np.eye(3), R), np.pi / 2)


if __name__ == "__main__":
    unittest.main()
