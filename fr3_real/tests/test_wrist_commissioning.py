import unittest

from expert_demo.wrist_commissioning import (
    build_guarded_q7_target,
    build_guarded_reverse_q7_target,
)


MINIMUM = [-2.7, -1.7, -2.9, -3.0, -2.8, 0.5, -3.0]
MAXIMUM = [2.7, 1.7, 2.9, -0.1, 2.8, 4.5, 3.1]


class WristCommissioningTests(unittest.TestCase):
    def call(self, current, prewind=None, angle=0.2, cable_max=3.0, reserve=1.0):
        return build_guarded_q7_target(
            current,
            current if prewind is None else prewind,
            forward_angle_rad=angle,
            cable_q7_min_rad=-0.15,
            cable_q7_max_rad=cable_max,
            minimum_forward_reserve_after_test_rad=reserve,
            other_joint_start_tolerance_rad=0.1,
            maximum_joint_delta_rad=0.3,
            mechanical_min=MINIMUM,
            mechanical_max=MAXIMUM,
            mechanical_margin_rad=0.05,
        )

    def test_builds_q7_only_target(self):
        current = [0, 0, 0, -1, 0, 1, -0.15]
        target, blockers = self.call(current)
        self.assertEqual(blockers, [])
        self.assertEqual(target[:6].tolist(), current[:6])
        self.assertAlmostEqual(target[6], 0.05)

    def test_accepts_safe_start_inside_interval_with_reserve(self):
        _, blockers = self.call([0, 0, 0, -1, 0, 1, 0.1])
        self.assertEqual(blockers, [])

    def test_rejects_start_outside_interval(self):
        _, blockers = self.call([0, 0, 0, -1, 0, 1, -0.3])
        self.assertIn("live q7 is outside the calibrated cable-safe interval", blockers)

    def test_rejects_target_outside_cable_interval(self):
        _, blockers = self.call(
            [0, 0, 0, -1, 0, 1, -0.15], angle=0.2, cable_max=0.0, reserve=0.0
        )
        self.assertIn("test target q7 leaves the calibrated cable-safe interval", blockers)

    def test_requires_forward_reserve(self):
        _, blockers = self.call([0, 0, 0, -1, 0, 1, 1.5], angle=0.2, reserve=1.5)
        self.assertIn(
            "test target does not retain the configured forward cable reserve", blockers
        )

    def test_builds_reverse_q7_only_target(self):
        current = [0, 0, 0, -1, 0, 1, 1.6]
        target, blockers = build_guarded_reverse_q7_target(
            current,
            current,
            reverse_angle_rad=1.5,
            cable_q7_min_rad=-0.15,
            cable_q7_max_rad=3.0,
            minimum_backward_reserve_after_test_rad=0.15,
            start_q7_tolerance_rad=0.03,
            other_joint_start_tolerance_rad=0.1,
            maximum_joint_delta_rad=1.65,
            mechanical_min=MINIMUM,
            mechanical_max=MAXIMUM,
            mechanical_margin_rad=0.05,
        )
        self.assertEqual(blockers, [])
        self.assertEqual(target[:6].tolist(), current[:6])
        self.assertAlmostEqual(target[6], 0.1)

    def test_reverse_requires_backward_reserve(self):
        current = [0, 0, 0, -1, 0, 1, 1.4]
        _, blockers = build_guarded_reverse_q7_target(
            current,
            current,
            reverse_angle_rad=1.5,
            cable_q7_min_rad=-0.15,
            cable_q7_max_rad=3.0,
            minimum_backward_reserve_after_test_rad=0.1,
            start_q7_tolerance_rad=0.03,
            other_joint_start_tolerance_rad=0.1,
            maximum_joint_delta_rad=1.65,
            mechanical_min=MINIMUM,
            mechanical_max=MAXIMUM,
            mechanical_margin_rad=0.05,
        )
        self.assertIn(
            "reverse target does not retain the configured backward cable reserve", blockers
        )


if __name__ == "__main__":
    unittest.main()
