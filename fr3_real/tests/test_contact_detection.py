import unittest

from expert_demo.contact_detection import first_sustained_crossing


class ContactDetectionTests(unittest.TestCase):
    def test_requires_configured_persistence(self):
        values = [0.0, 0.6, 0.7, 0.1, 0.6, 0.7, 0.8]
        self.assertEqual(first_sustained_crossing(values, 0.0, 0.5, 3), 4)

    def test_rejects_single_spike(self):
        self.assertIsNone(first_sustained_crossing([0.0, 1.0, 0.0], 0.0, 0.5, 2))

    def test_relative_baseline_is_applied(self):
        self.assertEqual(first_sustained_crossing([-2.0, -1.4, -1.3], -2.0, 0.5, 2), 1)


if __name__ == "__main__":
    unittest.main()
