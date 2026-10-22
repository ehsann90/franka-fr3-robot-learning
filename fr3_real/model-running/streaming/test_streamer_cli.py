"""Offline CLI regression tests for the reference streamer."""

import os
import subprocess
import unittest
from pathlib import Path


DEFAULT_BINARY = (Path(__file__).resolve().parents[3] / ".local/build/fr3-streaming-ros-libfranka/fr3_reference_joint_velocity_streamer")
BINARY = Path(os.environ.get("FR3_STREAMER_BINARY", DEFAULT_BINARY))


class StreamerCliTest(unittest.TestCase):
    def invoke(self, *options):
        command = [str(BINARY), "unused-robot-ip", "--check-args", *options]
        result = subprocess.run(command, capture_output=True, text=True, timeout=2.0)
        return result.returncode, result.stdout + result.stderr

    def test_each_scalar_limit_and_combined_values(self):
        for option, value, field, count in (("--velocity-caps", "0.05", "velocity_caps", 7),
                                            ("--acceleration-caps", "0.50", "acceleration_caps", 7),
                                            ("--jerk-caps", "5.0", "jerk_caps", 7),
                                            ("--guard-margin", "0.005", "guard_margin", 3)):
            with self.subTest(option=option):
                code, output = self.invoke(option, value)
                self.assertEqual(code, 0, output)
                line = next(line for line in output.splitlines() if line.startswith(f"{field}="))
                self.assertEqual([float(item) for item in line.split("[", 1)[1].rstrip("]").split(",")],
                                 [float(value)] * count)
        code, output = self.invoke("--velocity-caps", "0.05", "--acceleration-caps", "0.50",
                                   "--jerk-caps", "5.0", "--guard-margin", "0.005")
        self.assertEqual(code, 0, output)

    def test_defaults_are_preserved(self):
        code, output = self.invoke()
        self.assertEqual(code, 0, output)
        self.assertIn("velocity_caps=[0.1,0.1,0.1,0.1,0.1,0.1,0.1]", output)
        self.assertIn("acceleration_caps=[0.75,0.75,0.75,0.75,0.75,0.75,0.75]", output)
        self.assertIn("guard_margin=[0.005,0.005,0.002]", output)

    def test_missing_malformed_and_unknown(self):
        for option in ("--velocity-caps", "--acceleration-caps", "--jerk-caps", "--guard-margin"):
            with self.subTest(option=option):
                code, output = self.invoke(option)
                self.assertNotEqual(code, 0)
                self.assertIn("Missing value", output)
                code, output = self.invoke(option, "abc")
                self.assertNotEqual(code, 0)
                code, output = self.invoke(option, "0.05junk")
                self.assertNotEqual(code, 0)
        code, output = self.invoke("--unknown")
        self.assertNotEqual(code, 0)
        self.assertIn("Unknown option", output)


if __name__ == "__main__":
    unittest.main()
