"""Offline tests for explicit workstation selection and commissioning gate."""

import tempfile
import json
import os
import subprocess
import unittest
from pathlib import Path

from workstation_config import load_commissioning, load_workstation


PROFILE = Path(__file__).resolve().parents[2] / "configs/fr3_lab_pi05_v1.json"


class WorkstationConfigTest(unittest.TestCase):
    def test_profile_and_rejected_uncommissioned_record(self):
        profile = load_workstation(PROFILE)
        self.assertEqual(profile["robot_ip"], "192.170.10.101")
        self.assertFalse(profile["workspace_commissioned"])
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            record.write_text('{"workstation":"fr3_lab_pi05_v1","approved":false}')
            with self.assertRaises(ValueError):
                load_commissioning(record, profile)

    def test_requires_measured_pose_inside_bounds(self):
        profile = load_workstation(PROFILE)
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            record.write_text('''{"workstation":"fr3_lab_pi05_v1","approved":true,
                "reviewed_by":"test", "workspace":{"min":[0.18,-0.5,0.03],"max":[0.75,0.45,0.65]},
                "tcp_xyz_m":[0.0,0.0,0.2],"joint_positions_rad":[0,0,0,-1.5,0,2,0]}''')
            with self.assertRaisesRegex(ValueError, "workspace guard"):
                load_commissioning(record, profile)

    def test_approved_record_builds_explicit_streamer_command(self):
        profile = load_workstation(PROFILE)
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "record.json"
            record.write_text(json.dumps({
                "workstation": profile["name"], "approved": True, "reviewed_by": "offline test",
                "workspace": {"min": [0.18, -0.5, 0.03], "max": [0.75, 0.45, 0.65]},
                "tcp_xyz_m": [0.4, 0.0, 0.2],
                "joint_positions_rad": [0, 0, 0, -1.5, 0, 2, 0],
            }))
            self.assertEqual(load_commissioning(record, profile)["reviewed_by"], "offline test")
            runner = Path(__file__).with_name("run_lab_streamer.py")
            result = subprocess.run([os.sys.executable, str(runner), "--workstation-config",
                                     str(PROFILE), "--commissioning-record", str(record)],
                                    env={**os.environ, "FR3_STREAMER_BINARY": "/bin/echo"},
                                    capture_output=True, text=True, check=True)
            self.assertIn("192.170.10.101 --policy-hz 15.0", result.stdout)
            self.assertIn("--min-x 0.18 --max-x 0.75", result.stdout)


if __name__ == "__main__":
    unittest.main()
