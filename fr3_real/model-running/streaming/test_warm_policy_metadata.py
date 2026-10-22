"""Exercise warm-up metadata checks offline with fake cameras, robot, and server."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import warm_policy_server as warmup


class FakeCamera:
    def stop(self):
        pass


class WarmPolicyMetadataTest(unittest.TestCase):
    def run_warmup(self, metadata):
        class FakePolicy:
            def __init__(self, *_args):
                pass

            def get_server_metadata(self):
                return metadata

            def infer(self, _observation):
                return {"actions": np.zeros((15, 8), dtype=np.float64)}

        franky = SimpleNamespace(
            Robot=lambda _ip: SimpleNamespace(state=SimpleNamespace(q=np.zeros(7))),
            Gripper=lambda _ip: SimpleNamespace(state=SimpleNamespace(width=0.08)),
        )
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.dict(sys.modules, {"franky": franky}), \
             mock.patch.object(warmup.websocket_client_policy, "WebsocketClientPolicy", FakePolicy), \
             mock.patch.object(warmup, "start_camera", return_value=FakeCamera()), \
             mock.patch.object(warmup, "read_rgb", return_value=np.zeros((480, 640, 3), dtype=np.uint8)), \
             mock.patch.object(sys, "argv", ["warm_policy_server.py", "--checkpoint-label",
                                            "msbutt1/pi05-fr3-v3-12000", "--iterations", "2",
                                            "--log-dir", directory]), redirect_stdout(output):
            warmup.main()
            records = [json.loads(line) for log in Path(directory).rglob("warmup.jsonl")
                       for line in log.read_text().splitlines()]
        return output.getvalue(), records

    def test_nested_docker_metadata_warms_and_keeps_logs(self):
        metadata = {"saps_seeded_sampling": {
            "policy_checkpoint": "/openpi_assets/pi05-fr3-v3-12000",
            "policy_config_name": "pi05_droid",
        }}
        output, records = self.run_warmup(metadata)
        self.assertIn("[WARMUP] ready", output)
        self.assertEqual([record["event"] for record in records],
                         ["server_metadata", "warmup_inference", "warmup_inference"])
        self.assertEqual(records[0]["metadata"], metadata)

    def test_wrong_policy_configuration_is_rejected(self):
        metadata = {"saps_seeded_sampling": {
            "policy_checkpoint": "/openpi_assets/pi05-fr3-v3-12000",
            "policy_config_name": "pi05_other",
        }}
        with self.assertRaisesRegex(RuntimeError, "expected policy configuration"):
            self.run_warmup(metadata)

    def test_wrong_checkpoint_is_rejected(self):
        metadata = {"saps_seeded_sampling": {
            "policy_checkpoint": "/openpi_assets/pi05-fr3-v2-12000",
            "policy_config_name": "pi05_droid",
        }}
        with self.assertRaisesRegex(RuntimeError, "expected checkpoint"):
            self.run_warmup(metadata)

    def test_missing_metadata_warns_explicitly(self):
        output, records = self.run_warmup({})
        self.assertIn("WARNING server does not advertise checkpoint identity", output)
        self.assertIn("WARNING server does not advertise policy configuration", output)
        self.assertEqual(len(records), 3)


if __name__ == "__main__":
    unittest.main()
