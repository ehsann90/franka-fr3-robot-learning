"""Offline regression coverage for the lab policy server identity check."""

import unittest
from pathlib import Path

from stream_policy_chunks import validate_server_metadata
from workstation_config import load_workstation


PROFILE = load_workstation(
    Path(__file__).resolve().parents[2] / "configs/fr3_lab_pi05_v1.json"
)


class PolicyMetadataTest(unittest.TestCase):
    def test_verified_nested_docker_metadata(self):
        metadata = {
            "saps_seeded_sampling": {
                "policy_checkpoint": "/openpi_assets/pi05-fr3-v3-12000",
                "policy_config_name": "pi05_droid",
            }
        }
        self.assertEqual(validate_server_metadata(metadata, PROFILE), [])

    def test_profile_checkpoint_identifier(self):
        metadata = {
            "saps_seeded_sampling": {
                "policy_checkpoint": "msbutt1/pi05-fr3-v3-12000",
                "policy_config_name": "pi05_droid",
            }
        }
        self.assertEqual(validate_server_metadata(metadata, PROFILE), [])

    def test_wrong_checkpoint(self):
        for checkpoint in (
            "/openpi_assets/pi05-fr3-v2-12000",
            "someone_else/pi05-fr3-v3-12000",
        ):
            with self.subTest(checkpoint=checkpoint):
                metadata = {"saps_seeded_sampling": {
                    "policy_checkpoint": checkpoint,
                    "policy_config_name": "pi05_droid",
                }}
                with self.assertRaisesRegex(RuntimeError, "expected checkpoint"):
                    validate_server_metadata(metadata, PROFILE)

    def test_wrong_policy_configuration(self):
        metadata = {"saps_seeded_sampling": {
            "policy_checkpoint": "/openpi_assets/pi05-fr3-v3-12000",
            "policy_config_name": "pi05_fr3_other",
        }}
        with self.assertRaisesRegex(RuntimeError, "expected policy configuration"):
            validate_server_metadata(metadata, PROFILE)

    def test_missing_metadata_keeps_explicit_warnings(self):
        for metadata in ({}, None):
            with self.subTest(metadata=metadata):
                warnings = validate_server_metadata(metadata, PROFILE)
                self.assertEqual(len(warnings), 2)
                self.assertIn("checkpoint identity", warnings[0])
                self.assertIn("policy configuration", warnings[1])


if __name__ == "__main__":
    unittest.main()
