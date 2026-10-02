"""Offline tests for bolt held-width calibration persistence."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


FR3_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = FR3_ROOT / "franka-tests" / "calibrate_bolt_held_width.py"
SPEC = importlib.util.spec_from_file_location("bolt_width", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BoltHeldWidthTests(unittest.TestCase):
    def test_summary_uses_median_and_range(self) -> None:
        summary = MODULE.summarize_widths([0.012, 0.011, 0.013, 0.012, 0.012])
        self.assertAlmostEqual(summary["median_width_m"], 0.012)
        self.assertAlmostEqual(summary["peak_to_peak_width_m"], 0.002)

    def test_calibration_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bolt.json"
            MODULE.write_new_calibration(path, {"schema_version": 1})
            self.assertEqual(json.loads(path.read_text())["schema_version"], 1)
            with self.assertRaises(FileExistsError):
                MODULE.write_new_calibration(path, {"schema_version": 2})


if __name__ == "__main__":
    unittest.main()
