#!/usr/bin/env python3
"""Launch the reference FCI streamer with explicitly reviewed workspace bounds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from workstation_config import load_commissioning, load_workstation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstation-config", type=Path, required=True)
    parser.add_argument("--commissioning-record", type=Path, required=True)
    args = parser.parse_args()
    try:
        profile = load_workstation(args.workstation_config)
        record = load_commissioning(args.commissioning_record, profile)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    repository = Path(__file__).resolve().parents[3]
    binary = Path(os.environ.get("FR3_STREAMER_BINARY", repository / ".local/build/fr3-streaming-ros-libfranka/fr3_reference_joint_velocity_streamer"))
    if not binary.is_file():
        parser.error(f"Reference streamer binary not found: {binary}")
    lower, upper = record["workspace"]["min"], record["workspace"]["max"]
    command = [str(binary), profile["robot_ip"], "--policy-hz", str(profile["policy_hz"])]
    for axis, minimum, maximum in zip("xyz", lower, upper):
        command += [f"--min-{axis}", str(minimum), f"--max-{axis}", str(maximum)]
    print(f"[LAB] profile={profile['name']} checkpoint={profile['checkpoint']}", flush=True)
    print(f"[LAB] approved workspace={record['workspace']} reviewer={record['reviewed_by']}", flush=True)
    os.execv(str(binary), command)


if __name__ == "__main__":
    main()
