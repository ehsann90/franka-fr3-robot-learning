#!/usr/bin/env python3
"""Read one streamer state and both RealSense frames; send no commands."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
from openpi_client import image_tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadow_fr3_policy import read_rgb, start_camera
from streaming.stream_protocol import receive_latest_state
from streaming.workstation_config import load_workstation

JOINT_MIN = np.array([-2.6237, -1.6637, -2.7807, -2.9221, -2.6865, 0.6645, -2.8959])
JOINT_MAX = np.array([2.6237, 1.6637, 2.7807, -0.2718, 2.6865, 4.3969, 2.8959])
ORIGINAL_MIN = np.array([0.18, -0.50, 0.03])
ORIGINAL_MAX = np.array([0.75, 0.45, 0.65])
ORIGINAL_MARGIN = np.array([0.005, 0.005, 0.002])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workstation-config", type=Path, required=True)
    parser.add_argument("--state-port", type=int, default=51001)
    parser.add_argument("--skip-cameras", action="store_true")
    args = parser.parse_args()
    profile = load_workstation(args.workstation_config)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", args.state_port))
        sock.settimeout(3.0)
        state, _ = receive_latest_state(sock)
    record = {
        "q_rad": state.q.tolist(),
        "dq_rad_s": state.dq.tolist(),
        "O_T_EE_translation_m": state.tcp.tolist(),
        "joint_guard_inside": bool(np.all((state.q > JOINT_MIN) & (state.q < JOINT_MAX))),
        "original_workspace_guard_inside": bool(np.all(
            (state.tcp > ORIGINAL_MIN + ORIGINAL_MARGIN) &
            (state.tcp < ORIGINAL_MAX - ORIGINAL_MARGIN))),
        "original_workspace_commissioned_here": False,
        "stop_reason": state.stop_reason_name,
    }
    if not args.skip_cameras:
        frames = {}
        cameras = {}
        try:
            for name in ("exterior", "wrist"):
                cameras[name] = start_camera(profile[f"{name}_serial"], 640, 480, 60)
                rgb = read_rgb(cameras[name])
                processed = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, 224, 224))
                frames[name] = {"serial": profile[f"{name}_serial"], "rgb_shape": list(rgb.shape),
                                "policy_shape": list(processed.shape), "policy_dtype": str(processed.dtype)}
        finally:
            for camera in cameras.values():
                camera.stop()
        record["cameras"] = frames
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
