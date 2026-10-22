"""Explicit lab deployment selection and workspace commissioning gate."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_workstation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "name", "robot_ip", "exterior_serial", "wrist_serial", "server_host",
        "server_port", "policy_hz", "checkpoint", "openpi_config", "workspace_commissioned",
    }
    if set(data) != required:
        raise ValueError(f"Workstation keys differ: missing={required - set(data)}, extra={set(data) - required}")
    if data["workspace_commissioned"] is not False:
        raise ValueError("Versioned workstation profile must retain workspace_commissioned=false")
    if data["openpi_config"] != "pi05_droid" or float(data["policy_hz"]) != 15.0:
        raise ValueError("This deployment expects pi05_droid at 15 Hz")
    if not 1 <= int(data["server_port"]) <= 65535:
        raise ValueError("Invalid policy server port")
    return data


def load_commissioning(path: Path, workstation: dict) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("workstation") != workstation["name"] or data.get("approved") is not True:
        raise ValueError("Commissioning record must name this workstation and be approved")
    bounds = data.get("workspace")
    if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
        raise ValueError("Commissioning record needs workspace min/max")
    lower, upper = bounds["min"], bounds["max"]
    if not all(isinstance(v, list) and len(v) == 3 for v in (lower, upper)):
        raise ValueError("Workspace min/max must each have three coordinates")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in lower + upper):
        raise ValueError("Workspace coordinates must be finite numbers")
    if not all(lo < hi for lo, hi in zip(lower, upper)):
        raise ValueError("Workspace minimum must be below maximum")
    if not data.get("tcp_xyz_m") or not data.get("joint_positions_rad") or not data.get("reviewed_by"):
        raise ValueError("Commissioning record needs measured pose, joints, and reviewer")
    tcp, joints = data["tcp_xyz_m"], data["joint_positions_rad"]
    if not isinstance(tcp, list) or len(tcp) != 3 or not isinstance(joints, list) or len(joints) != 7:
        raise ValueError("Measured pose and joints have wrong dimensions")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in tcp + joints):
        raise ValueError("Measured pose and joints must be finite")
    q_min = [-2.6237, -1.6637, -2.7807, -2.9221, -2.6865, 0.6645, -2.8959]
    q_max = [2.6237, 1.6637, 2.7807, -0.2718, 2.6865, 4.3969, 2.8959]
    if not all(lo < q < hi for lo, q, hi in zip(q_min, joints, q_max)):
        raise ValueError("Recorded joints fail the reference streamer joint guard")
    margin = [0.005, 0.005, 0.002]
    if not all(lo + m < x < hi - m for lo, x, hi, m in zip(lower, tcp, upper, margin)):
        raise ValueError("Recorded EE translation fails the reference streamer workspace guard")
    return data
