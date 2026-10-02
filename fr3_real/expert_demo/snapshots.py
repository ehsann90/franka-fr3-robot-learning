"""Read-only FR3 state capture used by the pose teaching utility."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .transforms import quaternion_xyzw_from_rotation, validate_transform


class SnapshotReadError(RuntimeError):
    """Raised when a required robot or gripper state read does not complete."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _float_list(value: Any, expected_size: int, field_name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != expected_size:
        raise ValueError(f"{field_name} must contain {expected_size} values, got {array.size}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} contains NaN or infinite values")
    return array.tolist()


def _duration_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to_sec"):
        value = value.to_sec()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _optional_vector(state: Any, field_name: str, size: int) -> list[float] | None:
    if not hasattr(state, field_name):
        return None
    value = getattr(state, field_name)
    if value is None:
        return None
    return _float_list(value, size, field_name)


def _read_gripper_state(gripper: Any, attempts: int = 3, retry_delay_s: float = 0.1) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return gripper.state
        except Exception as error:  # Native Franky exceptions do not share one Python base type.
            last_error = error
            if attempt < attempts:
                time.sleep(retry_delay_s)
    raise SnapshotReadError(
        f"Franka Hand state timed out after {attempts} read-only attempts; "
        "no calibration pose was saved. Keep the arm stationary, check the Hand/network state, and retry. "
        f"Last error: {last_error}"
    ) from last_error


def capture_robot_snapshot(
    robot: Any,
    gripper: Any | None,
    *,
    snapshot_kind: str,
    notes: str = "",
) -> dict:
    """Capture one coherent best-effort snapshot without issuing commands.

    ``robot.state`` is fetched exactly once. Gripper state is fetched once and
    may be omitted by passing ``None``. No recovery, homing, guiding, motion,
    collision, or impedance method is called here.
    """

    try:
        state = robot.state
    except Exception as error:  # Native Franky exceptions do not share one Python base type.
        raise SnapshotReadError(f"FR3 state read failed; no calibration pose was saved: {error}") from error
    pose = state.O_T_EE
    pose_matrix = pose.matrix if hasattr(pose, "matrix") else pose
    T_base_ee = validate_transform(pose_matrix, name="state.O_T_EE")

    q = _float_list(state.q, 7, "q")
    dq = _float_list(state.dq, 7, "dq")
    snapshot: dict[str, Any] = {
        "snapshot_kind": snapshot_kind,
        "timestamp_utc": utc_now(),
        "host_time_ns": time.time_ns(),
        "robot_time_s": _duration_seconds(getattr(state, "time", None)),
        "q": q,
        "dq": dq,
        "T_base_ee": T_base_ee.tolist(),
        "translation_xyz_m": T_base_ee[:3, 3].tolist(),
        "orientation": {
            "quaternion_xyzw": quaternion_xyzw_from_rotation(T_base_ee[:3, :3]).tolist(),
            "quaternion_order": "xyzw",
        },
        "notes": notes,
    }

    for field_name, size in (
        ("q_d", 7),
        ("dq_d", 7),
        ("tau_J", 7),
        ("tau_J_d", 7),
        ("tau_ext_hat_filtered", 7),
        ("O_F_ext_hat_K", 6),
        ("K_F_ext_hat_K", 6),
    ):
        value = _optional_vector(state, field_name, size)
        if value is not None:
            snapshot[field_name] = value

    if gripper is None:
        snapshot["gripper"] = {"available": False}
    else:
        gripper_state = _read_gripper_state(gripper)
        snapshot["gripper"] = {
            "available": True,
            "width_m": float(gripper_state.width),
            "max_width_m": float(gripper_state.max_width),
            "is_grasped": bool(gripper_state.is_grasped),
            "temperature_c": int(gripper_state.temperature),
            "robot_time_s": _duration_seconds(getattr(gripper_state, "time", None)),
        }
    return snapshot
