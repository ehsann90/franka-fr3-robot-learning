"""Validation for read-only, operator-selected bolt ratchet endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from .transforms import validate_transform


def summarize_ratchet_limits(
    prewind_snapshot: dict[str, Any], forward_snapshot: dict[str, Any]
) -> dict[str, Any]:
    q0 = np.asarray(prewind_snapshot["q"], dtype=np.float64).reshape(-1)
    q1 = np.asarray(forward_snapshot["q"], dtype=np.float64).reshape(-1)
    if q0.size != 7 or q1.size != 7 or not np.isfinite(q0).all() or not np.isfinite(q1).all():
        raise ValueError("ratchet snapshots require seven finite joint positions")
    T0 = validate_transform(prewind_snapshot["T_base_ee"], name="prewind.T_base_ee")
    T1 = validate_transform(forward_snapshot["T_base_ee"], name="forward.T_base_ee")
    q_delta = q1 - q0
    if q_delta[6] <= 0.0:
        raise ValueError("forward tightening endpoint must have greater q7 than prewind endpoint")
    relative_R = T0[:3, :3].T @ T1[:3, :3]
    wrapped_tool_rotation = float(np.arctan2(relative_R[1, 0], relative_R[0, 0]))
    return {
        "q7_prewind_rad": float(q0[6]),
        "q7_forward_rad": float(q1[6]),
        "q7_available_forward_range_rad": float(q_delta[6]),
        "q7_available_forward_range_deg": float(np.degrees(q_delta[6])),
        "wrapped_endpoint_tool_Kz_rotation_rad": wrapped_tool_rotation,
        "wrapped_endpoint_tool_Kz_rotation_deg": float(np.degrees(wrapped_tool_rotation)),
        "maximum_other_joint_change_rad": float(np.max(np.abs(q_delta[:6]))),
        "endpoint_translation_change_m": float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])),
        "limits_are_operator_selected_cable_safe_limits": True,
        "limits_are_not_robot_mechanical_limits": True,
    }
