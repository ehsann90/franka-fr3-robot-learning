"""Static guards for the empty-Hand bolt wrist-rotation commissioning test."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_guarded_q7_target(
    current_q: Sequence[float],
    prewind_q: Sequence[float],
    *,
    forward_angle_rad: float,
    cable_q7_min_rad: float,
    cable_q7_max_rad: float,
    minimum_forward_reserve_after_test_rad: float,
    other_joint_start_tolerance_rad: float,
    maximum_joint_delta_rad: float,
    mechanical_min: Sequence[float],
    mechanical_max: Sequence[float],
    mechanical_margin_rad: float,
) -> tuple[np.ndarray, list[str]]:
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    prewind = np.asarray(prewind_q, dtype=np.float64).reshape(-1)
    lower = np.asarray(mechanical_min, dtype=np.float64).reshape(-1) + mechanical_margin_rad
    upper = np.asarray(mechanical_max, dtype=np.float64).reshape(-1) - mechanical_margin_rad
    if any(values.size != 7 or not np.isfinite(values).all() for values in (current, prewind, lower, upper)):
        raise ValueError("joint vectors must contain seven finite values")
    if not 0.0 < forward_angle_rad <= maximum_joint_delta_rad:
        raise ValueError("forward angle must be positive and within maximum joint delta")
    if cable_q7_min_rad >= cable_q7_max_rad:
        raise ValueError("cable-safe q7 interval is invalid")
    target = current.copy()
    target[6] += forward_angle_rad
    blockers: list[str] = []
    if float(np.max(np.abs(current[:6] - prewind[:6]))) > other_joint_start_tolerance_rad:
        blockers.append("live joints 1-6 are not near the calibrated prewind geometry")
    if current[6] < cable_q7_min_rad or current[6] > cable_q7_max_rad:
        blockers.append("live q7 is outside the calibrated cable-safe interval")
    if target[6] > cable_q7_max_rad or target[6] < cable_q7_min_rad:
        blockers.append("test target q7 leaves the calibrated cable-safe interval")
    if cable_q7_max_rad - target[6] < minimum_forward_reserve_after_test_rad:
        blockers.append("test target does not retain the configured forward cable reserve")
    if np.any(current < lower) or np.any(current > upper):
        blockers.append("live joints violate guarded mechanical ranges")
    if np.any(target < lower) or np.any(target > upper):
        blockers.append("test target violates guarded mechanical ranges")
    return target, blockers


def build_guarded_reverse_q7_target(
    current_q: Sequence[float],
    expected_start_q: Sequence[float],
    *,
    reverse_angle_rad: float,
    cable_q7_min_rad: float,
    cable_q7_max_rad: float,
    minimum_backward_reserve_after_test_rad: float,
    start_q7_tolerance_rad: float,
    other_joint_start_tolerance_rad: float,
    maximum_joint_delta_rad: float,
    mechanical_min: Sequence[float],
    mechanical_max: Sequence[float],
    mechanical_margin_rad: float,
) -> tuple[np.ndarray, list[str]]:
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected_start_q, dtype=np.float64).reshape(-1)
    lower = np.asarray(mechanical_min, dtype=np.float64).reshape(-1) + mechanical_margin_rad
    upper = np.asarray(mechanical_max, dtype=np.float64).reshape(-1) - mechanical_margin_rad
    if any(values.size != 7 or not np.isfinite(values).all() for values in (current, expected, lower, upper)):
        raise ValueError("joint vectors must contain seven finite values")
    if not 0.0 < reverse_angle_rad <= maximum_joint_delta_rad:
        raise ValueError("reverse angle must be positive and within maximum joint delta")
    if cable_q7_min_rad >= cable_q7_max_rad:
        raise ValueError("cable-safe q7 interval is invalid")
    target = current.copy()
    target[6] -= reverse_angle_rad
    blockers: list[str] = []
    if abs(current[6] - expected[6]) > start_q7_tolerance_rad:
        blockers.append("live q7 is not at the commissioned forward-test endpoint")
    if float(np.max(np.abs(current[:6] - expected[:6]))) > other_joint_start_tolerance_rad:
        blockers.append("live joints 1-6 are not near the commissioned forward-test geometry")
    if current[6] < cable_q7_min_rad or current[6] > cable_q7_max_rad:
        blockers.append("live q7 is outside the calibrated cable-safe interval")
    if target[6] < cable_q7_min_rad or target[6] > cable_q7_max_rad:
        blockers.append("reverse target q7 leaves the calibrated cable-safe interval")
    if target[6] - cable_q7_min_rad < minimum_backward_reserve_after_test_rad:
        blockers.append("reverse target does not retain the configured backward cable reserve")
    if np.any(current < lower) or np.any(current > upper):
        blockers.append("live joints violate guarded mechanical ranges")
    if np.any(target < lower) or np.any(target > upper):
        blockers.append("reverse target violates guarded mechanical ranges")
    return target, blockers
