"""Offline segmentation and validation for full Pilot-guided bolt episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PhaseBoundary:
    name: str
    start_time_ns: int
    stop_time_ns: int
    contact_critical: bool
    gripper_state: str


def nearest_index(times_ns: np.ndarray, query_ns: int) -> int:
    times = np.asarray(times_ns, dtype=np.int64)
    if times.ndim != 1 or not len(times):
        raise ValueError("times_ns must be a non-empty vector")
    index = int(np.searchsorted(times, int(query_ns)))
    if index <= 0:
        return 0
    if index >= len(times):
        return len(times) - 1
    return index - 1 if query_ns - times[index - 1] <= times[index] - query_ns else index


def build_phase_boundaries(events: Iterable[dict], actions: Iterable[dict]) -> list[PhaseBoundary]:
    events = list(events)
    actions = list(actions)
    event_by_name: dict[str, list[dict]] = {}
    for event in events:
        event_by_name.setdefault(event["event"], []).append(event)
    action_by_name: dict[str, list[dict]] = {}
    for action in actions:
        action_by_name.setdefault(action["action"], []).append(action)

    required_events = (
        "full_bolt_demo_started",
        "pickup_grasp_pose_reached",
        "bolt_self_supported",
        "head_regrasp_pose_reached",
        "full_bolt_demo_finished",
    )
    missing = [name for name in required_events if not event_by_name.get(name)]
    if missing:
        raise ValueError(f"missing required events: {missing}")
    for name in ("pickup_gripper_closed", "drop_gripper_opened", "head_gripper_closed", "seated_release_opened"):
        if not action_by_name.get(name):
            raise ValueError(f"missing required gripper action: {name}")

    boundaries = [
        PhaseBoundary(
            "approach_open",
            event_by_name["full_bolt_demo_started"][0]["time_ns"],
            event_by_name["pickup_grasp_pose_reached"][0]["time_ns"],
            False,
            "open",
        ),
        PhaseBoundary(
            "pickup_to_support_closed",
            action_by_name["pickup_gripper_closed"][0]["time_ns"],
            event_by_name["bolt_self_supported"][0]["time_ns"],
            False,
            "pickup_closed",
        ),
        PhaseBoundary(
            "reposition_head_open",
            action_by_name["drop_gripper_opened"][0]["time_ns"],
            event_by_name["head_regrasp_pose_reached"][0]["time_ns"],
            False,
            "open",
        ),
    ]

    starts = event_by_name.get("thread_drive_started", [])
    stops = event_by_name.get("thread_drive_complete", [])
    if len(starts) != len(stops) or not starts:
        raise ValueError("thread drive start/stop events are missing or unbalanced")
    reset_opens = action_by_name.get("ratchet_gripper_opened", [])
    reset_closes = action_by_name.get("ratchet_head_regrasped", [])
    if len(reset_opens) != len(reset_closes):
        raise ValueError("ratchet open/regrasp actions are unbalanced")
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True), start=1):
        cycle = int(start.get("payload", {}).get("cycle", index))
        if int(stop.get("payload", {}).get("cycle", cycle)) != cycle:
            raise ValueError("thread drive cycle numbering is inconsistent")
        boundaries.append(
            PhaseBoundary(
                f"thread_drive_{cycle}",
                start["time_ns"],
                stop["time_ns"],
                True,
                "head_closed",
            )
        )
        if index <= len(reset_opens):
            boundaries.append(
                PhaseBoundary(
                    f"ratchet_reset_{cycle}",
                    reset_opens[index - 1]["time_ns"],
                    reset_closes[index - 1]["time_ns"],
                    True,
                    "open_then_head_closed",
                )
            )
    boundaries.append(
        PhaseBoundary(
            "retreat_open",
            action_by_name["seated_release_opened"][0]["time_ns"],
            event_by_name["full_bolt_demo_finished"][0]["time_ns"],
            False,
            "open",
        )
    )
    if any(phase.stop_time_ns <= phase.start_time_ns for phase in boundaries):
        raise ValueError("one or more replay phases have non-positive duration")
    return boundaries


def summarize_phase(phase: PhaseBoundary, states: list[dict]) -> dict:
    times = np.asarray([row["time_ns"] for row in states], dtype=np.int64)
    start = nearest_index(times, phase.start_time_ns)
    stop = nearest_index(times, phase.stop_time_ns)
    if stop <= start:
        raise ValueError(f"phase {phase.name} has too few state samples")
    rows = states[start : stop + 1]
    q = np.asarray([row["q"] for row in rows], dtype=np.float64)
    dq = np.asarray([row["dq"] for row in rows], dtype=np.float64)
    xyz = np.asarray([row["ee_translation"] for row in rows], dtype=np.float64)
    wrench = np.asarray([row["K_F_ext_hat_K"] for row in rows], dtype=np.float64)
    return {
        "name": phase.name,
        "contact_critical": phase.contact_critical,
        "gripper_state": phase.gripper_state,
        "start_state_index": start,
        "stop_state_index": stop,
        "sample_count": len(rows),
        "duration_s": (times[stop] - times[start]) / 1e9,
        "start_q": q[0].tolist(),
        "stop_q": q[-1].tolist(),
        "q7_start_rad": float(q[0, 6]),
        "q7_stop_rad": float(q[-1, 6]),
        "q7_delta_rad": float(q[-1, 6] - q[0, 6]),
        "q7_min_rad": float(q[:, 6].min()),
        "q7_max_rad": float(q[:, 6].max()),
        "maximum_abs_dq_rad_s": float(np.abs(dq).max()),
        "xyz_min_m": xyz.min(axis=0).tolist(),
        "xyz_max_m": xyz.max(axis=0).tolist(),
        "tcp_translation_m": float(np.linalg.norm(xyz[-1] - xyz[0])),
        "K_Fz_range_N": [float(wrench[:, 2].min()), float(wrench[:, 2].max())],
        "K_Tz_range_Nm": [float(wrench[:, 5].min()), float(wrench[:, 5].max())],
    }
