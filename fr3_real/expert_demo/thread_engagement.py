"""Analysis of a Pilot-guided initial bolt thread-engagement calibration."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


THREAD_ENGAGEMENT_PREFIX_EVENTS = (
    "wrist_prewind_started",
    "wrist_prewind_complete",
    "unloaded_baseline_started",
    "unloaded_baseline_complete",
)

THREAD_ENGAGEMENT_SUFFIX_EVENTS = (
    "engaged_stationary_started",
    "engaged_stationary_complete",
)


def _window(
    samples: Sequence[dict[str, Any]], start_ns: int, stop_ns: int
) -> list[dict[str, Any]]:
    return [
        sample
        for sample in samples
        if start_ns <= int(sample["monotonic_time_ns"]) <= stop_ns
    ]


def _accumulated_signed_kz_rotation(rows: Sequence[dict[str, Any]]) -> float:
    """Accumulate local Kz rotation without the +/-pi endpoint ambiguity."""

    if len(rows) < 2:
        raise ValueError("rotation accumulation requires at least two samples")
    rotations = [np.asarray(row["T_base_ee"], dtype=np.float64)[:3, :3] for row in rows]
    result = 0.0
    for previous, current in zip(rotations[:-1], rotations[1:]):
        relative_R = previous.T @ current
        result += float(np.arctan2(relative_R[1, 0], relative_R[0, 0]))
    return result


def summarize_thread_engagement(
    samples: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    names = tuple(event["event"] for event in events)
    if names[:4] != THREAD_ENGAGEMENT_PREFIX_EVENTS:
        raise ValueError(f"unexpected event prefix {names[:4]}")
    if names[-2:] != THREAD_ENGAGEMENT_SUFFIX_EVENTS:
        raise ValueError(f"unexpected event suffix {names[-2:]}")
    event_times = {event["event"]: int(event["monotonic_time_ns"]) for event in events}
    prewind = _window(
        samples,
        event_times["wrist_prewind_started"],
        event_times["wrist_prewind_complete"],
    )
    baseline = _window(
        samples,
        event_times["unloaded_baseline_started"],
        event_times["unloaded_baseline_complete"],
    )
    engaged = _window(
        samples,
        event_times["engaged_stationary_started"],
        event_times["engaged_stationary_complete"],
    )
    if min(len(prewind), len(baseline), len(engaged)) < 2:
        raise ValueError("each calibration phase requires at least two samples")

    cycles: list[dict[str, Any]] = []
    cursor = 4
    while cursor < len(events) - 2:
        started = events[cursor]
        completed = events[cursor + 1] if cursor + 1 < len(events) else None
        if started["event"] != "ratchet_drive_started" or completed is None:
            raise ValueError(f"expected ratchet drive at event index {cursor}")
        if completed["event"] != "ratchet_drive_complete":
            raise ValueError(f"missing ratchet drive completion at event index {cursor + 1}")
        cycle = int(started.get("payload", {}).get("cycle", -1))
        if cycle != len(cycles) + 1 or int(completed.get("payload", {}).get("cycle", -1)) != cycle:
            raise ValueError("ratchet cycle numbers must be consecutive")
        outcome = completed.get("payload", {}).get("outcome")
        if outcome not in ("continue", "engaged"):
            raise ValueError(f"invalid ratchet drive outcome {outcome!r}")
        drive = _window(
            samples,
            int(started["monotonic_time_ns"]),
            int(completed["monotonic_time_ns"]),
        )
        if len(drive) < 2:
            raise ValueError(f"ratchet drive {cycle} requires at least two samples")
        reset: list[dict[str, Any]] = []
        cursor += 2
        if outcome == "continue":
            if cursor + 1 >= len(events):
                raise ValueError(f"ratchet drive {cycle} is missing its reset")
            reset_started, reset_completed = events[cursor], events[cursor + 1]
            if (
                reset_started["event"] != "ratchet_reset_started"
                or reset_completed["event"] != "ratchet_reset_complete"
            ):
                raise ValueError(f"ratchet drive {cycle} has an invalid reset sequence")
            if (
                int(reset_started.get("payload", {}).get("cycle", -1)) != cycle
                or int(reset_completed.get("payload", {}).get("cycle", -1)) != cycle
            ):
                raise ValueError("ratchet reset cycle does not match drive cycle")
            reset = _window(
                samples,
                int(reset_started["monotonic_time_ns"]),
                int(reset_completed["monotonic_time_ns"]),
            )
            if len(reset) < 2:
                raise ValueError(f"ratchet reset {cycle} requires at least two samples")
            cursor += 2
        cycles.append({"cycle": cycle, "outcome": outcome, "drive": drive, "reset": reset})
        if outcome == "engaged":
            break
    if not cycles or cycles[-1]["outcome"] != "engaged":
        raise ValueError("the final ratchet drive must be marked engaged")
    if cursor != len(events) - 2:
        raise ValueError("unexpected events after final engaged ratchet drive")

    def signed_kz_rotation(rows: Sequence[dict[str, Any]]) -> float:
        return _accumulated_signed_kz_rotation(rows)

    def axial_progress(rows: Sequence[dict[str, Any]]) -> float:
        T0 = np.asarray(rows[0]["T_base_ee"], dtype=np.float64)
        T1 = np.asarray(rows[-1]["T_base_ee"], dtype=np.float64)
        return float(np.dot(T1[:3, 3] - T0[:3, 3], T0[:3, 2]))

    drive_summaries = []
    for cycle in cycles:
        drive = cycle["drive"]
        reset = cycle["reset"]
        drive_summaries.append(
            {
                "cycle": cycle["cycle"],
                "outcome": cycle["outcome"],
                "drive_rotation_about_positive_Kz_rad": signed_kz_rotation(drive),
                "drive_axial_progress_along_positive_Kz_m": axial_progress(drive),
                "drive_q7_delta_rad": float(drive[-1]["q"][6] - drive[0]["q"][6]),
                "reset_rotation_about_positive_Kz_rad": (
                    signed_kz_rotation(reset) if reset else None
                ),
                "reset_q7_delta_rad": (
                    float(reset[-1]["q"][6] - reset[0]["q"][6]) if reset else None
                ),
            }
        )
    signed_rotation_kz = float(
        sum(item["drive_rotation_about_positive_Kz_rad"] for item in drive_summaries)
    )
    total_axial_progress = float(
        sum(item["drive_axial_progress_along_positive_Kz_m"] for item in drive_summaries)
    )
    prewind_rotation_kz = signed_kz_rotation(prewind)
    prewind_q7_delta = float(prewind[-1]["q"][6] - prewind[0]["q"][6])

    def wrench_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        values = np.asarray([row["K_F_ext_hat_K"] for row in rows], dtype=np.float64)
        dq = np.asarray([row["dq"] for row in rows], dtype=np.float64)
        return {
            "sample_count": len(rows),
            "K_wrench_mean": np.mean(values, axis=0).tolist(),
            "K_wrench_std": np.std(values, axis=0).tolist(),
            "K_wrench_min": np.min(values, axis=0).tolist(),
            "K_wrench_max": np.max(values, axis=0).tolist(),
            "maximum_abs_dq_rad_s": float(np.max(np.abs(dq))),
        }

    baseline_stats = wrench_stats(baseline)
    drive_samples = [sample for cycle in cycles for sample in cycle["drive"]]
    motion_stats = wrench_stats(drive_samples)
    engaged_stats = wrench_stats(engaged)
    baseline_mean = np.asarray(baseline_stats["K_wrench_mean"])
    engaged_mean = np.asarray(engaged_stats["K_wrench_mean"])
    return {
        "schema_version": 1,
        "rotation_measurement_method": "sum_incremental_local_Kz_yaw",
        "wrist_prewind_rotation_about_positive_Kz_rad": prewind_rotation_kz,
        "wrist_prewind_rotation_about_positive_Kz_deg": float(
            np.degrees(prewind_rotation_kz)
        ),
        "wrist_prewind_q7_delta_rad": prewind_q7_delta,
        "signed_rotation_about_positive_Kz_rad": signed_rotation_kz,
        "signed_rotation_about_positive_Kz_deg": float(np.degrees(signed_rotation_kz)),
        "tightening_rotation_sign_Kz": (
            "positive" if signed_rotation_kz > 0 else "negative" if signed_rotation_kz < 0 else "zero"
        ),
        "axial_progress_along_positive_Kz_m": total_axial_progress,
        "ratchet_cycle_count": len(cycles),
        "ratchet_cycles": drive_summaries,
        "phases": {
            "unloaded_baseline": baseline_stats,
            "manual_engagement_motion": motion_stats,
            "engaged_stationary": engaged_stats,
        },
        "engaged_minus_baseline_K_wrench": (engaged_mean - baseline_mean).tolist(),
        "engaged_axial_force_delta_N": float(engaged_mean[2] - baseline_mean[2]),
        "engaged_axial_torque_delta_Nm": float(engaged_mean[5] - baseline_mean[5]),
        "thresholds_derived": False,
        "manual_motion_wrench_is_human_confounded": True,
    }
