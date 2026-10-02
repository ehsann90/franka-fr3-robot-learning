"""Pure phase statistics for an operator-applied, read-only axial load capture."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


PHASE_EVENTS = {
    "unloaded": ("unloaded_baseline_started", "unloaded_baseline_complete"),
    "axial_load": ("axial_load_window_started", "axial_load_window_complete"),
    "recovered": ("unloaded_recovery_started", "unloaded_recovery_complete"),
}


def summarize_axial_load_phases(
    samples: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    event_times = {event["event"]: int(event["monotonic_time_ns"]) for event in events}
    times = np.asarray([sample["monotonic_time_ns"] for sample in samples], dtype=np.int64)
    wrench = np.asarray([sample["K_F_ext_hat_K"] for sample in samples], dtype=np.float64)
    dq = np.asarray([sample["dq"] for sample in samples], dtype=np.float64)
    if wrench.shape != (len(samples), 6) or dq.shape != (len(samples), 7):
        raise ValueError("wrench/dq arrays have invalid dimensions")
    if not np.isfinite(wrench).all() or not np.isfinite(dq).all():
        raise ValueError("wrench/dq arrays contain NaN or Inf")

    phases: dict[str, Any] = {}
    for phase, (start_event, stop_event) in PHASE_EVENTS.items():
        if start_event not in event_times or stop_event not in event_times:
            raise ValueError(f"missing phase events for {phase}")
        mask = (times >= event_times[start_event]) & (times <= event_times[stop_event])
        values = wrench[mask]
        velocities = dq[mask]
        if len(values) < 2:
            raise ValueError(f"phase {phase} has fewer than two samples")
        phases[phase] = {
            "sample_count": int(len(values)),
            "K_wrench_mean": np.mean(values, axis=0).tolist(),
            "K_wrench_std": np.std(values, axis=0).tolist(),
            "K_wrench_min": np.min(values, axis=0).tolist(),
            "K_wrench_max": np.max(values, axis=0).tolist(),
            "maximum_abs_dq_rad_s": float(np.max(np.abs(velocities))),
        }

    unloaded = np.asarray(phases["unloaded"]["K_wrench_mean"])
    loaded = np.asarray(phases["axial_load"]["K_wrench_mean"])
    recovered = np.asarray(phases["recovered"]["K_wrench_mean"])
    delta = loaded - unloaded
    return {
        "schema_version": 1,
        "phases": phases,
        "loaded_minus_unloaded_K_wrench": delta.tolist(),
        "dominant_force_component_index": int(np.argmax(np.abs(delta[:3]))),
        "dominant_torque_component_index": int(np.argmax(np.abs(delta[3:])) + 3),
        "axial_force_component": "K_F_ext_hat_K[2]",
        "axial_torque_component": "K_F_ext_hat_K[5]",
        "axial_force_loaded_minus_unloaded_N": float(delta[2]),
        "axial_torque_loaded_minus_unloaded_Nm": float(delta[5]),
        "recovered_minus_unloaded_K_wrench": (recovered - unloaded).tolist(),
        "thresholds_derived": False,
    }
