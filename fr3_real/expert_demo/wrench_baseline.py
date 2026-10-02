"""Validation and statistics for read-only FR3 external-wrench baselines."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


WRENCH_SIGNALS = ("O_F_ext_hat_K", "K_F_ext_hat_K")
WRENCH_COMPONENTS = ("Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm")


def _component_statistics(values: np.ndarray) -> list[dict[str, float | str]]:
    return [
        {
            "component": name,
            "mean": float(np.mean(values[:, index])),
            "std": float(np.std(values[:, index])),
            "minimum": float(np.min(values[:, index])),
            "maximum": float(np.max(values[:, index])),
            "peak_to_peak": float(np.ptp(values[:, index])),
            "p05": float(np.percentile(values[:, index], 5.0)),
            "median": float(np.median(values[:, index])),
            "p95": float(np.percentile(values[:, index], 95.0)),
        }
        for index, name in enumerate(WRENCH_COMPONENTS)
    ]


def summarize_wrench_baseline(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return descriptive statistics without deriving any safety threshold."""

    if len(samples) < 2:
        raise ValueError("at least two wrench samples are required")
    times = np.asarray([sample["monotonic_time_ns"] for sample in samples], dtype=np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("sample timestamps must be strictly increasing")
    dq = np.asarray([sample["dq"] for sample in samples], dtype=np.float64)
    if dq.shape != (len(samples), 7) or not np.isfinite(dq).all():
        raise ValueError("dq must have shape (sample_count, 7) with finite values")

    wrench_statistics: dict[str, Any] = {}
    for signal in WRENCH_SIGNALS:
        if not all(signal in sample for sample in samples):
            raise ValueError(f"{signal} is missing from one or more samples")
        values = np.asarray([sample[signal] for sample in samples], dtype=np.float64)
        if values.shape != (len(samples), 6) or not np.isfinite(values).all():
            raise ValueError(f"{signal} must have shape (sample_count, 6) with finite values")
        wrench_statistics[signal] = {
            "components": _component_statistics(values),
        }

    duration_s = float((times[-1] - times[0]) / 1e9)
    return {
        "schema_version": 1,
        "sample_count": len(samples),
        "sample_duration_s": duration_s,
        "achieved_sample_hz": float((len(samples) - 1) / duration_s),
        "maximum_timestamp_gap_s": float(np.max(np.diff(times)) / 1e9),
        "maximum_measured_abs_dq_rad_s": float(np.max(np.abs(dq))),
        "wrench_statistics": wrench_statistics,
        "thresholds_derived": False,
        "axis_convention_experimentally_verified": False,
        "k_frame_z_status": "candidate screw axis only; experimental verification required",
    }
