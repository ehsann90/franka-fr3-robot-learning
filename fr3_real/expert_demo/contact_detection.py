"""Pure relative-force contact detection helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def first_sustained_crossing(
    values: Sequence[float], baseline: float, threshold_delta: float, persistence: int
) -> int | None:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("values must be a finite one-dimensional sequence")
    if threshold_delta <= 0:
        raise ValueError("threshold_delta must be positive")
    if persistence < 1:
        raise ValueError("persistence must be at least one sample")
    above = samples - float(baseline) >= float(threshold_delta)
    run = 0
    for index, crossed in enumerate(above):
        run = run + 1 if crossed else 0
        if run >= persistence:
            return index - persistence + 1
    return None
