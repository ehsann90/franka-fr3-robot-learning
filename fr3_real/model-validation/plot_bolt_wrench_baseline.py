#!/usr/bin/env python3
"""Create a dependency-free SVG plot from a bolt PREINSERT wrench baseline."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


COLORS = ("#d62728", "#2ca02c", "#1f77b4")


def polyline(values: np.ndarray, left: float, top: float, width: float, height: float) -> str:
    lo = float(np.min(values))
    hi = float(np.max(values))
    padding = max((hi - lo) * 0.08, 1e-9)
    lo -= padding
    hi += padding
    x = left + np.linspace(0.0, width, len(values))
    y = top + height - (values - lo) * height / (hi - lo)
    return " ".join(f"{xv:.2f},{yv:.2f}" for xv, yv in zip(x, y)), lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.capture_dir.expanduser().resolve()
    output = (args.output or (root / "wrench_baseline.svg")).expanduser().resolve()
    try:
        metadata = json.loads((root / "metadata.json").read_text())
        arrays = np.load(root / "trajectory_arrays.npz")
        times = np.asarray(arrays["monotonic_time_ns"], dtype=np.int64)
        k_wrench = np.asarray(arrays["K_F_ext_hat_K"], dtype=np.float64)
        o_wrench = np.asarray(arrays["O_F_ext_hat_K"], dtype=np.float64)
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read wrench baseline {root}: {error}") from error
    if k_wrench.shape != (len(times), 6) or o_wrench.shape != k_wrench.shape:
        raise SystemExit("wrench arrays must have shape (sample_count, 6)")
    if len(times) > 2000:
        indices = np.linspace(0, len(times) - 1, 2000, dtype=int)
        times, k_wrench, o_wrench = times[indices], k_wrench[indices], o_wrench[indices]
    elapsed = (times - times[0]) / 1e9

    width, height = 1200, 900
    panel_w, panel_h = 1030, 155
    left = 110
    panel_specs = (
        ("K frame force (N)", k_wrench[:, :3], ("Fx", "Fy", "Fz")),
        ("K frame torque (Nm)", k_wrench[:, 3:], ("Tx", "Ty", "Tz")),
        ("Base O frame force (N)", o_wrench[:, :3], ("Fx", "Fy", "Fz")),
        ("Base O frame torque (Nm)", o_wrench[:, 3:], ("Tx", "Ty", "Tz")),
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace;fill:#222}.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
        '<text x="40" y="30" font-size="20">Bolt PREINSERT stationary external-wrench baseline</text>',
        f'<text x="40" y="54" font-size="13">{html.escape(metadata.get("trajectory_name", root.name))}</text>',
        '<text x="40" y="75" font-size="13" fill="#a00">K-frame Z is a candidate screw axis only; not experimentally verified. No thresholds shown.</text>',
    ]
    for panel_index, (title, values, labels) in enumerate(panel_specs):
        top = 105 + panel_index * 190
        svg.append(f'<text x="{left}" y="{top - 10}" font-size="16">{title}</text>')
        svg.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" class="axis"/>')
        for fraction in (0.25, 0.5, 0.75):
            gy = top + panel_h * fraction
            svg.append(f'<line x1="{left}" y1="{gy}" x2="{left + panel_w}" y2="{gy}" class="grid"/>')
        for component in range(3):
            points, lo, hi = polyline(values[:, component], left, top, panel_w, panel_h)
            svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[component]}" stroke-width="1.4"/>')
            legend_x = left + component * 245
            svg.append(
                f'<text x="{legend_x}" y="{top + panel_h + 17}" font-size="12" fill="{COLORS[component]}">'
                f'{labels[component]} [{lo:.4g}, {hi:.4g}]</text>'
            )
        svg.append(f'<text x="{left + panel_w - 150}" y="{top + panel_h + 17}" font-size="12">0 .. {elapsed[-1]:.2f} s</text>')
    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n")
    print(f"[plot] wrote {output}")
    print("[plot] K-frame Z remains an unverified screw-axis candidate; no thresholds derived")


if __name__ == "__main__":
    main()
