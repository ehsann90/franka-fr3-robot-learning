#!/usr/bin/env python3
"""Create a dependency-free SVG and robust statistics for wrist dynamic baseline."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


COLORS = ("#1f77b4", "#d62728")


def points(t, values, left, top, width, height, lo, hi):
    x = left + (t - t[0]) * width / max(t[-1] - t[0], 1e-9)
    y = top + height - (values - lo) * height / max(hi - lo, 1e-12)
    return " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))


def stats(values: np.ndarray) -> dict:
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    root = args.capture_dir.expanduser().resolve()
    try:
        arrays = np.load(root / "dynamic_baseline.npz")
        trace = np.asarray(arrays["trace"], dtype=np.float64)
        columns = arrays["columns"].tolist()
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"cannot read dynamic baseline {root}: {error}") from error
    if trace.ndim != 2 or trace.shape[1] != len(columns) or not np.isfinite(trace).all():
        raise SystemExit("dynamic trace has invalid dimensions or non-finite values")
    index = {name: columns.index(name) for name in ("robot_time_s", "phase", "q7", "dq7", "K_Fz", "K_Tz", "tau_ext_7")}
    time_s = trace[:, index["robot_time_s"]] - trace[0, index["robot_time_s"]]
    phase = trace[:, index["phase"]]
    forward = trace[phase == 1]
    reverse = trace[phase == 2]
    report = {
        "schema_version": 1,
        "capture": root.name,
        "sample_count": len(trace),
        "median_callback_rate_hz": float(1.0 / np.median(np.diff(trace[:, index["robot_time_s"]]))),
        "maximum_callback_gap_ms": float(np.max(np.diff(trace[:, index["robot_time_s"]])) * 1000.0),
        "return_q7_error_rad": float(reverse[-1, index["q7"]] - forward[0, index["q7"]]),
        "maximum_tcp_drift_mm": float(np.max(np.linalg.norm(trace[:, -3:] - trace[0, -3:], axis=1)) * 1000.0),
        "phases": {},
        "thresholds_derived": False,
        "note": "Use direction/q7-conditioned residuals; raw K_Tz is not a jam threshold.",
    }
    for rows, name in ((forward, "forward"), (reverse, "reverse")):
        moving = rows[np.abs(rows[:, index["dq7"]]) > 0.05]
        report["phases"][name] = {
            "sample_count": len(rows),
            "moving_sample_count": len(moving),
            "q7_start_rad": float(rows[0, index["q7"]]),
            "q7_stop_rad": float(rows[-1, index["q7"]]),
            "moving_K_Fz_N": stats(moving[:, index["K_Fz"]]),
            "moving_K_Tz_Nm": stats(moving[:, index["K_Tz"]]),
            "moving_tau_ext_joint7_Nm": stats(moving[:, index["tau_ext_7"]]),
        }
    (root / "dynamic_baseline_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    keep = np.linspace(0, len(trace) - 1, min(len(trace), 4000), dtype=int)
    t = time_s[keep]
    panels = (
        ("q7 / dq7", (trace[keep, index["q7"]], trace[keep, index["dq7"]]), ("q7 rad", "dq7 rad/s")),
        ("Tool-axis force", (trace[keep, index["K_Fz"]],), ("K_Fz N",)),
        ("Tool-axis torque", (trace[keep, index["K_Tz"]],), ("K_Tz Nm",)),
        ("External joint-7 torque", (trace[keep, index["tau_ext_7"]],), ("tau_ext_7 Nm",)),
    )
    width, height, left, panel_w, panel_h = 1280, 940, 105, 1110, 150
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace;fill:#222}.axis{stroke:#777}.grid{stroke:#ddd}</style>',
        '<text x="35" y="30" font-size="20">Empty-Hand q7 dynamic wrench baseline</text>',
        f'<text x="35" y="53" font-size="13">{html.escape(root.name)}</text>',
        '<text x="35" y="74" font-size="13">+90 then -90 degrees; no bolt. No thresholds derived.</text>',
    ]
    split_t = float(forward[-1, index["robot_time_s"]] - trace[0, index["robot_time_s"]])
    for panel_number, (title, series, labels) in enumerate(panels):
        top = 105 + panel_number * 200
        merged = np.concatenate(series)
        lo, hi = float(np.min(merged)), float(np.max(merged))
        pad = max((hi - lo) * 0.08, 1e-6)
        lo, hi = lo - pad, hi + pad
        svg.append(f'<text x="{left}" y="{top - 9}" font-size="16">{title}</text>')
        svg.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" fill="none" class="axis"/>')
        split_x = left + split_t * panel_w / max(t[-1], 1e-9)
        svg.append(f'<line x1="{split_x:.2f}" y1="{top}" x2="{split_x:.2f}" y2="{top + panel_h}" stroke="#777" stroke-dasharray="5 4"/>')
        for i, (values, label) in enumerate(zip(series, labels)):
            svg.append(f'<polyline points="{points(t, values, left, top, panel_w, panel_h, lo, hi)}" fill="none" stroke="{COLORS[i]}" stroke-width="1.3"/>')
            svg.append(f'<text x="{left + i * 330}" y="{top + panel_h + 18}" font-size="12" fill="{COLORS[i]}">{label} [{np.min(values):.4g}, {np.max(values):.4g}]</text>')
        svg.append(f'<text x="{left + panel_w - 270}" y="{top + panel_h + 18}" font-size="12">forward | reverse; 0..{t[-1]:.2f}s</text>')
    svg.append("</svg>")
    (root / "dynamic_baseline_trace.svg").write_text("\n".join(svg) + "\n")
    print(f"[analysis] callback rate: {report['median_callback_rate_hz']:.2f} Hz; max gap {report['maximum_callback_gap_ms']:.3f} ms")
    print(f"[analysis] return error:  {report['return_q7_error_rad']:.6f} rad")
    print(f"[analysis] TCP drift:     {report['maximum_tcp_drift_mm']:.3f} mm")
    print(f"[analysis] wrote {root / 'dynamic_baseline_analysis.json'}")
    print(f"[analysis] wrote {root / 'dynamic_baseline_trace.svg'}")
    print("[analysis] no torque, force, engagement, or jam threshold was derived")


if __name__ == "__main__":
    main()
