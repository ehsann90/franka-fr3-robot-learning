#!/usr/bin/env python3
"""Validate and plot one or more Pilot-guided bolt engagement captures."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


FR3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FR3_ROOT))

from expert_demo.thread_engagement import summarize_thread_engagement  # noqa: E402


DEFAULT_GEOMETRY = FR3_ROOT / "calibration" / "bolt_thread_geometry_v1.json"
COLORS = ("#1f77b4", "#d62728", "#2ca02c")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def line_points(
    elapsed: np.ndarray,
    values: np.ndarray,
    left: float,
    top: float,
    width: float,
    height: float,
    lo: float,
    hi: float,
) -> str:
    x = left + (elapsed - elapsed[0]) * width / max(elapsed[-1] - elapsed[0], 1e-9)
    y = top + height - (values - lo) * height / max(hi - lo, 1e-12)
    return " ".join(f"{xv:.2f},{yv:.2f}" for xv, yv in zip(x, y))


def plot_capture(root: Path, lead_m: float) -> dict[str, Any]:
    metadata = json.loads((root / "metadata.json").read_text())
    events = read_jsonl(root / "events.jsonl")
    samples = read_jsonl(root / "trajectory.jsonl")
    summary = summarize_thread_engagement(samples, events)
    arrays = np.load(root / "trajectory_arrays.npz")
    times_ns = np.asarray(arrays["monotonic_time_ns"], dtype=np.int64)
    q = np.asarray(arrays["q"], dtype=np.float64)
    T = np.asarray(arrays["T_base_ee"], dtype=np.float64)
    wrench = np.asarray(arrays["K_F_ext_hat_K"], dtype=np.float64)
    if (
        len(times_ns) < 2
        or q.shape != (len(times_ns), 7)
        or T.shape != (len(times_ns), 4, 4)
        or wrench.shape != (len(times_ns), 6)
    ):
        raise ValueError("trajectory arrays have inconsistent dimensions")
    if not all(np.isfinite(values).all() for values in (q, T, wrench)):
        raise ValueError("trajectory contains NaN or Inf")

    elapsed = (times_ns - times_ns[0]) / 1e9
    dt_ms = np.diff(times_ns) / 1e6
    drive_windows = [
        (
            int(start["monotonic_time_ns"]),
            int(stop["monotonic_time_ns"]),
        )
        for start, stop in zip(
            [event for event in events if event["event"] == "ratchet_drive_started"],
            [event for event in events if event["event"] == "ratchet_drive_complete"],
        )
    ]
    actual_axial = np.zeros(len(times_ns), dtype=np.float64)
    thread_expected = np.zeros(len(times_ns), dtype=np.float64)
    for index in range(1, len(times_ns)):
        previous, current = T[index - 1], T[index]
        actual_axial[index] = actual_axial[index - 1] + float(
            np.dot(current[:3, 3] - previous[:3, 3], previous[:3, 2])
        )
        thread_expected[index] = thread_expected[index - 1]
        midpoint = (int(times_ns[index - 1]) + int(times_ns[index])) // 2
        if any(start <= midpoint <= stop for start, stop in drive_windows):
            relative_R = previous[:3, :3].T @ current[:3, :3]
            delta_angle = float(np.arctan2(relative_R[1, 0], relative_R[0, 0]))
            thread_expected[index] += delta_angle * lead_m / (2.0 * np.pi)

    if len(times_ns) > 3000:
        keep = np.linspace(0, len(times_ns) - 1, 3000, dtype=int)
        elapsed_plot = elapsed[keep]
        q7_plot = q[keep, 6]
        actual_plot = actual_axial[keep] * 1000.0
        expected_plot = thread_expected[keep] * 1000.0
        fz_plot = wrench[keep, 2]
        tz_plot = wrench[keep, 5]
    else:
        elapsed_plot, q7_plot = elapsed, q[:, 6]
        actual_plot, expected_plot = actual_axial * 1000.0, thread_expected * 1000.0
        fz_plot, tz_plot = wrench[:, 2], wrench[:, 5]

    panels = (
        ("Joint 7 position (rad)", (q7_plot,), ("q7",)),
        (
            "Cumulative axial motion (mm)",
            (actual_plot, expected_plot),
            ("measured EE motion", "CAD thread-only expectation"),
        ),
        ("Tool-axis force K_Fz (N)", (fz_plot,), ("Fz",)),
        ("Tool-axis torque K_Tz (Nm)", (tz_plot,), ("Tz",)),
    )
    width, height = 1280, 960
    left, panel_width, panel_height = 105, 1110, 155
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace;fill:#222}.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.event{stroke:#999;stroke-width:.8;stroke-dasharray:3 3}</style>',
        '<text x="35" y="30" font-size="20">Bolt Pilot thread-engagement trace</text>',
        f'<text x="35" y="52" font-size="13">{html.escape(root.name)}</text>',
        f'<text x="35" y="72" font-size="13">CAD lead: {lead_m * 1000:.4f} mm/rev; Pilot wrench/motion is human-confounded and is not a controller threshold.</text>',
    ]
    for panel_index, (title, series, labels) in enumerate(panels):
        top = 105 + panel_index * 205
        combined = np.concatenate(series)
        lo, hi = float(np.min(combined)), float(np.max(combined))
        padding = max((hi - lo) * 0.08, 1e-6)
        lo, hi = lo - padding, hi + padding
        svg.append(f'<text x="{left}" y="{top - 10}" font-size="16">{html.escape(title)}</text>')
        svg.append(f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="none" class="axis"/>')
        for fraction in (0.25, 0.5, 0.75):
            y = top + panel_height * fraction
            svg.append(f'<line x1="{left}" y1="{y}" x2="{left + panel_width}" y2="{y}" class="grid"/>')
        for event in events:
            event_elapsed = (int(event["monotonic_time_ns"]) - int(times_ns[0])) / 1e9
            x = left + event_elapsed * panel_width / max(elapsed[-1], 1e-9)
            svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_height}" class="event"/>')
        for index, (values, label) in enumerate(zip(series, labels)):
            points = line_points(
                elapsed_plot, values, left, top, panel_width, panel_height, lo, hi
            )
            color = COLORS[index]
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.35"/>')
            svg.append(
                f'<text x="{left + index * 360}" y="{top + panel_height + 18}" font-size="12" fill="{color}">'
                f'{html.escape(label)} [{float(np.min(values)):.4g}, {float(np.max(values)):.4g}]</text>'
            )
        svg.append(
            f'<text x="{left + panel_width - 170}" y="{top + panel_height + 18}" font-size="12">'
            f'0 .. {elapsed[-1]:.2f} s</text>'
        )
    svg.append("</svg>")
    output = root / "thread_engagement_trace.svg"
    output.write_text("\n".join(svg) + "\n")

    rotation_deg = float(summary["signed_rotation_about_positive_Kz_deg"])
    expected_mm = rotation_deg / 360.0 * lead_m * 1000.0
    observed_mm = float(summary["axial_progress_along_positive_Kz_m"]) * 1000.0
    return {
        "capture": root.name,
        "accepted": bool(metadata.get("accepted_by_operator")),
        "samples": len(times_ns),
        "sample_errors": int(metadata.get("sample_error_count", -1)),
        "median_rate_hz": float(1000.0 / np.median(dt_ms)),
        "maximum_gap_ms": float(np.max(dt_ms)),
        "ratchet_cycles": int(summary["ratchet_cycle_count"]),
        "tightening_rotation_deg": rotation_deg,
        "expected_thread_descent_mm": expected_mm,
        "observed_pilot_ee_descent_mm": observed_mm,
        "pilot_excess_axial_motion_mm": observed_mm - expected_mm,
        "engaged_delta_Fz_N": float(summary["engaged_axial_force_delta_N"]),
        "engaged_delta_Tz_Nm": float(summary["engaged_axial_torque_delta_Nm"]),
        "plot": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dirs", nargs="+", type=Path)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    args = parser.parse_args()
    geometry = json.loads(args.geometry.expanduser().resolve().read_text())
    lead_m = float(geometry["printed_thread_lead_mm_per_revolution"]) / 1000.0
    reports = []
    for capture_dir in args.capture_dirs:
        root = capture_dir.expanduser().resolve()
        try:
            reports.append(plot_capture(root, lead_m))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot validate {root}: {error}") from error
    print("capture  cycles  rotation_deg  expected_mm  observed_mm  excess_mm  dTz_Nm  gap_ms")
    for report in reports:
        print(
            f"{report['capture']}  {report['ratchet_cycles']:>2d}  "
            f"{report['tightening_rotation_deg']:+10.3f}  "
            f"{report['expected_thread_descent_mm']:+10.3f}  "
            f"{report['observed_pilot_ee_descent_mm']:+10.3f}  "
            f"{report['pilot_excess_axial_motion_mm']:+9.3f}  "
            f"{report['engaged_delta_Tz_Nm']:+7.4f}  "
            f"{report['maximum_gap_ms']:.3f}"
        )
        print(f"  plot: {report['plot']}")
    print("No automatic force/torque threshold was derived from Pilot-confounded data.")


if __name__ == "__main__":
    main()
