#!/usr/bin/env python3
"""Prepare and collect bracket placement demos over the bracket workspace grid.

Each cell gets its own translated path, live read-only IK solve, offline path
audit, unrecorded wall-to-cell setup, and camera-recorded cell-to-wall demo.
The workbook and JSON ledger are bracket-specific; cube-pick trackers are not
read or modified.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model-running"))
sys.path.insert(0, str(ROOT / "data-pipeline"))

from create_grid_tracker import inline_cell, row_xml, styles_xml, workbook_xml, write_xlsx  # noqa: E402
from prepare_bracket_new_cell_path import build_target_rows  # noqa: E402
from prepare_bracket_wall_cell_cycle import prepare as prepare_cycle  # noqa: E402


DEFAULT_PREVIEW = ROOT / "calibration" / "bracket_workspace_corners_v2" / "candidate_cells_preview.json"
DEFAULT_WALL_KEEP_OUT = ROOT / "calibration" / "bracket_wall_keepout_v1" / "probed_points.json"
DEFAULT_SOURCE_CONFIG_DIR = ROOT / "configs"
DEFAULT_WALL_CONFIG = ROOT / "configs" / "bracket_manual_replay_1x_highcap_b93.json"
DEFAULT_TEMPLATE_CONFIG = ROOT / "configs" / "bracket_new_cell_b97_trial_v1.json"
DEFAULT_OUTPUT = ROOT / "diagnostics" / "bracket_workspace_cell_loop_v1"
DEFAULT_TRACKER_XLSX = ROOT / "configs" / "bracket_workspace_cell_tracker.xlsx"
DEFAULT_TRACKER_JSON = DEFAULT_OUTPUT / "tracker_state.json"
IK_BINARY = ROOT / "model-running" / "streaming" / "build" / "fr3_bracket_outer_ik_preview"
ROBOT_IP = "172.16.0.2"


def proposal_rows(preview_path: Path) -> list[dict]:
    data = json.loads(preview_path.read_text())
    rows = data.get("candidates", [])
    proposals = [str(row.get("proposal", "")) for row in rows]
    if not rows or len(proposals) != len(set(proposals)):
        raise ValueError("proposal preview is empty or has duplicate IDs")
    for row in rows:
        xyz = np.asarray(row.get("ee_xyz_m"), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"{row.get('proposal')}: invalid EE target")
    return rows


def load_ledger(path: Path, rows: list[dict], xlsx_path: Path) -> dict:
    if path.is_file():
        ledger = json.loads(path.read_text())
        if set(ledger.get("cells", {})) != {row["proposal"] for row in rows}:
            raise ValueError("existing bracket tracker does not match this proposal grid")
        return ledger
    cells = {}
    for row in rows:
        proposal = row["proposal"]
        cells[proposal] = {
            "setup_status": "UNTESTED",
            "dataset_status": "UNTESTED" if row.get("inside_recorded_grasp_hull") else "SKIP",
            "episode_index": None,
            "episode_path": None,
            "notes": "" if row.get("inside_recorded_grasp_hull") else "Outside accepted grasp hull",
        }
    # B97's P01_02 round trip has already been physically completed. The setup
    # is reusable, while its dataset status remains open for future capture.
    if "P01_02" in cells:
        cells["P01_02"]["setup_status"] = "PASS"
        cells["P01_02"]["notes"] = "B97 wall-cell-wall setup passed; dataset not captured"
    ledger = {"schema_version": 1, "proposal_preview": str(DEFAULT_PREVIEW), "cells": cells}
    save_ledger(path, ledger, rows, xlsx_path)
    return ledger


def write_tracker_xlsx(path: Path, rows: list[dict], ledger: dict) -> None:
    headers = [
        "cell_id", "proposal", "setup_status", "dataset_status", "x_m", "y_m",
        "table_z_m", "nearest_accepted_cell", "nearest_distance_m",
        "inside_grasp_hull", "episode_index", "episode_path", "notes",
    ]
    rendered = [row_xml(1, headers, style=1)]
    for cell_id, row in enumerate(rows, start=1):
        status = ledger["cells"][row["proposal"]]
        xyz = row["ee_xyz_m"]
        values = [
            cell_id, row["proposal"], status["setup_status"], status["dataset_status"],
            *[round(float(v), 5) for v in xyz], row.get("nearest_accepted_cell"),
            row.get("nearest_accepted_xy_distance_m"),
            bool(row.get("inside_recorded_grasp_hull")), status.get("episode_index"),
            status.get("episode_path"), status.get("notes", ""),
        ]
        rendered.append(row_xml(cell_id + 1, values))
    last = len(rows) + 1
    columns = ''.join(
        f'<col min="{i}" max="{i}" width="{width}" customWidth="1"/>'
        for i, width in enumerate((10, 14, 16, 16, 13, 13, 13, 22, 20, 18, 16, 50, 52), start=1)
    )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{columns}</cols><sheetData>{"".join(rendered)}</sheetData>'
        f'<autoFilter ref="A1:M{last}"/></worksheet>'
    )
    write_xlsx(path, sheet)


def save_ledger(path: Path, ledger: dict, rows: list[dict], xlsx_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    temporary.replace(path)
    write_tracker_xlsx(xlsx_path, rows, ledger)


def source_config_for(row: dict) -> Path:
    source_cell = str(row["nearest_accepted_cell"]).upper()
    if not re.fullmatch(r"B\d+", source_cell):
        raise ValueError(f"unsupported nearest source cell: {source_cell}")
    path = DEFAULT_SOURCE_CONFIG_DIR / f"bracket_manual_replay_1x_highcap_{source_cell.lower()}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no accepted source config for {source_cell}: {path}")
    return path


def prepare_cell(row: dict, output_root: Path, template_path: Path, python: str) -> tuple[Path, Path]:
    proposal = row["proposal"]
    cell_root = output_root / proposal
    cell_root.mkdir(parents=True, exist_ok=True)
    source_config = source_config_for(row)
    template = json.loads(template_path.read_text())
    spec_path = cell_root / "cell_spec.json"
    spec = {
        "schema_version": 1,
        "cell_label": proposal,
        "target_probed_ee_origin_xyz_m": row["ee_xyz_m"],
        "source_bracket_slot": str(row["nearest_accepted_cell"]).upper(),
        "source_replay_config": str(source_config),
        "fixed_wall_bracket_slot": "B93",
        "fixed_wall_replay_config": str(DEFAULT_WALL_CONFIG),
        "dataset_episode": False,
        "motion_approved": False,
        "replay_enabled": False,
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    generated = subprocess.run(
        [python, str(ROOT / "model-running" / "prepare_bracket_new_cell_path.py"),
         "--spec", str(spec_path), "--wall-probes", str(DEFAULT_WALL_KEEP_OUT),
         "--output-dir", str(cell_root)],
        cwd=ROOT, check=False,
    )
    if generated.returncode:
        raise RuntimeError(f"{proposal}: target path generation failed")

    targets = cell_root / "translated_bracket_ik_targets.csv"
    candidate = cell_root / "translated_bracket_ik_candidate.csv"
    if not IK_BINARY.is_file():
        raise FileNotFoundError(f"read-only IK solver is missing: {IK_BINARY}")
    solved = subprocess.run(
        [str(IK_BINARY), ROBOT_IP, str(targets), str(candidate)],
        input="\n", text=True, cwd=ROOT, check=False,
    )
    if solved.returncode:
        raise RuntimeError(f"{proposal}: read-only continuous IK solve failed")

    config = json.loads(template_path.read_text())
    config["bracket_slot"] = proposal
    config["candidate"] = str(candidate)
    config["output_dir"] = str(cell_root)
    config["calibration_only"] = False
    config["note"] = f"Cell {proposal}; source {spec['source_bracket_slot']}; fixed wall pose B93."
    config_path = cell_root / "cell_replay_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    manifest_dir = cell_root / "cycle_prepare"
    manifest = prepare_cycle(config_path, manifest_dir)
    if manifest["blockers"]:
        raise RuntimeError(f"{proposal}: wall-cell inverse offline blockers: {manifest['blockers']}")
    offline = subprocess.run(
        [python, str(ROOT / "model-running" / "commission_manual_bracket_arm.py"),
         "--config", str(config_path), "--offline-validation"],
        cwd=ROOT, check=False,
    )
    if offline.returncode:
        raise RuntimeError(f"{proposal}: forward placement offline validation failed")
    return config_path, cell_root


def new_episode_paths(recording_root: Path, before: set[Path]) -> list[Path]:
    return sorted((path for path in recording_root.iterdir() if path.is_dir() and path not in before),
                  key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true", help="Create/update the bracket Excel tracker; no robot connection")
    parser.add_argument("--execute-supervised-trials", action="store_true", help="Run per-cell setup and placement recording after read-only/offline checks")
    parser.add_argument("--start-proposal", default="P01_01")
    parser.add_argument(
        "--retry-proposal", action="append", default=[], metavar="Pxx_yy",
        help="Explicitly retry a previously BLOCKED/FAIL proposal (repeatable); PASS/SKIP rows cannot be retried",
    )
    parser.add_argument(
        "--record-placement-only", action="store_true",
        help=(
            "resume only the cell-to-wall recording for one explicit retry proposal; "
            "use when wall-to-cell setup already succeeded and the bracket remains at the cell"
        ),
    )
    parser.add_argument("--count", type=int, default=1, help="Number of eligible proposal rows to process")
    parser.add_argument("--start-episode", type=int, help="Required for physical demo collection")
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tracker-json", type=Path, default=DEFAULT_TRACKER_JSON)
    parser.add_argument("--tracker-xlsx", type=Path, default=DEFAULT_TRACKER_XLSX)
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.execute_supervised_trials and (args.start_episode is None or args.start_episode < 1):
        raise SystemExit("--execute-supervised-trials requires a positive --start-episode")
    if args.preview_only and args.execute_supervised_trials:
        raise SystemExit("choose --preview-only or --execute-supervised-trials")
    if args.record_placement_only and not args.execute_supervised_trials:
        raise SystemExit("--record-placement-only requires --execute-supervised-trials")
    if args.record_placement_only and len(args.retry_proposal) != 1:
        raise SystemExit("--record-placement-only requires exactly one --retry-proposal")
    if args.record_placement_only and args.count != 1:
        raise SystemExit("--record-placement-only requires --count 1")

    preview_path = args.preview.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tracker_json = args.tracker_json.expanduser().resolve()
    tracker_xlsx = args.tracker_xlsx.expanduser().resolve()
    rows = proposal_rows(preview_path)
    ledger = load_ledger(tracker_json, rows, tracker_xlsx)
    write_tracker_xlsx(tracker_xlsx, rows, ledger)
    print("=== BRACKET WORKSPACE CELL DATASET LOOP ===")
    print(f"grid proposals:       {len(rows)}")
    print(f"tracker workbook:     {tracker_xlsx}")
    print(f"bracket-only output:  {output_root}")
    if not args.execute_supervised_trials:
        print("Tracker ready. No robot, Hand, or camera connection was made.")
        return

    proposal_names = [row["proposal"] for row in rows]
    if args.start_proposal not in proposal_names:
        raise SystemExit(f"unknown proposal {args.start_proposal}; choose one of {proposal_names}")
    start_index = proposal_names.index(args.start_proposal)
    if args.retry_proposal:
        unknown = sorted(set(args.retry_proposal) - set(proposal_names))
        if unknown:
            raise SystemExit(f"unknown retry proposal(s): {', '.join(unknown)}")
        chosen = []
        for proposal in args.retry_proposal:
            row = next(item for item in rows if item["proposal"] == proposal)
            status = ledger["cells"][proposal]["dataset_status"]
            if status in {"PASS", "SKIP"}:
                raise SystemExit(f"cannot retry {proposal}: dataset status is {status}")
            if not bool(row.get("inside_recorded_grasp_hull")):
                raise SystemExit(f"cannot retry {proposal}: outside the recorded grasp hull")
            chosen.append(row)
        if len(chosen) > args.count:
            raise SystemExit("--count must be at least the number of --retry-proposal values")
    else:
        chosen = [
            row for row in rows[start_index:]
            if ledger["cells"][row["proposal"]]["dataset_status"] == "UNTESTED"
            and bool(row.get("inside_recorded_grasp_hull"))
        ][:args.count]
    if not chosen:
        raise SystemExit("no untested in-hull bracket cells remain from --start-proposal")
    print("cells to run:", ", ".join(row["proposal"] for row in chosen))
    if args.record_placement_only:
        print("Resume mode: bracket must already be seated at the selected cell; only cell-to-wall recording will run.")
    else:
        print("Each cell starts at the fixed wall pose, places the bracket on the cell, then records cell-to-wall with cameras.")
    print("Keep the bracket, pins, cell surface, and complete robot path clear; remain at the emergency stop.")

    python = sys.executable
    recording_root = ROOT / "recordings" / "full_bracket_bolt_autonomous_raw_v1"
    for index, row in enumerate(chosen):
        proposal = row["proposal"]
        episode_index = args.start_episode + index
        cell_status = ledger["cells"][proposal]
        print(f"\n[CELL-LOOP] {proposal}, placement episode {episode_index}", flush=True)
        if args.record_placement_only:
            cell_root = output_root / proposal
            config_path = cell_root / "cell_replay_config.json"
            candidate_path = cell_root / "translated_bracket_ik_candidate.csv"
            if not config_path.is_file() or not candidate_path.is_file():
                raise SystemExit(
                    f"[BLOCKED] {proposal} has no prepared cell path; normal setup is required first"
                )
            print(
                f"[CELL-LOOP] using existing commissioned candidate {candidate_path}; "
                "not repeating wall-to-cell transfer",
                flush=True,
            )
        else:
            try:
                config_path, cell_root = prepare_cell(row, output_root, args.template_config.resolve(), python)
            except Exception as error:
                cell_status["setup_status"] = "BLOCKED"
                cell_status["notes"] = str(error)
                save_ledger(tracker_json, ledger, rows, tracker_xlsx)
                print(f"[BLOCKED] {error}; no motion was started for this cell")
                raise SystemExit(
                    f"[STOPPED] {proposal} target generation blocked; review the wall-clearance report before selecting another cell"
                )

        before = {path for path in recording_root.iterdir() if path.is_dir()} if recording_root.exists() else set()
        if args.record_placement_only:
            command = [
                python, str(ROOT / "model-running" / "record_autonomous_bolt_engagement_demo.py"),
                "--record-demo", "--full-assembly", "--bracket-only", "--four-x",
                "--bracket-source", "manual_2026_08_16", "--bracket-arm-config", str(config_path),
                "--episode-index", str(episode_index),
            ]
        else:
            command = [
                python, str(ROOT / "model-running" / "run_bracket_wall_to_cell.py"),
                "--config", str(config_path), "--episode-index", str(episode_index),
                "--execute-supervised-trial",
            ]
        cell_status["setup_status"] = "RUNNING"
        cell_status["episode_index"] = episode_index
        save_ledger(tracker_json, ledger, rows, tracker_xlsx)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        episodes = new_episode_paths(recording_root, before) if recording_root.exists() else []
        accepted_episode = None
        for episode in reversed(episodes):
            try:
                metadata = json.loads((episode / "metadata.json").read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("episode_index") == episode_index:
                accepted_episode = (episode, metadata)
                break
        if completed.returncode == 0 and accepted_episode is not None:
            episode, metadata = accepted_episode
            if metadata.get("success") is True and metadata.get("accepted_by_operator") is True:
                cell_status.update({
                    "setup_status": "PASS", "dataset_status": "PASS",
                    "episode_path": str(episode),
                    "notes": "accepted bracket-only placement episode",
                })
                save_ledger(tracker_json, ledger, rows, tracker_xlsx)
                print(f"[PASS] {proposal} recorded: {episode}")
                continue
        cell_status["setup_status"] = "FAIL"
        cell_status["dataset_status"] = "FAIL"
        cell_status["notes"] = f"supervised cell cycle exited {completed.returncode}; inspect {cell_root} and episode output"
        save_ledger(tracker_json, ledger, rows, tracker_xlsx)
        raise SystemExit(f"[STOPPED] {proposal} did not finish with an operator-accepted dataset episode")

    print(f"\n[CELL-LOOP] tracker updated: {tracker_xlsx}")
    print("Review FAIL/BLOCKED rows before resuming; the loop stops after a physical-cycle failure.")


if __name__ == "__main__":
    main()
