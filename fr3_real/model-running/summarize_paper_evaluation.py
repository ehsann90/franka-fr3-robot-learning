#!/usr/bin/env python3
# Usage: from fr3_real/, run: python model-running/summarize_paper_evaluation.py --checkpoint v3_12000 --manifest configs/fr3_spatial_validation_v3.json
"""Summarize a recorded FR3 paper-evaluation ledger into reproducible results files.

This read-only tool filters the JSONL evaluation ledger by manifest and checkpoint,
then writes a JSON summary and a per-position CSV. PASS/FAIL trials are treated as
valid outcomes; INVALID rows are reported separately and never counted as failures.
Run it after each complete trial round and retain its output with the experiment logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import EVAL_LOGS_DIR


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        default=EVAL_LOGS_DIR / "fr3_policy_eval_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_LOGS_DIR / "paper_summaries",
    )
    args = parser.parse_args()

    manifest_path = resolved(args.manifest)
    results_path = resolved(args.results)
    if not results_path.exists():
        raise SystemExit(f"No evaluation ledger exists at {results_path}")

    manifest = json.loads(manifest_path.read_text())
    positions = {position["eval_id"]: position for position in manifest["positions"]}
    expected_trials = {
        eval_id: int(position.get("trials", manifest.get("trials_per_position", 1)))
        for eval_id, position in positions.items()
    }
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    selected = [
        row
        for row in rows
        if row.get("checkpoint") == args.checkpoint
        and resolved(Path(row.get("manifest", ""))) == manifest_path
    ]
    if not selected:
        raise SystemExit("No ledger rows match the requested checkpoint and manifest")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        if row.get("eval_id") in positions:
            grouped[row["eval_id"]].append(row)

    output_rows = []
    all_outcomes = Counter(row["outcome"] for row in selected)
    valid = all_outcomes["PASS"] + all_outcomes["FAIL"]
    for eval_id, position in positions.items():
        rows_for_position = grouped[eval_id]
        counts = Counter(row["outcome"] for row in rows_for_position)
        output_rows.append(
            {
                "eval_id": eval_id,
                "x": position["x"],
                "y": position["y"],
                "table_z": position["table_z"],
                "planned_trials": expected_trials[eval_id],
                "recorded_rows": len(rows_for_position),
                "pass": counts["PASS"],
                "fail": counts["FAIL"],
                "invalid": counts["INVALID"],
                "valid_success_rate": (
                    round(counts["PASS"] / (counts["PASS"] + counts["FAIL"]), 4)
                    if counts["PASS"] + counts["FAIL"]
                    else None
                ),
            }
        )

    summary = {
        "checkpoint": args.checkpoint,
        "manifest": str(manifest_path),
        "planned_positions": len(positions),
        "planned_trials": sum(expected_trials.values()),
        "recorded_rows": len(selected),
        "pass": all_outcomes["PASS"],
        "fail": all_outcomes["FAIL"],
        "invalid": all_outcomes["INVALID"],
        "valid_trials": valid,
        "valid_success_rate": round(all_outcomes["PASS"] / valid, 4) if valid else None,
        "complete_valid_protocol": valid == sum(expected_trials.values()),
        "per_position": output_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{manifest_path.stem}_{args.checkpoint}"
    json_path = args.output_dir / f"{stem}_summary.json"
    csv_path = args.output_dir / f"{stem}_per_position.csv"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    rate = "n/a" if summary["valid_success_rate"] is None else f"{100 * summary['valid_success_rate']:.1f}%"
    print(
        f"[SUMMARY] valid={valid} pass={all_outcomes['PASS']} fail={all_outcomes['FAIL']} "
        f"invalid={all_outcomes['INVALID']} success={rate} complete={summary['complete_valid_protocol']}"
    )
    print(f"[SUMMARY] json={json_path}")
    print(f"[SUMMARY] csv={csv_path}")


if __name__ == "__main__":
    main()
