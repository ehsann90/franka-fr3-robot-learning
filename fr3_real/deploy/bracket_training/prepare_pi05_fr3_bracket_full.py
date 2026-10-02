#!/usr/bin/env python3
"""Add a full pi0.5-DROID fine-tuning config for the FR3 bracket pick-and-place task.

Run from the OpenPI repository root. This clones OpenPI's built-in
``pi05_droid_finetune`` TrainConfig -- keeping the pi05-DROID checkpoint, action
representation, and normalization assets -- and repoints it at the bracket-only
LeRobot dataset with a run length sized for that dataset.

This is deliberately a separate script from ``prepare_pi05_fr3_real_droid_full.py``
(the cube pick-and-place one). That script hardcodes
``repo_id="local/fr3_real_pick_place_droid"`` and step counts tuned for its
118-episode dataset, so it cannot be reused here.

Sizing basis (measured from the 50 accepted raw captures on 2026-08-18):
  bracket span 26.5-47.4 s, mean 33.2 s  ->  ~24,900 frames at 15 Hz
  13000 steps x batch 48 = 624,000 samples ~= 25 epochs over that dataset.

25 epochs is a lot of passes over 27 minutes of data, so the run is deliberately
checkpointed every 1000 steps. The final checkpoint is not assumed to be the best
one -- evaluate several and pick on held-out bracket positions, not on loss.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CONFIG_PATH = Path("src/openpi/training/config.py")
BASE_NAME = "pi05_droid_finetune"

DEFAULT_NEW_NAME = "pi05_fr3_bracket_full"
DEFAULT_REPO_ID = "local/fr3_bracket_droid"
DEFAULT_STEPS = 13_000
DEFAULT_BATCH_SIZE = 48
DEFAULT_SAVE_INTERVAL = 1_000
DEFAULT_KEEP_PERIOD = 2_000
DEFAULT_WARMUP_STEPS = 500
DEFAULT_PEAK_LR = 1.0e-5
DEFAULT_DECAY_LR = 1.0e-6


def extract_train_config(text: str, config_name: str) -> tuple[int, int, str]:
    """Return (start, end, source) of the TrainConfig literal named config_name."""
    marker = f'name="{config_name}"'
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find config {config_name!r}")

    block_start = text.rfind("TrainConfig(", 0, start)
    if block_start < 0:
        raise RuntimeError(f"Could not find the TrainConfig for {config_name!r}")

    depth = 0
    for index, char in enumerate(text[block_start:]):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                block_end = block_start + index + 1
                return block_start, block_end, text[block_start:block_end]
    raise RuntimeError(f"Could not bracket config {config_name!r}")


def replace_required(block: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, block, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one config match for {pattern!r}, found {count}")
    return updated


def remove_existing(text: str, config_name: str) -> str:
    """Drop a previously generated config block, including its separating comma."""
    block_start, block_end, _ = extract_train_config(text, config_name)
    # The generator inserts blocks as ",\n    <TrainConfig...>". Reclaim that
    # separator too so repeated --replace runs cannot accumulate stray commas.
    prefix = text[:block_start]
    stripped = prefix.rstrip()
    if stripped.endswith(","):
        prefix = stripped[:-1]
    return prefix + text[block_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default=DEFAULT_NEW_NAME)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help="Must exactly match the --repo_id used during LeRobot conversion.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument("--keep-period", type=int, default=DEFAULT_KEEP_PERIOD)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--peak-lr", type=float, default=DEFAULT_PEAK_LR)
    parser.add_argument("--decay-lr", type=float, default=DEFAULT_DECAY_LR)
    parser.add_argument("--replace", action="store_true",
                        help="Rewrite the config if it already exists (needed to change any value above).")
    args = parser.parse_args()

    if not CONFIG_PATH.exists():
        raise SystemExit("Run this from the OpenPI repository root (src/openpi/training/config.py not found).")

    text = CONFIG_PATH.read_text()
    exists = f'name="{args.config_name}"' in text
    if exists and not args.replace:
        print(f"[skip] config {args.config_name!r} already exists (pass --replace to rewrite it)")
        return
    if exists:
        print(f"[drop] removing previous {args.config_name!r} block")
        text = remove_existing(text, args.config_name)

    _, insert_at, block = extract_train_config(text, BASE_NAME)
    block = block.replace(f'name="{BASE_NAME}"', f'name="{args.config_name}"', 1)
    block = replace_required(block, r'repo_id="[^"]+"', f'repo_id="{args.repo_id}"')
    block = replace_required(block, r"num_train_steps=[\d_]+", f"num_train_steps={args.steps}")
    block = replace_required(block, r"batch_size=[\d_]+", f"batch_size={args.batch_size}")
    block = replace_required(block, r"save_interval=[\d_]+", f"save_interval={args.save_interval}")
    block = replace_required(block, r"keep_period=[\d_]+", f"keep_period={args.keep_period}")

    # Cosine decay is matched to the run length so the schedule actually lands on
    # decay_lr at the final step instead of being truncated mid-decay.
    schedule = f"""        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps={args.warmup_steps},
            peak_lr={args.peak_lr:g},
            decay_steps={args.steps},
            decay_lr={args.decay_lr:g},
        ),
"""
    anchor = f"        num_train_steps={args.steps},"
    if anchor not in block:
        raise RuntimeError("Could not find schedule insertion point")
    block = block.replace(anchor, schedule + anchor, 1)

    CONFIG_PATH.write_text(text[:insert_at] + ",\n    " + block + text[insert_at:])
    print(f"[add ] config {args.config_name!r}")
    print(f"       dataset repo_id : {args.repo_id}")
    print(f"       full-parameter fine-tune, batch={args.batch_size}, steps={args.steps}")
    print(f"       cosine: warmup={args.warmup_steps}, peak_lr={args.peak_lr:g} -> decay_lr={args.decay_lr:g}")
    print(f"       checkpoints every {args.save_interval} steps, kept permanently every {args.keep_period}")


if __name__ == "__main__":
    main()
