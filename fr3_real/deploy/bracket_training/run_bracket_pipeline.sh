#!/usr/bin/env bash
# Unattended FR3 bracket pick-and-place pipeline: convert -> config -> norm stats -> train.
#
# Everything below writes to ONE combined log so a single `tail -f` follows the
# whole run, and each phase is fail-fast: if conversion dies, training never
# starts and the status file says so.
#
# Launch (survives logout / SSH disconnect):
#   cd ~/franka_r3 && setsid nohup ./run_bracket_pipeline.sh >/dev/null 2>&1 &
#
# Dry check (no work, ~2 seconds):
#   ./run_bracket_pipeline.sh --check

set -Eeuo pipefail

# ----------------------------------------------------------------------------
# Configuration (override any of these by exporting before launch)
# ----------------------------------------------------------------------------
ROOT="${ROOT:-$HOME/franka_r3}"
OPENPI_DIR="${OPENPI_DIR:-$ROOT/openpi}"
PY="${PY:-$OPENPI_DIR/.venv/bin/python}"

RAW_DIR="${RAW_DIR:-$ROOT/full_bracket_bolt_autonomous_raw_v1}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/lerobot_cache}"
REPO_ID="${REPO_ID:-local/fr3_bracket_droid_79}"

CONVERT_PY="${CONVERT_PY:-$ROOT/convert_bracket_to_lerobot.py}"
PREPARE_PY="${PREPARE_PY:-$ROOT/prepare_pi05_fr3_bracket_full.py}"

CONFIG_NAME="${CONFIG_NAME:-pi05_fr3_bracket_full}"
EXP_NAME="${EXP_NAME:-fr3_bracket_79_$(date +%Y%m%d)}"

STEPS="${STEPS:-13000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
KEEP_PERIOD="${KEEP_PERIOD:-1000}"

FPS="${FPS:-15}"
SCHEMA="${SCHEMA:-droid_joint_velocity}"
DERIVED_VIEW="${DERIVED_VIEW:-bracket_only}"
GRIPPER_MAX_WIDTH="${GRIPPER_MAX_WIDTH:-0.08}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-240}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-80}"
# Episode 52's bolt drive faulted after a clean bracket placement; the operator
# confirmed the bracket half on 2026-08-18. bracket_only resolves no bolt
# boundaries, so its bracket phase is good data. Named explicitly rather than
# admitted by loosening a filter, and still verified for a complete bracket
# event sequence plus a real grasp.
FORCE_INCLUDE="${FORCE_INCLUDE:-20260818_144139_full_bracket_bolt_autonomous_0052_B52_S1_CENTER}"

# Default ON (skip). The proven cube-task launcher run_fr3_real_dell_full.sh
# goes straight from conversion to training with no compute_norm_stats step,
# because configs cloned from pi05_droid_finetune inherit pi05-DROID's
# normalization assets. Recomputing stats from this much smaller dataset would
# also change the action normalization the pretrained checkpoint expects.
SKIP_NORM_STATS="${SKIP_NORM_STATS:-1}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"

# Environment matched to run_fr3_real_dell_full.sh, the launcher proven on this
# machine for the cube task.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
# Without offline mode a training run can block trying to reach wandb.
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# The converter and the trainer must agree on where the LeRobot cache lives.
export HF_LEROBOT_HOME="$CACHE_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-$ROOT/bracket_pipeline_${STAMP}.log}"
STATUS="${STATUS:-$ROOT/bracket_pipeline_status.txt}"
PIDFILE="${PIDFILE:-$ROOT/bracket_pipeline.pid}"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

# ----------------------------------------------------------------------------
# Logging / status plumbing
# ----------------------------------------------------------------------------
if [[ $CHECK_ONLY -eq 0 ]]; then
  exec > >(tee -a "$LOG") 2>&1
fi

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
say()  { echo "[$(ts)] $*"; }
phase() {
  echo
  echo "================================================================================"
  echo "[$(ts)] PHASE: $*"
  echo "================================================================================"
  [[ $CHECK_ONLY -eq 0 ]] && set_status "RUNNING" "$*"
  return 0
}
set_status() {
  [[ $CHECK_ONLY -eq 1 ]] && return 0
  cat > "$STATUS" <<EOF
state       : $1
phase       : $2
started     : $START_HUMAN
updated     : $(ts)
log         : $LOG
config      : $CONFIG_NAME
exp_name    : $EXP_NAME
repo_id     : $REPO_ID
checkpoints : $OPENPI_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME
EOF
}

START_TS=$(date +%s)
START_HUMAN="$(ts)"

on_error() {
  local rc=$?        # must stay the first statement to capture the real code
  local line=$1
  echo
  say "!!! FAILED at line $line (exit $rc)"
  say "!!! Nothing further will run. See $LOG"
  set_status "FAILED" "failed at line $line (exit $rc)"
  exit "$rc"
}
trap 'on_error $LINENO' ERR

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
echo "================================================================================"
echo "FR3 BRACKET PICK-AND-PLACE :: pi0.5-DROID full fine-tune pipeline"
echo "started        : $START_HUMAN"
echo "log            : $LOG"
echo "bracket only   : $DERIVED_VIEW  (no bolt, no threading)"
echo "raw episodes   : $RAW_DIR"
echo "lerobot cache  : $CACHE_ROOT"
echo "repo id        : $REPO_ID"
echo "openpi config  : $CONFIG_NAME"
echo "experiment     : $EXP_NAME"
echo "steps / batch  : $STEPS / $BATCH_SIZE"
echo "norm stats     : $( [[ "$SKIP_NORM_STATS" == "1" ]] && echo 'skipped (inherits pi05-DROID assets)' || echo 'will be computed' )"
echo "wandb / cuda   : $WANDB_MODE / device $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

phase "preflight"
fail=0
check() { if eval "$2"; then say "  ok    $1"; else say "  MISS  $1"; fail=1; fi; }

check "python           $PY"            "[[ -x '$PY' ]]"
check "openpi repo      $OPENPI_DIR"    "[[ -d '$OPENPI_DIR/src/openpi' ]]"
check "train script"                    "[[ -f '$OPENPI_DIR/scripts/train.py' ]]"
check "raw dir          $RAW_DIR"       "[[ -d '$RAW_DIR' ]]"
check "converter        $CONVERT_PY"    "[[ -f '$CONVERT_PY' ]]"
check "prepare script   $PREPARE_PY"    "[[ -f '$PREPARE_PY' ]]"

if [[ -d "$RAW_DIR" ]]; then
  n_dirs=$(find "$RAW_DIR" -mindepth 2 -maxdepth 2 -name metadata.json | wc -l)
  say "  info  raw capture folders on disk: $n_dirs"
  [[ "$n_dirs" -gt 0 ]] || fail=1
fi

# Count what the CONVERTER will actually emit, not how many folders exist. Retakes,
# superseded takes and ineligible captures all mean those two numbers differ (87
# folders -> 80 episodes today), so comparing folder count to EXPECTED_EPISODES
# raised a warning on a completely healthy dataset. This resolves the same
# selection the convert phase will, with the same flags, and fails on a real
# mismatch -- which also catches an interrupted rsync before the long phase.
if [[ -x "$PY" && -d "$RAW_DIR" && -f "$CONVERT_PY" ]]; then
  n_sel=$("$PY" - "$CONVERT_PY" "$RAW_DIR" "$DERIVED_VIEW" $FORCE_INCLUDE <<'PYEOF' 2>/dev/null || echo ERROR
import importlib.util, json, sys
from pathlib import Path
converter, raw_dir, view, *force = sys.argv[1:]
spec = importlib.util.spec_from_file_location("conv", converter)
mod = importlib.util.module_from_spec(spec)
sys.modules["conv"] = mod
spec.loader.exec_module(mod)
force = tuple(force)
selected = 0
for ep in mod.episode_dirs(Path(raw_dir), False, False, True, force):
    metadata = json.loads((ep / "metadata.json").read_text())
    selected += len(mod.derived_episode_views(ep, metadata, view, "x", True, force))
print(selected)
PYEOF
)
  if [[ "$n_sel" == "ERROR" ]]; then
    say "  WARN  could not resolve the converter's episode selection here"
  else
    say "  info  episodes the converter will emit: $n_sel (expected $EXPECTED_EPISODES)"
    if [[ "$n_sel" -ne "$EXPECTED_EPISODES" ]]; then
      say "  FAIL  selected $n_sel != EXPECTED_EPISODES $EXPECTED_EPISODES"
      fail=1
    fi
  fi
fi

if [[ -x "$PY" ]]; then
  if "$PY" -c "import cv2" 2>/dev/null; then say "  ok    cv2 importable"; else say "  MISS  cv2"; fail=1; fi
  if "$PY" -c "import lerobot" 2>/dev/null; then say "  ok    lerobot importable"; else say "  MISS  lerobot"; fail=1; fi
fi

check "norm-stats script" "[[ -f '$OPENPI_DIR/scripts/compute_norm_stats.py' ]]"

# Prove the trainer's CLI actually accepts the shape we are going to call it
# with, before spending ~an hour converting video. Informational only: a
# --help that needs a GPU-free import chain can legitimately be unavailable.
if [[ -x "$PY" && -f "$OPENPI_DIR/scripts/train.py" ]]; then
  if (cd "$OPENPI_DIR" && timeout 180 "$PY" scripts/train.py --help >/dev/null 2>&1); then
    say "  ok    scripts/train.py --help runs"
  else
    say "  info  scripts/train.py --help did not run cleanly (import chain or CLI differs)"
    say "        -> confirm the train invocation near the end of this script matches"
    say "           the one proven in run_fr3_real_dell_full.sh"
  fi
fi

# Purely informational. `pipefail` is on, so this must never be allowed to
# fail the run just because the GPU query is unavailable.
if command -v nvidia-smi >/dev/null 2>&1; then
  { nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null \
      | sed 's/^/  gpu   /'; } || say "  info  nvidia-smi present but returned no data"
fi

if [[ $fail -ne 0 ]]; then
  say "preflight FAILED -- fix the MISS lines above before launching"
  [[ $CHECK_ONLY -eq 0 ]] && set_status "FAILED" "preflight"
  exit 1
fi
say "preflight passed"

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo
  say "--check complete; no work performed."
  exit 0
fi

echo $$ > "$PIDFILE"
say "pid $$ written to $PIDFILE"

# ----------------------------------------------------------------------------
# Phase 1: register the OpenPI training config
#
# Deliberately ahead of conversion: this takes about a second, and a broken
# config here would otherwise only surface after ~an hour of video encoding.
# ----------------------------------------------------------------------------
DEST="$CACHE_ROOT/$REPO_ID"

phase "register OpenPI train config"
cd "$OPENPI_DIR"
cp -n src/openpi/training/config.py "src/openpi/training/config.py.bak_${STAMP}" || true
say "backup: src/openpi/training/config.py.bak_${STAMP}"
"$PY" "$PREPARE_PY" \
  --config-name "$CONFIG_NAME" \
  --repo-id "$REPO_ID" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --replace
"$PY" -c "import ast; ast.parse(open('src/openpi/training/config.py').read()); print('  ok    config.py parses cleanly')"
if "$PY" -c "import openpi.training.config" 2>/dev/null; then
  say "  ok    openpi.training.config imports with the new entry"
else
  say "  info  could not import openpi.training.config here (may need repo root on sys.path)"
fi

# ----------------------------------------------------------------------------
# Phase 2: raw -> LeRobot (bracket portion only)
# ----------------------------------------------------------------------------
if [[ "$SKIP_CONVERT" == "1" ]]; then
  phase "convert (SKIPPED by SKIP_CONVERT=1)"
else
  phase "convert raw -> LeRobot  [bracket portion only]"
  say "destination: $DEST"
  say "this is the long phase -- $EXPECTED_EPISODES episodes of two-camera video"
  "$PY" "$CONVERT_PY" \
    --raw_dir "$RAW_DIR" \
    --output_dir "$CACHE_ROOT" \
    --repo_id "$REPO_ID" \
    --derived_view "$DERIVED_VIEW" \
    --schema "$SCHEMA" \
    --fps "$FPS" \
    --gripper_max_width "$GRIPPER_MAX_WIDTH" \
    --image_width "$IMAGE_WIDTH" \
    --image_height "$IMAGE_HEIGHT" \
    --allow_bracket_only_captures \
    --latest_take_per_episode \
    --force_include $FORCE_INCLUDE \
    --overwrite
  say "conversion finished"
fi

phase "verify converted dataset"
[[ -d "$DEST" ]] || { say "converted dataset missing at $DEST"; exit 1; }
{ du -sh "$DEST" | sed 's/^/  size  /'; } || true
if [[ -f "$DEST/meta/info.json" ]]; then
  "$PY" - "$DEST" "$EXPECTED_EPISODES" <<'PYEOF'
import json, sys
from pathlib import Path
info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
expected = int(sys.argv[2])
for k in ("total_episodes", "total_frames", "fps", "robot_type"):
    if k in info:
        print(f"  info  {k}: {info[k]}")
n = info.get("total_episodes")
if n == 0:
    sys.exit("converted dataset contains zero episodes")
if n is not None and n != expected:
    # bracket_only emits exactly one episode per raw capture, so a mismatch
    # means captures were silently skipped rather than converted.
    print(f"  WARN  expected {expected} converted episodes, found {n}")
PYEOF
else
  say "  note  meta/info.json not found; skipping structured dataset summary"
fi

# ----------------------------------------------------------------------------
# Phase 3: normalization statistics
# ----------------------------------------------------------------------------
if [[ "$SKIP_NORM_STATS" == "1" ]]; then
  phase "norm stats (SKIPPED by SKIP_NORM_STATS=1)"
else
  phase "compute normalization statistics"
  "$PY" scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
  say "norm stats finished"
fi

# ----------------------------------------------------------------------------
# Phase 4: train
# ----------------------------------------------------------------------------
phase "train  ($CONFIG_NAME / $EXP_NAME, $STEPS steps, batch $BATCH_SIZE)"
say "checkpoints -> $OPENPI_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"
"$PY" scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START_TS ))
phase "complete"
say "total wall time: $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m"
say "checkpoints:"
ls -1 "$OPENPI_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME" 2>/dev/null | sed 's/^/  /' || say "  (none found)"
set_status "SUCCESS" "complete"
say "PIPELINE SUCCEEDED"
