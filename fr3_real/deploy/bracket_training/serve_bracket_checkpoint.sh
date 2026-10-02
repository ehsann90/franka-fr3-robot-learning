#!/usr/bin/env bash
# Serve one bracket checkpoint. Run on the training machine.
#
#   ./serve_bracket_checkpoint.sh 6000
#   ./serve_bracket_checkpoint.sh            # lists the checkpoints available
#
# Exists because the full serve_policy.py invocation is ~200 characters and kept
# being split by terminal line-wrapping on paste. A split at the wrong place made
# serve_policy.py run with NO policy argument, at which point it silently falls
# back to its built-in default (pi0_aloha_sim) and starts a multi-GB download of
# a model that has nothing to do with this task.

set -Eeuo pipefail

ROOT="${ROOT:-$HOME/franka_r3}"
OPENPI_DIR="${OPENPI_DIR:-$ROOT/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_fr3_bracket_full}"
EXP_NAME="${EXP_NAME:-fr3_bracket_79_20260818}"
PORT="${PORT:-8000}"
PY="${PY:-$OPENPI_DIR/.venv/bin/python}"

EXP_DIR="$OPENPI_DIR/checkpoints/$CONFIG_NAME/$EXP_NAME"
[[ -d "$EXP_DIR" ]] || { echo "no such experiment dir: $EXP_DIR" >&2; exit 1; }

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <step>"
  echo
  echo "checkpoints in $EXP_DIR:"
  ls -1 "$EXP_DIR" | grep -E '^[0-9]+$' | sort -n | sed 's/^/  /'
  exit 1
fi

STEP="$1"
CKPT_DIR="$EXP_DIR/$STEP"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "no checkpoint at step $STEP. available:" >&2
  ls -1 "$EXP_DIR" | grep -E '^[0-9]+$' | sort -n | sed 's/^/  /' >&2
  exit 1
fi
# A checkpoint mid-write has no params/ yet; serving it fails in a confusing way.
[[ -d "$CKPT_DIR/params" ]] || { echo "step $STEP has no params/ -- incomplete checkpoint" >&2; exit 1; }

pkill -f serve_policy 2>/dev/null || true
sleep 1

echo "================================================================"
echo " serving  : $CONFIG_NAME @ step $STEP"
echo " dir      : $CKPT_DIR"
echo " port     : $PORT"
echo "================================================================"
echo "Expect 'server listening on 0.0.0.0:$PORT' and NO 'Downloading gs://' line."
echo

cd "$OPENPI_DIR"
exec "$PY" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config="$CONFIG_NAME" \
  --policy.dir="$CKPT_DIR"
