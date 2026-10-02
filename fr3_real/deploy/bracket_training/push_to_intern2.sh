#!/usr/bin/env bash
# Push the bracket dataset + pipeline scripts to the training machine.
# Run this from the laptop. rsync will prompt for the intern2 password each
# time (no key is installed), so expect up to two prompts.
#
#   ./push_to_intern2.sh              # scripts + the 10 GB raw dataset
#   ./push_to_intern2.sh --scripts    # scripts only (fast; for re-pushing fixes)

set -Eeuo pipefail

REMOTE="${REMOTE:-intern2@10.6.38.133}"
REMOTE_ROOT="${REMOTE_ROOT:-franka_r3}"          # relative to the remote $HOME
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FR3_ROOT="$(cd "$HERE/../.." && pwd)"
RAW_DIR="$FR3_ROOT/recordings/full_bracket_bolt_autonomous_raw_v1"

SCRIPTS_ONLY=0
[[ "${1:-}" == "--scripts" ]] && SCRIPTS_ONLY=1

echo "remote      : $REMOTE:~/$REMOTE_ROOT"
echo "raw dataset : $RAW_DIR"
echo

if [[ $SCRIPTS_ONLY -eq 0 ]]; then
  [[ -d "$RAW_DIR" ]] || { echo "missing raw dir: $RAW_DIR" >&2; exit 1; }
  n=$(find "$RAW_DIR" -mindepth 2 -maxdepth 2 -name metadata.json | wc -l)
  echo "==> pushing $n raw episodes (~10 GB, resumable if interrupted)"
  rsync -avh --partial --progress \
    "$RAW_DIR/" \
    "$REMOTE:$REMOTE_ROOT/full_bracket_bolt_autonomous_raw_v1/"
  echo
fi

echo "==> pushing pipeline scripts"
# The converter is renamed on the remote so it cannot shadow the existing
# convert_raw_recordings_to_lerobot.py that the cube-task pipeline uses.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$HERE/convert_raw_recordings_to_lerobot.py" "$TMP/convert_bracket_to_lerobot.py"
cp "$HERE/prepare_pi05_fr3_bracket_full.py"     "$TMP/"
cp "$HERE/run_bracket_pipeline.sh"              "$TMP/"
cp "$HERE/watch_bracket_pipeline.sh"            "$TMP/"
cp "$HERE/serve_bracket_checkpoint.sh"         "$TMP/"
cp "$HERE/BRACKET_TRAINING_HANDOFF.md"          "$TMP/"
chmod +x "$TMP"/*.sh

rsync -avh "$TMP/" "$REMOTE:$REMOTE_ROOT/"

echo
echo "done. next, on the training machine:"
echo "  ssh $REMOTE"
echo "  cd ~/$REMOTE_ROOT && ./run_bracket_pipeline.sh --check"
