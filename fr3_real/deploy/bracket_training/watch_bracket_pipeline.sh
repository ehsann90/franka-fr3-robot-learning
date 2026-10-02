#!/usr/bin/env bash
# Single pane of glass for the bracket pipeline: conversion AND training share
# one log, so this shows the run's state, progress, and then follows that log.
#
#   ./watch_bracket_pipeline.sh            # summary, then follow the live log
#   ./watch_bracket_pipeline.sh --status   # summary only, exit immediately
#   ./watch_bracket_pipeline.sh --tail 200 # summary, then follow with more backlog

set -uo pipefail

ROOT="${ROOT:-$HOME/franka_r3}"
OPENPI_DIR="${OPENPI_DIR:-$ROOT/openpi}"
STATUS="${STATUS:-$ROOT/bracket_pipeline_status.txt}"
PIDFILE="${PIDFILE:-$ROOT/bracket_pipeline.pid}"

STATUS_ONLY=0
TAIL_LINES=60
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status) STATUS_ONLY=1; shift ;;
    --tail)   TAIL_LINES="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

LOG="$(ls -1t "$ROOT"/bracket_pipeline_*.log 2>/dev/null | head -1)"

echo "================================================================================"
echo " FR3 BRACKET PIPELINE  ::  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

if [[ -f "$STATUS" ]]; then
  sed 's/^/  /' "$STATUS"
else
  echo "  (no status file yet at $STATUS)"
fi

echo
if [[ -f "$PIDFILE" ]]; then
  pid="$(cat "$PIDFILE")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "  process     : RUNNING (pid $pid)"
    ps -o etime=,cmd= -p "$pid" 2>/dev/null | sed 's/^/                /'
  else
    echo "  process     : NOT RUNNING (stale pid $pid) -- check the status line above"
  fi
else
  echo "  process     : no pidfile"
fi

# Checkpoint progress, if training has reached that point.
ckdir=""
if [[ -f "$STATUS" ]]; then
  ckdir="$(awk -F': ' '/^checkpoints/{print $2}' "$STATUS" | tr -d ' ')"
fi
if [[ -n "$ckdir" && -d "$ckdir" ]]; then
  echo
  echo "  checkpoints in $ckdir:"
  ls -1 "$ckdir" 2>/dev/null | sort -n | sed 's/^/                /'
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/  gpu         : /'
fi

echo
if [[ -z "$LOG" ]]; then
  echo "  no log file found under $ROOT/bracket_pipeline_*.log"
  exit 0
fi
echo "  log         : $LOG"

# Phase timeline pulled straight out of the combined log.
echo
echo "  phase timeline:"
grep -E '^\[.*\] PHASE: ' "$LOG" 2>/dev/null | sed 's/^/                /' || echo "                (none yet)"

if [[ $STATUS_ONLY -eq 1 ]]; then
  echo
  echo "  (--status: not following. Re-run without --status to tail the live log.)"
  exit 0
fi

echo
echo "================================================================================"
echo " following $LOG  (Ctrl-C to stop; this does not affect the run)"
echo "================================================================================"
exec tail -n "$TAIL_LINES" -f "$LOG"
