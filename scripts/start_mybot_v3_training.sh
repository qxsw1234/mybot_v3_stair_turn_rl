#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/training.pid"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Training is already running with PID $OLD_PID"
    exit 1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/training_$STAMP.log"

nohup setsid "$PROJECT_ROOT/scripts/run_mybot_v3_training.sh" "$@" \
  >"$LOG_FILE" 2>&1 </dev/null &
TRAIN_PID=$!

echo "$TRAIN_PID" > "$PID_FILE"
ln -sfn "$(basename "$LOG_FILE")" "$LOG_DIR/training_latest.log"

echo "Training started"
echo "PID: $TRAIN_PID"
echo "Log: $LOG_FILE"
