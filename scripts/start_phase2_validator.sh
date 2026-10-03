#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIRECTORY TRAINING_PID" >&2
  exit 2
fi

PROJECT=/home/ldl/mybot_v3_stair_turn_rl
RUN_DIRECTORY=$1
TRAINING_PID=$2

source /home/ldl/anaconda3/etc/profile.d/conda.sh
conda activate robodog_gym
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

exec python -u scripts/monitor_phase2_validation.py \
  --run "$RUN_DIRECTORY" \
  --training-pid "$TRAINING_PID" \
  --start 59000 \
  --end 62999 \
  --poll-seconds 30
