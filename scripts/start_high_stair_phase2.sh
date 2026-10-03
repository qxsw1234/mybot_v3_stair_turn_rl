#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ldl/mybot_v3_stair_turn_rl
SOURCE_RUN=/home/ldl/mybot_v3_stair_turn_rl/runs/mybot_v3_stair_turn_improved_resume_040500/2026-10-02_03-23-39.173775

source /home/ldl/anaconda3/etc/profile.d/conda.sh
conda activate robodog_gym
cd "$PROJECT"
export PYTHONPATH="$PROJECT"

exec python -u scripts/train_mybot_v3_high_stair_phase2.py \
  --headless \
  --num-envs 5120 \
  --iterations 4000 \
  --resume-run "$SOURCE_RUN" \
  --checkpoint 59000
