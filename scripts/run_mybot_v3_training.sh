#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /home/ldl/anaconda3/etc/profile.d/conda.sh
conda activate robodog_gym

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python -u scripts/train_mybot_v3_stair_turn.py --headless "$@"
