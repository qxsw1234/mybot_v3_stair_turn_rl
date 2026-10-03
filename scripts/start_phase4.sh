#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/ldl/mybot_v3_stair_turn_rl
PYTHON=/home/ldl/anaconda3/envs/robodog_gym/bin/python
ENV_BIN=/home/ldl/anaconda3/envs/robodog_gym/bin
SOURCE_RUN="${PHASE4_SOURCE_RUN:-$PROJECT/runs/mybot_v3_sim2sim_phase3_resume_059000/2026-10-03_02-38-13.891327}"
CHECKPOINT="${PHASE4_CHECKPOINT:-59250}"
ITERATIONS="${PHASE4_ITERATIONS:-1500}"
FROM_SCRATCH="${PHASE4_FROM_SCRATCH:-0}"
if [[ "$FROM_SCRATCH" == "1" ]]; then
    CHECKPOINT=0
fi
END_ITERATION=$((CHECKPOINT + ITERATIONS - 1))
WAIT_PID="${1:-}"
LOG_DIR="$PROJECT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="$LOG_DIR/phase4_training_${STAMP}.log"
MONITOR_LOG="$LOG_DIR/phase4_monitor_${STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT"
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [[ -n "$WAIT_PID" ]]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 15
    done
fi

# Give CUDA a moment to release the previous PhysX context.
sleep 5

if pgrep -f "^$PYTHON -u scripts/train_mybot_v3_sim2sim_phase4.py " >/dev/null; then
    echo "Phase-4 training is already running; refusing to start a duplicate." >&2
    exit 1
fi

TRAIN_ARGS=(
    --headless
    --num-envs 4096
    --iterations "$ITERATIONS"
)
if [[ "$FROM_SCRATCH" != "1" ]]; then
    TRAIN_ARGS+=(--resume-run "$SOURCE_RUN" --checkpoint "$CHECKPOINT")
fi
"$PYTHON" -u scripts/train_mybot_v3_sim2sim_phase4.py \
    "${TRAIN_ARGS[@]}" >>"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$LOG_DIR/phase4_training.pid"

RUN_DIR=""
for _ in $(seq 1 120); do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "Phase-4 training exited before creating its run directory." >&2
        tail -80 "$TRAIN_LOG" >&2
        exit 1
    fi
    RUN_DIR="$(sed -n 's/^Loggind directory: //p' "$TRAIN_LOG" | tail -1)"
    if [[ -n "$RUN_DIR" && -d "$RUN_DIR" ]]; then
        break
    fi
    sleep 2
done

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "Timed out waiting for the Phase-4 run directory." >&2
    exit 1
fi

printf '%s\n' "$RUN_DIR" > "$LOG_DIR/phase4_run_dir.txt"
"$PYTHON" -u scripts/monitor_phase4_sim2sim.py \
    --run "$RUN_DIR" \
    --training-pid "$TRAIN_PID" \
    --start "$CHECKPOINT" \
    --end "$END_ITERATION" \
    --poll-seconds 15 \
    >>"$MONITOR_LOG" 2>&1 &
MONITOR_PID=$!
printf '%s\n' "$MONITOR_PID" > "$LOG_DIR/phase4_monitor.pid"

wait "$TRAIN_PID"
