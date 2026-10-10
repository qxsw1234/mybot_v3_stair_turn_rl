#!/usr/bin/env bash
# Formal Low Bar gate evaluation (M1.1 protocol).
#
# Runs every candidate inside one process so per-episode outcomes are paired,
# and re-evaluates the baseline in the same process instead of comparing against
# a stored number.  Results from different --num-envs values are NOT comparable:
# a different batch size consumes a different random stream, so the same seed
# produces different episodes.  Always include the baseline here.
#
# Usage:
#   scripts/run_low_bar_gate.sh --run RUN_DIR --iterations 60920 60922 \
#       [--baseline-run RUN_DIR --baseline-iteration 60898] \
#       [--num-envs 500] [--seeds 20261017 20261018] \
#       [--profile train-matched] [--tag m1_2_stage1]
#
# Any other eval_low_bar_isaac.py flag is passed through unchanged.
# Artifacts:
#   logs/eval_low_bar_gate_<tag>_<stamp>.log
#   <RUN_DIR>/eval_low_bar_gate_<tag>_<stamp>.json
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN=""
TAG="gate"
NUM_ENVS=""
SEED_COUNT=0
PROFILE=""
DRY_RUN=0
declare -a PASSTHROUGH=()

while [ $# -gt 0 ]; do
    case "$1" in
        --run) RUN="$2"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --num-envs) NUM_ENVS="$2"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --seeds)
            PASSTHROUGH+=("$1")
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                PASSTHROUGH+=("$1")
                SEED_COUNT=$((SEED_COUNT + 1))
                shift
            done
            ;;
        --profile) PROFILE="$2"; PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [ -z "$RUN" ]; then
    echo "error: --run is required" >&2
    exit 2
fi
if [ ! -d "$RUN" ]; then
    echo "error: run directory not found: $RUN" >&2
    exit 2
fi

[ -n "$NUM_ENVS" ] || { NUM_ENVS=500; PASSTHROUGH+=(--num-envs 500); }
if [ "$SEED_COUNT" -eq 0 ]; then
    PASSTHROUGH+=(--seeds 20261017 20261018)
    SEED_COUNT=2
fi
[ -n "$PROFILE" ] || PASSTHROUGH+=(--profile train-matched)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$PROJECT/logs/eval_low_bar_gate_${TAG}_${STAMP}.log"
OUT="$RUN/eval_low_bar_gate_${TAG}_${STAMP}.json"
mkdir -p "$PROJECT/logs"

echo "Low Bar gate: tag=$TAG envs=$NUM_ENVS episodes/candidate=$((NUM_ENVS * SEED_COUNT))"
echo "log: $LOG"
echo "out: $OUT"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "command:"
    printf '  python -u %s/scripts/eval_low_bar_isaac.py' "$PROJECT"
    printf ' %q' "${PASSTHROUGH[@]}" --out "$OUT"
    printf '\n'
    exit 0
fi

python -u "$PROJECT/scripts/eval_low_bar_isaac.py" \
    "${PASSTHROUGH[@]}" --out "$OUT" > "$LOG" 2>&1

python3 "$PROJECT/scripts/report_low_bar_gate.py" "$OUT"
