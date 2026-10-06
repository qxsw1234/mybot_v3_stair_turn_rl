#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ldl/anaconda3/envs/robodog_gym/bin/python"
SIM_SCRIPT="$PROJECT_ROOT/scripts/sim2sim_mujoco_robocon_map.py"
DEFAULT_DURATION="${MUJOCO_DURATION:-3600}"

usage() {
    printf '%s\n' \
        "用法:" \
        "  ./run_mujoco.sh [start|t_stairs|wall|low_bar|slope|slalom|gravel|bridge]" \
        "  ./run_mujoco.sh --replace [位置]   # 关闭旧窗口后重新启动" \
        "  ./run_mujoco.sh --stop             # 关闭当前仿真" \
        "" \
        "默认位置: start"
}

running_pids() {
    # Match both an absolute script path and the usual project-relative path.
    pgrep -f -- "scripts/sim2sim_mujoco_robocon_map.py" || true
}

stop_existing() {
    local pids
    pids="$(running_pids)"
    if [[ -z "$pids" ]]; then
        printf '%s\n' "[launcher] 当前没有运行中的 RoboCon MuJoCo。"
        return
    fi
    printf '%s\n' "[launcher] 正在停止已有 MuJoCo: ${pids//$'\n'/ }"
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && kill "$pid"
    done <<< "$pids"
    for _ in {1..30}; do
        [[ -z "$(running_pids)" ]] && return
        sleep 0.1
    done
    printf '%s\n' "[launcher] 旧进程仍未退出，请检查: $(running_pids)" >&2
    return 1
}

replace=false
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --stop)
        stop_existing
        exit 0
        ;;
    --replace)
        replace=true
        shift
        ;;
esac

spawn="${1:-start}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$spawn" in
    start|t_stairs|wall|low_bar|slope|slalom|gravel|bridge) ;;
    *)
        printf '%s\n' "[launcher] 未知位置: $spawn" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ "$replace" == true ]]; then
    stop_existing
else
    pids="$(running_pids)"
    if [[ -n "$pids" ]]; then
        printf '%s\n' \
            "[launcher] MuJoCo 已经在运行（PID: ${pids//$'\n'/ }）。" \
            "[launcher] 如需重启，请运行: ./run_mujoco.sh --replace $spawn"
        exit 0
    fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "[launcher] Python 不存在: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$SIM_SCRIPT" ]]; then
    printf '%s\n' "[launcher] 仿真入口不存在: $SIM_SCRIPT" >&2
    exit 1
fi

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
if [[ ! -f "$XAUTHORITY" ]]; then
    printf '%s\n' "[launcher] Xauthority 不存在: $XAUTHORITY" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
printf '%s\n' \
    "[launcher] 启动 RoboCon MuJoCo，位置=$spawn，时长=${DEFAULT_DURATION}s" \
    "[launcher] W/S 前后，A/D 横移，Q/E 转向，R 复位；Ctrl+C 退出"
exec "$PYTHON_BIN" -u "$SIM_SCRIPT" \
    --duration "$DEFAULT_DURATION" \
    --spawn "$spawn" \
    "$@"
