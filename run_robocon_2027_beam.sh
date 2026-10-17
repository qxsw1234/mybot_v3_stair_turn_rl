#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ldl/anaconda3/envs/robodog_gym/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python 不存在: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
    export DISPLAY=:0
fi
if [[ -z "${XAUTHORITY:-}" ]]; then
    if [[ "$DISPLAY" == :0* ]]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    else
        export XAUTHORITY="$HOME/.Xauthority"
    fi
fi
if [[ ! -f "${XAUTHORITY:-}" ]]; then
    printf '图形会话授权文件不存在: %s\n' "${XAUTHORITY:-未设置}" >&2
    exit 1
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" scripts/teleop_robocon_2027_beam.py "$@"
