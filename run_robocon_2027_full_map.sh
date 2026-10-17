#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ldl/anaconda3/envs/robodog_gym/bin/python"
if [[ -z "${DISPLAY:-}" ]]; then export DISPLAY=:0; fi
if [[ -z "${XAUTHORITY:-}" ]]; then
    if [[ "$DISPLAY" == :0* ]]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    else
        export XAUTHORITY="$HOME/.Xauthority"
    fi
fi
if [[ ! -f "$XAUTHORITY" ]]; then
    printf '图形会话授权文件不存在：%s\n' "$XAUTHORITY" >&2
    exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON_BIN" scripts/play_robocon_2027_full_map.py "$@"
