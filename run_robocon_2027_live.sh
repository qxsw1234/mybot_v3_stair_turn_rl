#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ldl/anaconda3/envs/robodog_gym/bin/python"
scene="${1:-beam}"
if [[ $# -gt 0 ]]; then shift; fi
case "$scene" in
    slalom|beam|low_frame) ;;
    *) printf '仅支持当前 2027 规则场景：slalom、beam、low_frame\n' >&2; exit 2 ;;
esac
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
request_file="$(mktemp /tmp/robocon_2027_scene.XXXXXXXX)"
trap 'rm -f "$request_file"' EXIT
printf '2027 当前规则场景：4 绕桩，5 单木桥，6 限高；R 重置，W/S/A/D/Q/E 驾驶。\n'
while true; do
    : > "$request_file"
    case "$scene" in
        slalom|low_frame)
            "$PYTHON_BIN" scripts/teleop_robocon_2027_scene.py "$scene" \
                --scene-request-file "$request_file" "$@"
            ;;
        beam)
            "$PYTHON_BIN" scripts/teleop_robocon_2027_beam.py \
                --scene-request-file "$request_file" "$@"
            ;;
    esac
    if [[ ! -s "$request_file" ]]; then break; fi
    next_scene="$(cat "$request_file")"
    case "$next_scene" in
        slalom|beam|low_frame) scene="$next_scene" ;;
        *) printf '无效切换请求：%s\n' "$next_scene" >&2; exit 1 ;;
    esac
    printf '正在打开 2027 规则场景：%s\n' "$scene"
done
