#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/ldl/anaconda3/envs/robodog_gym/bin/python"

usage() {
    cat <<'EOF'
可用场景（键盘控制；窗口显示在 desktop 图形桌面）：
  1 / map       历史完整地图：起点、T 台阶、高墙、限高、坡、绕杆、砂砾坑、木桥；窗口内按 1-8 跳转
  2 / slalom    2027 规则绕桩：059950；P 切换冻结路径自动控制
  3 / beam      2027 规则单木桥：059950 ↔ 060148；1/2/3 切换策略模式
  4 / low_frame 2027 规则限高：064341 Stage 1 研究权重；P 切换直线自动控制
  5 / stairs    历史 8/10/12 cm 三档直线台阶：059950；窗口内按 1-3 切换档位

用法：
  ./run_robocon_scenes.sh                  # 交互菜单
  ./run_robocon_scenes.sh --list
  ./run_robocon_scenes.sh map bridge       # 历史完整地图的木桥入口
  ./run_robocon_scenes.sh slalom --autopilot
  ./run_robocon_scenes.sh beam --duration 3600
  ./run_robocon_scenes.sh low_frame --autopilot
  ./run_robocon_scenes.sh stairs

2027 梅花桩、轮胎阵、沟壑组合、高墙和砖木坑暂无规则尺寸可验收仿真场景。
历史地图仅供体验这些障碍的既有几何与通用步态，不能记作 2027 规则成绩。
EOF
}

scene="${1:-}"
if [[ -z "$scene" ]]; then
    usage
    printf '\n选择 1-5：'
    read -r scene
fi
if [[ "$scene" == "--list" || "$scene" == "-h" || "$scene" == "--help" ]]; then
    usage
    exit 0
fi
shift "$(( $# > 0 ? 1 : 0 ))"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python 不存在：%s\n' "$PYTHON_BIN" >&2
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
if [[ ! -f "$XAUTHORITY" ]]; then
    printf '图形会话授权文件不存在：%s\n' "$XAUTHORITY" >&2
    exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
case "$scene" in
    1|map)
        spawn="${1:-start}"
        if [[ $# -gt 0 ]]; then shift; fi
        printf '打开历史完整地图：%s（非 2027 规则场地）\n' "$spawn"
        exec ./run_mujoco.sh "$spawn" "$@"
        ;;
    2|slalom)
        exec "$PYTHON_BIN" scripts/teleop_robocon_2027_scene.py slalom "$@"
        ;;
    3|beam)
        exec "$PYTHON_BIN" scripts/teleop_robocon_2027_beam.py "$@"
        ;;
    4|low_frame)
        exec "$PYTHON_BIN" scripts/teleop_robocon_2027_scene.py low_frame "$@"
        ;;
    5|stairs)
        printf '打开历史直线台阶；不是 2027 沟壑组合规则场景。\n'
        exec "$PYTHON_BIN" scripts/sim2sim_mujoco_3stairs.py "$@"
        ;;
    *)
        printf '未知场景：%s\n' "$scene" >&2
        usage >&2
        exit 2
        ;;
esac
