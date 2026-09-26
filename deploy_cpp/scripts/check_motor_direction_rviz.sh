#!/usr/bin/env bash
set -euo pipefail

PORT0="${PORT0:-/dev/ttyUSB0}"
PORT1="${PORT1:-/dev/ttyUSB1}"
RATE_HZ="${RATE_HZ:-50.0}"
RVIZ="${RVIZ:-true}"
CONFIG_FILE="${CONFIG_FILE:-}"
ASSUME_YES=0

source_setup_if_exists() {
  local setup_file="$1"
  if [[ ! -f "${setup_file}" ]]; then
    return 0
  fi

  set +u
  # shellcheck disable=SC1090
  source "${setup_file}"
  local status=$?
  set -u
  return "${status}"
}

usage() {
  cat <<'USAGE'
Usage:
  deploy_cpp/scripts/check_motor_direction_rviz.sh [options]

Options:
  --port0 PATH        Serial port for CAN IDs 1-6. Default: /dev/ttyUSB0
  --port1 PATH        Serial port for CAN IDs 7-12. Default: /dev/ttyUSB1
  --rate-hz HZ        Motor read/publish rate. Default: 50.0
  --config PATH       Robot YAML config. Default: deploy_cpp/config/robots/mybot_v2_1_cse.yaml
  --no-rviz           Do not launch RViz.
  --yes               Skip the safety confirmation prompt.
  -h, --help          Show this help.

This starts deploy_cpp motor_debug_node. The node sends q=0, dq=0,
kp=0, kd=0, tau=0 to every motor, then publishes the encoder feedback
as /joint_states for RViz.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port0)
      PORT0="$2"
      shift 2
      ;;
    --port0=*)
      PORT0="${1#*=}"
      shift
      ;;
    --port1)
      PORT1="$2"
      shift 2
      ;;
    --port1=*)
      PORT1="${1#*=}"
      shift
      ;;
    --rate-hz)
      RATE_HZ="$2"
      shift 2
      ;;
    --rate-hz=*)
      RATE_HZ="${1#*=}"
      shift
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --config=*)
      CONFIG_FILE="${1#*=}"
      shift
      ;;
    --no-rviz)
      RVIZ="false"
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DEPLOY_CPP_DIR}/.." && pwd)"

if [[ -z "${CONFIG_FILE}" ]]; then
  CONFIG_FILE="${DEPLOY_CPP_DIR}/config/robots/mybot_v2_1_cse.yaml"
fi

source_setup_if_exists /opt/ros/humble/setup.bash

if [[ -f "${REPO_ROOT}/install/setup.bash" ]]; then
  source_setup_if_exists "${REPO_ROOT}/install/setup.bash"
elif [[ -f "${DEPLOY_CPP_DIR}/install/setup.bash" ]]; then
  source_setup_if_exists "${DEPLOY_CPP_DIR}/install/setup.bash"
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 was not found. Source /opt/ros/humble/setup.bash and the workspace setup.bash first." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ "${ASSUME_YES}" -ne 1 ]]; then
  echo "Motor direction check will send zero command to all motors:"
  echo "  q=0, dq=0, kp=0, kd=0, tau=0"
  echo "The robot will not hold itself. Support or lift the robot before continuing."
  printf 'Type YES to continue: '
  read -r answer
  if [[ "${answer}" != "YES" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

echo "Starting motor direction check:"
echo "  port0: ${PORT0}"
echo "  port1: ${PORT1}"
echo "  rate:  ${RATE_HZ} Hz"
echo "  rviz:  ${RVIZ}"
echo "  cfg:   ${CONFIG_FILE}"

exec ros2 launch deploy_cpp motor_debug.launch.py \
  port0:="${PORT0}" \
  port1:="${PORT1}" \
  rate_hz:="${RATE_HZ}" \
  rviz:="${RVIZ}" \
  config_file:="${CONFIG_FILE}"
