#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/miniconda3/envs/hyh-dl/bin/python}"
LOG_DIR="${PROJECT_ROOT}/agent/runtime/logs"
PID_DIR="${PROJECT_ROOT}/agent/runtime/pids"
AEF_LOG="${LOG_DIR}/aef_inference.log"
AGENT_LOG="${LOG_DIR}/agent_backend.log"

mkdir -p "${LOG_DIR}" "${PID_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

is_running() {
  local pattern="$1"
  pgrep -f "${pattern}" >/dev/null 2>&1
}

start_detached() {
  local log_file="$1"
  local pid_file="$2"
  shift 2
  nohup setsid "$@" >"${log_file}" 2>&1 </dev/null &
  echo $! > "${pid_file}"
}

wait_http() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 60); do
    if curl --noproxy '*' -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} is ready: ${url}"
      return 0
    fi
    sleep 1
  done
  echo "${name} did not become ready: ${url}" >&2
  return 1
}

if is_running "aef_inference.server.*--port 7862"; then
  echo "AEF inference service is already running."
else
  echo "Starting AEF inference service..."
  start_detached "${AEF_LOG}" "${PID_DIR}/aef_inference.pid" \
    "${PYTHON}" -m aef_inference.server \
    --host 127.0.0.1 \
    --port 7862 \
    --config configs/yajiang_v1_2_continue_200.yaml \
    --deploy-model outputs/aef_hyh_yajiang_v1_2_continue_200/exports/aef_hyh_yajiang_v1_2_continue_200_deploy.pt \
    --cache-dir outputs/aef_inference_service_v1_2_continue_200 \
    --device auto
fi
wait_http "http://127.0.0.1:7862/api/health" "AEF inference service"

if is_running "(agent.backend.app|uvicorn agent.backend.app:app).*--port 7870"; then
  echo "Agent backend is already running."
else
  echo "Starting agent backend..."
  start_detached "${AGENT_LOG}" "${PID_DIR}/agent_backend.pid" \
    "${PYTHON}" -m uvicorn agent.backend.app:app \
    --host 0.0.0.0 \
    --port 7870
fi
wait_http "http://127.0.0.1:7870/api/health" "Agent backend"

echo "Public URL: http://112.111.7.74:1112/"
echo "API docs:   http://112.111.7.74:1112/api-docs"
