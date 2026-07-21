#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIPELINE_DIR=${PROJECT_ROOT}/outputs/v1_2_pipeline
PID_FILE=${PIPELINE_DIR}/pipeline.pid
STOP_FILE=${PIPELINE_DIR}/stop
LOG_FILE=${PIPELINE_DIR}/pipeline.log
RUNNER=${PROJECT_ROOT}/scripts/run_v1_2_pipeline.sh

is_running() {
  [[ -s "${PID_FILE}" ]] || return 1
  local pid
  pid=$(<"${PID_FILE}")
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q 'run_v1_2_pipeline.sh'
}

start_pipeline() {
  mkdir -p "${PIPELINE_DIR}"
  if is_running; then
    echo "v1.2 pipeline is already running (PID $(<"${PID_FILE}"))"
    return 0
  fi

  rm -f "${PID_FILE}" "${STOP_FILE}"
  nohup setsid bash "${RUNNER}" >> "${LOG_FILE}" 2>&1 < /dev/null &
  local launcher_pid=$!
  sleep 2

  if is_running; then
    echo "Started v1.2 pipeline (PID $(<"${PID_FILE}"), launcher PID ${launcher_pid})"
    echo "Log: ${LOG_FILE}"
    return 0
  fi

  echo "Pipeline failed to stay running; inspect ${LOG_FILE}" >&2
  tail -n 40 "${LOG_FILE}" >&2 || true
  return 1
}

show_status() {
  if is_running; then
    echo "v1.2 pipeline is running (PID $(<"${PID_FILE}"))"
  else
    echo "v1.2 pipeline is not running"
  fi
  echo "Log: ${LOG_FILE}"
  if [[ -f "${LOG_FILE}" ]]; then
    tail -n 25 "${LOG_FILE}"
  fi
}

stop_pipeline() {
  mkdir -p "${PIPELINE_DIR}"
  touch "${STOP_FILE}"
  if ! is_running; then
    echo "v1.2 pipeline is not running; stop marker created"
    return 0
  fi

  local pid
  pid=$(<"${PID_FILE}")
  echo "Stopping v1.2 pipeline process group ${pid}"
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
}

case "${1:-status}" in
  start)
    start_pipeline
    ;;
  status)
    show_status
    ;;
  stop)
    stop_pipeline
    ;;
  *)
    echo "Usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
