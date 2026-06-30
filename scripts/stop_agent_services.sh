#!/usr/bin/env bash
set -euo pipefail

stop_pattern() {
  local label="$1"
  local pattern="$2"
  local pids
  pids="$(pgrep -f "${pattern}" || true)"
  if [[ -z "${pids}" ]]; then
    echo "${label} is not running."
    return 0
  fi
  echo "Stopping ${label}: ${pids}"
  # shellcheck disable=SC2086
  kill ${pids}
}

stop_pattern "Agent backend" "(agent.backend.app|uvicorn agent.backend.app:app).*--port 7870"
stop_pattern "AEF inference service" "aef_inference.server.*--port 7862"
