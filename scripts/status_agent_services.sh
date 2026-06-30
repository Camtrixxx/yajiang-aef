#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

show_process() {
  local label="$1"
  local pattern="$2"
  echo "== ${label} process =="
  pgrep -af "${pattern}" || true
}

show_http() {
  local label="$1"
  local url="$2"
  printf "== %s ==\n" "${label}"
  curl --noproxy '*' --connect-timeout 5 --max-time 10 -s -o /dev/null -w "%{http_code} %{content_type} %{errormsg}\n" "${url}" || true
}

show_process "AEF inference service" "aef_inference.server.*--port 7862"
show_process "Agent backend" "(agent.backend.app|uvicorn agent.backend.app:app).*--port 7870"
show_http "AEF local health" "http://127.0.0.1:7862/api/health"
show_http "Agent local health" "http://127.0.0.1:7870/api/health"
show_http "Agent public UI" "http://112.111.7.74:1112/"
show_http "Agent public docs" "http://112.111.7.74:1112/api-docs"
