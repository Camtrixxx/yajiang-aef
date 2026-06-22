#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m agent.backend.app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7870}"
