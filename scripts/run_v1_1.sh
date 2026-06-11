#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(python - <<'PY'
import os
print(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
PY
)"
fi
MASTER_PORT="${MASTER_PORT:-29611}"
OUTPUT_DIR="outputs/aef_hyh_yajiang_v1_1"
LOG_DIR="${OUTPUT_DIR}/logs"
CONSOLE_LOG="${LOG_DIR}/console.log"

mkdir -p "${LOG_DIR}"

echo "Logging console output to ${CONSOLE_LOG}"
echo "Running v1.1 on CUDA GPU(s) ${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${NPROC_PER_NODE}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_with_manifest.py \
  --config configs/yajiang_v1_1.yaml \
  --manifest data/full_npy/train.jsonl \
  --device auto 2>&1 | tee "${CONSOLE_LOG}"
