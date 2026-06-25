#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${PROJECT_ROOT}/configs/yajiang_v1_3_continue_200.yaml
MANIFEST=${PROJECT_ROOT}/data/full_npy/train.jsonl
RESUME=${RESUME:-${PROJECT_ROOT}/outputs/aef_hyh_yajiang_v1_3/checkpoints/best.pt}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE=${#_GPU_LIST[@]}
fi
MASTER_PORT=${MASTER_PORT:-29616}
DEVICE=${DEVICE:-auto}
SPLIT=${SPLIT:-train}
OUTPUT_DIR="outputs/aef_hyh_yajiang_v1_3_continue_200"
LOG_DIR="${OUTPUT_DIR}/logs"
CONSOLE_LOG="${LOG_DIR}/console.log"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for path in "${CONFIG}" "${MANIFEST}" "${RESUME}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"

echo "Continuing v1.3 to 200 epochs on CUDA GPU(s) ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"
echo "Resume: ${RESUME}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "Processes: ${NPROC_PER_NODE}"
echo "Logging console output to ${CONSOLE_LOG}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" scripts/train_with_manifest.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --resume "${RESUME}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" 2>&1 | tee "${CONSOLE_LOG}"
