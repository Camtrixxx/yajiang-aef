#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${PROJECT_ROOT}/configs/yajiang_v1_2.yaml
MANIFEST=${PROJECT_ROOT}/data/full_npy/train.jsonl
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra _GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE=${#_GPU_LIST[@]}
fi
MASTER_PORT=${MASTER_PORT:-29612}
DEVICE=${DEVICE:-auto}
SPLIT=${SPLIT:-train}
OUTPUT_DIR="outputs/aef_hyh_yajiang_v1_2"
LOG_DIR="${OUTPUT_DIR}/logs"
CONSOLE_LOG="${LOG_DIR}/console.log"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Run scripts/build_full_manifest.py first." >&2
  exit 1
fi

NUM_RECORDS=$(wc -l < "${MANIFEST}")
if [[ "${NUM_RECORDS}" -eq 0 ]]; then
  echo "Manifest is empty: ${MANIFEST}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Running v1.2 on CUDA GPU(s) ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"
echo "Records: ${NUM_RECORDS}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "Processes: ${NPROC_PER_NODE}"
echo "Logging console output to ${CONSOLE_LOG}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" scripts/train_with_manifest.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" 2>&1 | tee "${CONSOLE_LOG}"
