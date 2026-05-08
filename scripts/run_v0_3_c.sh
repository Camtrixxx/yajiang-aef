#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspace/hyh/yajiang-aef
CONFIG=${PROJECT_ROOT}/configs/yajiang_v0_3_c.yaml
MANIFEST=${PROJECT_ROOT}/data/full_npy/train.jsonl
NPU_IDS=${NPU_IDS:-4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29583}
DEVICE=${DEVICE:-auto}
SPLIT=${SPLIT:-train}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Run scripts/prepare_jrc_water_npy.py and scripts/build_full_manifest.py first." >&2
  exit 1
fi

NUM_RECORDS=$(wc -l < "${MANIFEST}")
if [[ "${NUM_RECORDS}" -eq 0 ]]; then
  echo "Manifest is empty: ${MANIFEST}" >&2
  exit 1
fi

echo "Running v0.3c on NPU(s) ${NPU_IDS}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"
echo "Records: ${NUM_RECORDS}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"
echo "Processes: ${NPROC_PER_NODE}"

ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" scripts/train_with_manifest.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --split "${SPLIT}" \
  --device "${DEVICE}"
