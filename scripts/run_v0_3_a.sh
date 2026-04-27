#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspace/hyh/yajiang-aef
CONFIG=${PROJECT_ROOT}/configs/yajiang_v0_3_a.yaml
MANIFEST=${PROJECT_ROOT}/data/full_npy/train.jsonl
GPU_ID=${GPU_ID:-6}
DEVICE=${DEVICE:-cuda}
SPLIT=${SPLIT:-train}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  echo "Run scripts/prepare_full_npy.py and scripts/build_full_manifest.py first." >&2
  exit 1
fi

NUM_RECORDS=$(wc -l < "${MANIFEST}")
if [[ "${NUM_RECORDS}" -eq 0 ]]; then
  echo "Manifest is empty: ${MANIFEST}" >&2
  exit 1
fi

echo "Running v0.3-a on GPU ${GPU_ID}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"
echo "Records: ${NUM_RECORDS}"
echo "Split: ${SPLIT}"
echo "Device: ${DEVICE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
python scripts/train_with_manifest.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --split "${SPLIT}" \
  --device "${DEVICE}"
