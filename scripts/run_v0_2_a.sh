#!/usr/bin/env bash
set -e

PROJECT_ROOT=/workspace/hyh/yajiang-aef
CONFIG=${PROJECT_ROOT}/configs/yajiang_v0_2_a.yaml
MANIFEST=${PROJECT_ROOT}/data/debug_small_npy/train.jsonl
NPU_ID=${NPU_ID:-0}
DEVICE=${DEVICE:-npu:0}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

echo "Running v0.2-a on NPU ${NPU_ID}"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"
echo "Device: ${DEVICE}"

ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}" \
python scripts/train_with_manifest.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --device "${DEVICE}"
