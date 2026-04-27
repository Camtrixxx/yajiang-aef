#!/usr/bin/env bash
set -e

PROJECT_ROOT=/workspace/hyh/yajiang-aef
CONFIG=${PROJECT_ROOT}/configs/yajiang_v0_2_c.yaml
MANIFEST=${PROJECT_ROOT}/data/debug_small_npy/train.jsonl

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

echo "Running v0.2-c on GPU 6"
echo "Config: ${CONFIG}"
echo "Manifest: ${MANIFEST}"

CUDA_VISIBLE_DEVICES=6 \
python scripts/train_with_manifest.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}"