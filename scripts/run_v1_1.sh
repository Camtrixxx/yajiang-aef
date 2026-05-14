#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS:-4,5,6,7}"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29611}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_with_manifest.py \
  --config configs/yajiang_v1_1.yaml \
  --manifest data/full_npy/train.jsonl \
  --device auto
