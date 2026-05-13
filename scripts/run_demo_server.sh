#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

python scripts/serve_demo.py \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7860}" \
  --config configs/yajiang_v0_3_c.yaml \
  --manifest data/full_npy/train.jsonl \
  --deploy-model outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt \
  --device auto
