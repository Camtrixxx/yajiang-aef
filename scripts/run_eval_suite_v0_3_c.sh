#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Evaluation is inference-heavy and normally runs comfortably on one Ascend NPU.
# Keep it on the agreed back-half cards by default.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"

python scripts/evaluate_model_suite.py \
  --config configs/yajiang_v0_3_c.yaml \
  --manifest data/full_npy/train.jsonl \
  --deploy-model outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt \
  --output-dir outputs/model_eval/v0_3_c \
  --device auto \
  --max-patches "${MAX_PATCHES:-128}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-pixels-per-patch "${MAX_PIXELS_PER_PATCH:-128}" \
  --demo-indices 4 1425
