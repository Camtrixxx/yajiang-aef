#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/evaluate_model_suite.py \
  --config configs/yajiang_v1_2_continue_100.yaml \
  --manifest data/full_npy/train.jsonl \
  --deploy-model outputs/aef_hyh_yajiang_v1_2_continue_100/exports/aef_hyh_yajiang_v1_2_continue_100_deploy.pt \
  --output-dir outputs/model_eval/v1_2_continue_100 \
  --device auto \
  --max-patches "${MAX_PATCHES:-512}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-pixels-per-patch "${MAX_PIXELS_PER_PATCH:-256}" \
  --demo-indices 4 1425
