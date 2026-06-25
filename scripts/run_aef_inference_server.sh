#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

python -m aef_inference.server \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-7861}" \
  --config "${AEF_CONFIG:-configs/yajiang_v1_2.yaml}" \
  --manifest "${AEF_MANIFEST:-data/full_npy/train.jsonl}" \
  --deploy-model "${AEF_DEPLOY_MODEL:-outputs/aef_hyh_yajiang_v1_2/exports/aef_hyh_yajiang_v1_2_deploy.pt}" \
  --cache-dir "${AEF_CACHE_DIR:-outputs/aef_inference_service}" \
  --device "${AEF_DEVICE:-auto}"
