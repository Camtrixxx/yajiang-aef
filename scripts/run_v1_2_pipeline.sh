#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW_ROOT=${RAW_ROOT:-/data/heyuhang/dataset/raw/yajiang}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data/full_npy}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE=${#GPU_LIST[@]}
fi
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
PIPELINE_DIR=${PROJECT_ROOT}/outputs/v1_2_pipeline
PID_FILE=${PIPELINE_DIR}/pipeline.pid
LOCK_FILE=${PIPELINE_DIR}/pipeline.lock
STOP_FILE=${PIPELINE_DIR}/stop
READY_FILE=${DATA_ROOT}/.v1_2_data_ready

mkdir -p "${PIPELINE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another v1.2 pipeline already holds ${LOCK_FILE}" >&2
  exit 1
fi

printf '%s\n' "$$" > "${PID_FILE}"
cleanup() {
  rm -f "${PID_FILE}"
}
trap cleanup EXIT

if [[ -f "${STOP_FILE}" ]]; then
  echo "Stop file exists: ${STOP_FILE}" >&2
  echo "Remove it before starting the pipeline." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hyh-dl

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  echo "[$(timestamp)] $*"
}

stop_requested() {
  [[ -f "${STOP_FILE}" ]]
}

wait_before_retry() {
  local remaining=${RETRY_DELAY_SECONDS}
  while (( remaining > 0 )); do
    stop_requested && return 1
    local interval=5
    (( remaining < interval )) && interval=${remaining}
    sleep "${interval}"
    remaining=$((remaining - interval))
  done
}

checkpoint_epoch_at_least() {
  local checkpoint=$1
  local expected_epoch=$2
  [[ -f "${checkpoint}" ]] || return 1
  python - "${checkpoint}" "${expected_epoch}" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
raise SystemExit(0 if int(checkpoint.get("epoch", 0)) >= int(sys.argv[2]) else 1)
PY
}

prepare_data() {
  if [[ -f "${READY_FILE}" && -s "${DATA_ROOT}/train.jsonl" ]]; then
    log "Prepared data marker found; skipping conversion"
    return 0
  fi

  log "Preparing S2, S1, DEM, and WorldCover arrays"
  python scripts/prepare_full_npy.py \
    --src-root "${RAW_ROOT}" \
    --dst-root "${DATA_ROOT}" \
    --skip-existing || return $?

  log "Preparing Landsat arrays"
  python scripts/prepare_landsat_npy.py \
    --src-root "${RAW_ROOT}/landsat" \
    --dst-root "${DATA_ROOT}" \
    --skip-existing || return $?

  log "Preparing JRC Water arrays"
  python scripts/prepare_jrc_water_npy.py \
    --src-root "${RAW_ROOT}/jrc_water" \
    --dst-root "${DATA_ROOT}" \
    --skip-existing || return $?

  log "Building training manifest"
  python scripts/build_full_manifest.py \
    --data-root "${DATA_ROOT}" \
    --output "${DATA_ROOT}/train.jsonl" || return $?

  local raw_patch_count
  local manifest_count
  raw_patch_count=$(find "${RAW_ROOT}/s2" -mindepth 1 -maxdepth 1 -type d -name 'patch_*' | wc -l)
  manifest_count=$(wc -l < "${DATA_ROOT}/train.jsonl")
  if [[ "${manifest_count}" -ne "${raw_patch_count}" ]]; then
    echo "Manifest count ${manifest_count} does not match raw patch count ${raw_patch_count}" >&2
    return 1
  fi

  printf 'ready_at=%s\npatches=%s\n' "$(timestamp)" "${manifest_count}" > "${READY_FILE}"
  log "Data preparation complete: ${manifest_count} patches"
}

run_training() {
  local label=$1
  local config=$2
  local output_dir=$3
  local expected_epoch=$4
  local fallback_resume=${5:-}
  local master_port=$6
  local final_checkpoint=${output_dir}/checkpoints/final.pt
  local latest_checkpoint=${output_dir}/checkpoints/latest.pt
  local resume_checkpoint=""
  local console_log=${output_dir}/logs/console.log

  if checkpoint_epoch_at_least "${final_checkpoint}" "${expected_epoch}"; then
    log "${label} is already complete"
    return 0
  fi

  if [[ -f "${latest_checkpoint}" ]]; then
    resume_checkpoint=${latest_checkpoint}
  elif [[ -n "${fallback_resume}" ]]; then
    resume_checkpoint=${fallback_resume}
  fi

  if [[ -n "${resume_checkpoint}" && ! -f "${resume_checkpoint}" ]]; then
    echo "Resume checkpoint not found: ${resume_checkpoint}" >&2
    return 1
  fi

  mkdir -p "$(dirname "${console_log}")"
  log "Starting ${label}; resume=${resume_checkpoint:-none}"

  local command=(
    torchrun
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${master_port}"
    scripts/train_with_manifest.py
    --config "${config}"
    --manifest "${DATA_ROOT}/train.jsonl"
    --split train
    --device auto
  )
  if [[ -n "${resume_checkpoint}" ]]; then
    command+=(--resume "${resume_checkpoint}")
  fi

  "${command[@]}" 2>&1 | tee -a "${console_log}"
}

retry_stage() {
  local label=$1
  shift
  local attempt=1

  while ! stop_requested; do
    log "Stage ${label}, attempt ${attempt}"
    if "$@"; then
      log "Stage ${label} completed"
      return 0
    else
      local status=$?
    fi
    log "Stage ${label} failed with status ${status}; retrying in ${RETRY_DELAY_SECONDS}s"
    attempt=$((attempt + 1))
    wait_before_retry || break
  done

  log "Stop requested; leaving stage ${label}"
  return 130
}

BASE_OUTPUT=${PROJECT_ROOT}/outputs/aef_hyh_yajiang_v1_2
CONTINUE_100_OUTPUT=${PROJECT_ROOT}/outputs/aef_hyh_yajiang_v1_2_continue_100
CONTINUE_200_OUTPUT=${PROJECT_ROOT}/outputs/aef_hyh_yajiang_v1_2_continue_200

log "v1.2 pipeline started on GPUs ${CUDA_VISIBLE_DEVICES}"
retry_stage data prepare_data
retry_stage epoch_050 run_training \
  epoch_050 \
  "${PROJECT_ROOT}/configs/yajiang_v1_2.yaml" \
  "${BASE_OUTPUT}" \
  50 \
  "" \
  29612
retry_stage epoch_100 run_training \
  epoch_100 \
  "${PROJECT_ROOT}/configs/yajiang_v1_2_continue_100.yaml" \
  "${CONTINUE_100_OUTPUT}" \
  100 \
  "${BASE_OUTPUT}/checkpoints/best.pt" \
  29613
retry_stage epoch_200 run_training \
  epoch_200 \
  "${PROJECT_ROOT}/configs/yajiang_v1_2_continue_200.yaml" \
  "${CONTINUE_200_OUTPUT}" \
  200 \
  "${CONTINUE_100_OUTPUT}/checkpoints/best.pt" \
  29614

log "v1.2 pipeline completed successfully"
