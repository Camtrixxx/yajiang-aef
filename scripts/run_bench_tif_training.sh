#!/usr/bin/env bash
# Raw tif vs npy as training input, under both the naive and the current config.
#
# The naive pair answers "what did the format conversion buy at the starting
# point" (answer: almost nothing -- it hides behind 2.7 s/step compute). The
# current pair answers the question that actually matters now, once compute is
# 8x faster and there is far less to hide behind.
#
# Arms are "trainingArm:format:workers". Workers is explicit rather than
# inherited so the current/tif arm can be retried with more workers: if tif is
# loader-bound, more workers buy wall clock back; if not, they change nothing.
#
# One fresh torchrun per arm. Output is tee'd to a log -- the naive_ddp bench
# only printed to stdout and the numbers had to be recovered from a session
# transcript afterwards.
set -u

PY=/home/heyuhang/miniconda3/envs/hyh-dl/bin/python
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PORT=${PORT:-29671}
ARMS=${ARMS:-"naive:npy:8 naive:tif:8 current:npy:2 current:tif:2 current:tif:8"}
LOG=${LOG:-outputs/bench_tif_training.log}

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$(dirname "$LOG")"

{
  echo "# $(date '+%F %T')  ${NPROC} GPU"
  echo "arm                 step_ms   peak_GB   epoch_s  loader_s compute_s  200ep_h"
  echo "---------------------------------------------------------------------------"
} | tee -a "$LOG"

for spec in $ARMS; do
  IFS=: read -r arm fmt workers <<<"$spec"

  # current arms compile with max-autotune-no-cudagraphs: ~590 s on the first
  # step before anything is measured. Expected, not a hang.
  out=$(CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=.:scripts \
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
      --master_port="$PORT" \
      scripts/bench_tif_training.py --arm "$arm" --format "$fmt" \
        --workers "$workers" 2>&1)

  line=$(echo "$out" | grep -E '^RESULT' | head -1)
  if [ -z "$line" ]; then
    { echo "$spec: FAILED"; echo "$out" | tail -20 | sed 's/^/    /'; } | tee -a "$LOG"
    PORT=$((PORT + 1))
    continue
  fi
  echo "$line" | awk -F'\t' \
    '{printf "%-18s %9s %9s %9s %9s %9s %8s\n", $2, $3, $4, $5, $6, $7, $8}' \
    | tee -a "$LOG"
  echo "$out" | grep -E '^CFG' | sed 's/^/    /' | tee -a "$LOG"

  PORT=$((PORT + 1))
done
