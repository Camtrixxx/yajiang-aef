#!/usr/bin/env bash
# Naive DDP vs current: what would a plain torchrun script cost?
#
# One fresh torchrun per arm -- compile state and cudnn.benchmark heuristics
# leak across runs within a process, which would contaminate later arms.
#
# The naive arm runs fp32 at max_frames=16 and may OOM; that is reported as a
# result rather than treated as a script failure.
set -u

PY=/home/heyuhang/miniconda3/envs/hyh-dl/bin/python
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PORT=${PORT:-29655}
ARMS=${ARMS:-"naive +tf32 +amp current"}

cd "$(dirname "$0")/.." || exit 1

echo "arm         step_ms  spread   peak_GB  epoch_s  200ep_hours"
echo "--------------------------------------------------------------"

for arm in $ARMS; do
  # Explicit interpreter: a bare `torchrun` can resolve to the system Python,
  # whose onnx/protobuf conflict makes torch._dynamo blow up on import.
  out=$(CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=.:scripts \
    "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" \
      --master_port="$PORT" \
      scripts/bench_naive_ddp.py --arm "$arm" 2>&1)

  line=$(echo "$out" | grep -E '^RESULT' | head -1)
  if [ -z "$line" ]; then
    echo "$arm: FAILED"
    echo "$out" | tail -15 | sed 's/^/    /'
    continue
  fi
  echo "$line" | awk -F'\t' \
    '{printf "%-11s %8s %7s %9s %8s %11s\n", $2, $3, $4, $5, $6, $7}'
  echo "$out" | grep -E '^NOTE' | sed 's/^/    /'

  PORT=$((PORT + 1))
done
