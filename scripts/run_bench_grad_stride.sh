#!/usr/bin/env bash
# Drive scripts/bench_grad_stride.py once per arm. A fresh torchrun per arm is
# required here, not just tidy: TORCH_WARN_ONCE means the reducer's warning
# fires once per process, so reusing a process would hide it in later arms.
set -u
cd /data/heyuhang/yajiang-aef

PY=/home/heyuhang/miniconda3/envs/hyh-dl/bin/python
export PYTHONPATH=.:scripts
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
MODE=${MODE:-max-autotune-no-cudagraphs}
PORT=${PORT:-29661}

for arm in none bucket_view channels_last both; do
  PORT=$((PORT + 1))
  echo "== arm: ${arm} =="
  timeout 2400 $PY -m torch.distributed.run \
    --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/bench_grad_stride.py --arm "$arm" --mode "$MODE" \
    2>&1 | grep -E "^RESULT|^  MISMATCH|Error|Traceback|out of memory" || true
done
