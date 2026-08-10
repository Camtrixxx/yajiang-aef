#!/usr/bin/env bash
# What does compile_mode buy on 8 cards, at the config that is now the default
# (frames=13 norm=group fast_attention=1 bs=4)?
#
# `default` is re-measured here rather than reused from the earlier sweep, so the
# comparison is within-session and cannot be confounded by machine state.
#
# max-autotune benchmarks many kernel variants per op on every rank, so the first
# step is minutes, not seconds -- hence the large per-arm timeout. Results land in
# /tmp/torchinductor_heyuhang, so a re-run of the same arm is much faster; the
# steady-state step time is unaffected either way.
set -u
cd /data/heyuhang/yajiang-aef

PY=/home/heyuhang/miniconda3/envs/hyh-dl/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
PORT=${PORT:-29671}

run () {  # label mode
  local label=$1 mode=$2
  PORT=$((PORT + 1))
  local t0=$SECONDS
  timeout 5400 $PY -m torch.distributed.run \
    --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/bench_ddp8.py \
    --frames 13 --norm group --batch-size 4 \
    --mode "$mode" --fast-attn 1 --label "$label" \
    2>&1 | grep -E "^RESULT|Error|Traceback|out of memory" || true
  echo "    (wall including compile: $((SECONDS - t0)) s)"
}

echo "== compile_mode on 8 cards, frames=13 norm=group bs=4 =="
run "eager                        " eager
run "default                      " default
run "max-autotune-no-cudagraphs   " max-autotune-no-cudagraphs
run "max-autotune                 " max-autotune
