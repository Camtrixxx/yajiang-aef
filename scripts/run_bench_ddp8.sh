#!/usr/bin/env bash
# Drive scripts/bench_ddp8.py once per arm. Each arm gets a fresh torchrun so
# compile state from one arm cannot leak into the next.
set -u
cd /data/heyuhang/yajiang-aef

PY=/home/heyuhang/miniconda3/envs/hyh-dl/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
MODE=${MODE:-default}
PORT=${PORT:-29631}

run () {  # label frames norm bs mode fast
  local label=$1 frames=$2 norm=$3 bs=$4 mode=$5 fast=$6
  PORT=$((PORT + 1))
  timeout 1800 $PY -m torch.distributed.run \
    --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/bench_ddp8.py \
    --frames "$frames" --norm "$norm" --batch-size "$bs" \
    --mode "$mode" --fast-attn "$fast" --label "$label" \
    2>&1 | grep -E "^RESULT|Error|Traceback|out of memory" || true
}

echo "== A. does group+13 hold up on ${NPROC} cards? (bs=4) =="
run "eager baseline                          " 16 batch 4 eager 0
run "compile+fast  frames=16 norm=batch      " 16 batch 4 "$MODE" 1
run "compile+fast  frames=16 norm=group      " 16 group 4 "$MODE" 1
run "compile+fast  frames=13 norm=group      " 13 group 4 "$MODE" 1

echo
echo "== B. spend the spare memory on batch_size (frames=13 norm=group) =="
for bs in 8 12 16; do
  run "compile+fast  frames=13 norm=group bs=${bs}" 13 group "$bs" "$MODE" 1
done
