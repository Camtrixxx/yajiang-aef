#!/usr/bin/env bash
# DDP scaling curve: 1 -> 8 GPUs at fixed per-rank batch size.
#
# Answers "will doubling the card count double the throughput", which is the
# only part of a hardware-swap estimate that can be measured on the hardware we
# already have. The shape matters more than the endpoint: if the DDP overhead
# (step_N - step_1) is flat as N grows, that is the ring all-reduce signature
# and the curve extrapolates past 8. If it grows with N, it does not.
#
# Steps only (--skip-epoch): the epoch pass would fold in the input pipeline,
# whose per-rank shard shrinks as N grows -- that confounds the comm question.
#
# Throughput is derived, not measured: samples_per_s = N * batch / step. Peak
# memory is reported to confirm it does not grow with world size (it must not;
# if it does, the per-card memory budget depends on cluster size).
set -u

PY=/home/heyuhang/miniconda3/envs/hyh-dl/bin/python
ARM=${ARM:-current}
BATCH=${BATCH:-4}          # per rank, must match cfg.data.batch_size
NLIST=${NLIST:-"1 2 4 8"}
PORT=${PORT:-29701}
LOG=${LOG:-outputs/bench_ddp_scaling.log}

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$(dirname "$LOG")"

{
  echo "# $(date '+%F %T')  DDP scaling, arm=${ARM}, bs=${BATCH}/rank, step only"
  echo "gpus  step_ms  spread  peak_GB  samples_per_s"
} | tee -a "$LOG"

for N in $NLIST; do
  # Take the first N devices. Fine here because the topology is a full NV8 mesh
  # (nvidia-smi topo -m) -- on a partially connected box the subset would matter.
  GP=$(seq -s, 0 $((N - 1)))

  out=$(CUDA_VISIBLE_DEVICES=$GP PYTHONPATH=.:scripts \
    "$PY" -m torch.distributed.run --nproc_per_node="$N" \
      --master_port="$PORT" \
      scripts/bench_naive_ddp.py --arm "$ARM" --skip-epoch 2>&1)

  line=$(echo "$out" | grep -E '^RESULT' | head -1)
  if [ -z "$line" ]; then
    { echo "$N: FAILED"; echo "$out" | tail -10 | sed 's/^/    /'; } | tee -a "$LOG"
    PORT=$((PORT + 1))
    continue
  fi
  echo "$line" | awk -F'\t' -v n="$N" -v b="$BATCH" \
    '{printf "%-5s %8s %7s %8s %14.1f\n", n, $3, $4, $5, n*b/($3/1000)}' \
    | tee -a "$LOG"

  PORT=$((PORT + 1))
done
