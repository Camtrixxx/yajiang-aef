"""How long does a 200-epoch run take with NO optimizations at all?

The 606.4 ms "baseline" in docs/experiments/v1.2_training_acceleration.md is not
actually a naive run -- it already carries bf16 AMP, TF32 and cudnn.benchmark,
because those were enabled long before the acceleration work started. So it
answers "what did the acceleration work buy" but not "what would a plain
torchrun + DDP script cost".

This measures the layers separately, cheapest change first:

  naive     fp32, no TF32, no cudnn.benchmark, nn.MultiheadAttention,
            BatchNorm stem, max_frames=16, num_workers=8, no compile
  +tf32     add TF32 matmul + cudnn.allow_tf32 + cudnn.benchmark  (3 lines)
  +amp      add bf16 autocast                                     (2 lines)
            ^ this arm is the 606.4 ms figure the doc calls "baseline"
  current   everything from the acceleration work

Each arm is a separate torchrun (compile state and cudnn heuristics leak across
runs otherwise). Reports steady step ms, peak GB, plain epoch s, and the
projected 200-epoch wall clock, which is the number a report actually wants.

fp32 at max_frames=16 may OOM at 81.92 GB/card -- bf16 autocast is what makes
the activation footprint fit. An OOM is a result, not a failure: it means the
naive arm cannot run this config at all, so it is reported as such.

Usage:
  bash scripts/run_bench_naive_ddp.sh
"""
import argparse
import os
import statistics
import time
import types

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.losses import compute_total_loss

# Reuse the helpers rather than re-deriving the forward signature, which is easy
# to get subtly wrong.
from bench_ddp8 import allreduce_max, fwd_kwargs, move, ns

# Arm definitions. Each entry says which optimizations are ON.
ARMS = {
    "naive":   dict(tf32=False, amp=False, fast_attn=False, norm="batch",
                    frames=16, workers=8, compile_mode="eager"),
    "+tf32":   dict(tf32=True,  amp=False, fast_attn=False, norm="batch",
                    frames=16, workers=8, compile_mode="eager"),
    "+amp":    dict(tf32=True,  amp=True,  fast_attn=False, norm="batch",
                    frames=16, workers=8, compile_mode="eager"),
    "current": dict(tf32=True,  amp=True,  fast_attn=True,  norm="group",
                    frames=13, workers=2,
                    compile_mode="max-autotune-no-cudagraphs"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--epochs-projected", type=int, default=200)
    p.add_argument("--skip-epoch", action="store_true",
                   help="step timing only; skips the full-epoch pass")
    args = p.parse_args()

    arm = ARMS[args.arm]
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")

    # The whole point of the naive arm: these are OFF unless the arm turns them on.
    torch.backends.cuda.matmul.allow_tf32 = arm["tf32"]
    torch.backends.cudnn.allow_tf32 = arm["tf32"]
    torch.backends.cudnn.benchmark = arm["tf32"]
    if not arm["tf32"]:
        # PyTorch >= 2.0 defaults this to "highest" already, but set it explicitly
        # so the arm does not silently inherit a global default from elsewhere.
        torch.set_float32_matmul_precision("highest")
    else:
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    cfg = ns(yaml.safe_load(open(args.config)))
    cfg.data.max_frames = arm["frames"]
    cfg.data.num_workers = arm["workers"]
    cfg.model.stem_norm = arm["norm"]
    cfg.model.fast_attention = arm["fast_attn"]

    ds = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.data.batch_size),
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=bool(cfg.data.persistent_workers),
        prefetch_factor=int(cfg.data.prefetch_factor),
        collate_fn=aef_collate_fn,
        drop_last=True,
    )

    model = AEFModel(cfg).to(dev)
    model = DDP(model, device_ids=[local],
                find_unused_parameters=bool(cfg.training.find_unused_parameters))
    if arm["compile_mode"] != "eager":
        model = torch.compile(model, mode=arm["compile_mode"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr))

    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if arm["amp"] \
        else torch.enable_grad

    loader.sampler.set_epoch(0)
    batch = move(next(iter(loader)), dev)

    def one_step(b):
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            out = model(**fwd_kwargs(b))
            loss = compute_total_loss(out, b, cfg)
        loss.total.backward()
        opt.step()

    oom = False
    try:
        for _ in range(args.warmup):
            one_step(batch)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        oom = True

    # An OOM on one rank must be agreed on by all ranks, or the survivors hang
    # forever in the next collective.
    flag = torch.tensor([1.0 if oom else 0.0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item() > 0:
        if rank == 0:
            print(f"RESULT\t{args.arm}\tOOM\tOOM\tOOM\tOOM\tOOM", flush=True)
            print(f"NOTE\t{args.arm}\tout of memory at max_frames="
                  f"{arm['frames']} without bf16 autocast", flush=True)
        dist.destroy_process_group()
        return

    dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        one_step(batch)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    step_ms = allreduce_max(statistics.median(times), dev)
    spread = allreduce_max(max(times) - min(times), dev)
    peak = allreduce_max(torch.cuda.max_memory_allocated() / 1e9, dev)

    # Plain epoch: no per-stage syncs, so nothing is serialized that would
    # normally overlap. This is the number that scales to a real run.
    epoch_s = float("nan")
    if not args.skip_epoch:
        dist.barrier()
        t0 = time.perf_counter()
        for real in loader:
            one_step(move(real, dev))
        torch.cuda.synchronize()
        epoch_s = allreduce_max(time.perf_counter() - t0, dev)

    if rank == 0:
        hours = epoch_s * args.epochs_projected / 3600.0
        print(f"RESULT\t{args.arm}\t{step_ms:.1f}\t{spread:.1f}\t{peak:.2f}\t"
              f"{epoch_s:.1f}\t{hours:.2f}", flush=True)
        print(f"CFG\t{args.arm}\ttf32={arm['tf32']} amp={arm['amp']} "
              f"fast_attn={arm['fast_attn']} norm={arm['norm']} "
              f"frames={arm['frames']} workers={arm['workers']} "
              f"compile={arm['compile_mode']}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
