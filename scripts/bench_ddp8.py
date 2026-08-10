"""8-card DDP benchmark: does group+13 hold up, and can spare memory buy speed?

Two questions, one run per arm (launch with torchrun --nproc_per_node=8):

  1. Does the single-card group+13 win (1.139x) survive DDP, where all-reduce
     is not compiled and its share of the step rises?
  2. Memory is not the constraint here (24.45 of 81.92 GB used at bs=4), speed
     is. Raising batch_size cuts the number of all-reduce rounds per epoch,
     so it should convert idle memory into throughput.

Reported per arm: steady step ms (max across ranks, so stragglers count),
patches/s, peak GB, and projected epoch seconds. The step time uses a fixed
pre-fetched batch so it measures compute+comm only; the epoch projection uses
the real DataLoader so it includes input pipeline cost.

NOTE ON batch_size: unlike compile / fast_attention / (group + max_frames=13),
batch_size is NOT a pure performance knob. It changes gradient noise and the
effective learning rate, so a larger batch alters the training trajectory and
needs an LR revisit. It is measured here as a speed option, not endorsed as a
drop-in default.
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


def ns(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(v) for v in d]
    return d


def move(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: move(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move(v, dev) for v in x)
    return x


def fwd_kwargs(batch):
    return dict(
        source_frames=batch["source_frames"],
        source_timestamps_ms=batch["source_timestamps_ms"],
        source_frame_mask=batch["source_frame_mask"],
        source_input_mask=batch["source_input_mask"],
        source_type_ids=batch["source_type_ids"],
        valid_start_ms=batch["valid_start_ms"],
        valid_end_ms=batch["valid_end_ms"],
        target_relative_time=batch["target_relative_time"],
        target_metadata=batch["target_metadata"],
    )


def build_loader(cfg, manifest, rank, world):
    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=bool(cfg.data.persistent_workers),
        prefetch_factor=int(cfg.data.prefetch_factor),
        collate_fn=aef_collate_fn,
        drop_last=True,
    )
    return ds, loader


def allreduce_max(value, dev):
    t = torch.tensor([value], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--norm", default="batch", choices=["batch", "group"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mode", default="default", help="compile mode, or 'eager'")
    p.add_argument("--fast-attn", type=int, default=1)
    p.add_argument("--workers", type=int, default=-1,
                   help="override data.num_workers; -1 keeps the config value")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--label", default="")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    cfg = ns(yaml.safe_load(open(args.config)))
    cfg.data.max_frames = args.frames
    cfg.data.batch_size = args.batch_size
    cfg.model.stem_norm = args.norm
    cfg.model.fast_attention = bool(args.fast_attn)
    if args.workers >= 0:
        cfg.data.num_workers = args.workers

    ds, loader = build_loader(cfg, args.manifest, rank, world)
    model = AEFModel(cfg).to(dev)
    model = DDP(model, device_ids=[local],
                find_unused_parameters=bool(cfg.training.find_unused_parameters))
    if args.mode != "eager":
        model = torch.compile(model, mode=args.mode)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr))

    # Fixed batch: isolates compute+comm from the input pipeline.
    loader.sampler.set_epoch(0)
    batch = move(next(iter(loader)), dev)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**fwd_kwargs(batch))
            loss = compute_total_loss(out, batch, cfg)
        loss.total.backward()
        opt.step()
        return float(loss.total.detach())

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    step_ms = allreduce_max(statistics.median(times), dev)
    spread = allreduce_max(max(times) - min(times), dev)
    peak = allreduce_max(torch.cuda.max_memory_allocated() / 1e9, dev)

    # Real epoch, attributed. The step timing above uses a batch already resident
    # on the GPU, so it excludes two things the epoch pays: waiting for the loader
    # to hand over a batch, and the H2D copy. Measure both rather than infer them.
    #
    # Caveat: the syncs below serialize work that would otherwise overlap, so
    # wait+h2d+compute sums to MORE than the plain epoch time. The plain number is
    # reported alongside so the perturbation is visible instead of hidden.
    steps_per_epoch = len(loader)
    dist.barrier()
    t_wait = t_h2d = t_comp = 0.0
    t0 = time.perf_counter()
    it = iter(loader)
    while True:
        tw = time.perf_counter()
        try:
            real = next(it)
        except StopIteration:
            break
        t_wait += time.perf_counter() - tw

        th = time.perf_counter()
        real = move(real, dev)
        torch.cuda.synchronize()
        t_h2d += time.perf_counter() - th

        tc = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**fwd_kwargs(real))
            loss = compute_total_loss(out, real, cfg)
        loss.total.backward()
        opt.step()
        torch.cuda.synchronize()
        t_comp += time.perf_counter() - tc
    epoch_s = allreduce_max(time.perf_counter() - t0, dev)
    t_wait = allreduce_max(t_wait, dev)
    t_h2d = allreduce_max(t_h2d, dev)
    t_comp = allreduce_max(t_comp, dev)

    if rank == 0:
        global_bs = args.batch_size * world
        thru = global_bs / (step_ms / 1e3)
        label = args.label or (f"frames={args.frames} norm={args.norm} "
                               f"bs={args.batch_size} mode={args.mode} "
                               f"fast={args.fast_attn}")
        print(f"RESULT\t{label}\t{step_ms:.1f}\t{spread:.1f}\t{peak:.2f}\t"
              f"{thru:.1f}\t{steps_per_epoch}\t{epoch_s:.1f}", flush=True)
        print(f"ATTRIB\t{label}\tloader_wait={t_wait:.2f}s\th2d={t_h2d:.2f}s\t"
              f"compute={t_comp:.2f}s\tsum={t_wait + t_h2d + t_comp:.2f}s\t"
              f"plain_epoch={epoch_s:.2f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
