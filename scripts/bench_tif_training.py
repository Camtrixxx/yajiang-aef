"""What would training cost if we had never converted the tif data to npy?

bench_naive_ddp.py answers "what does a plain DDP script cost" but still reads
npy. This answers the other half: plain DDP script AND raw GeoTIFF input, i.e.
the true starting point before any work was done on this project.

Arms are (training config) x (data format), identical in every respect except
where the bytes come from:

  naive     fp32, no TF32, no cudnn.benchmark, no AMP, no compile,
            nn.MultiheadAttention, BatchNorm stem, max_frames=16, workers=8
  current   everything from the acceleration work: TF32, bf16 autocast,
            max-autotune-no-cudagraphs compile, fast_attention, GroupNorm stem,
            max_frames=13, workers=2

The `current` arms are the interesting ones. Under `naive` the loader hides
behind a 2.7 s/step compute, so the format barely shows up (152.9 vs 148.9 s
epoch). Once compute drops to ~350 ms/step there is far less to hide behind,
and the question is whether tif crosses over into being the bottleneck. Since
workers=2 was tuned for warm-page-cache npy, `current` also gets a workers=8
tif arm: if tif is loader-bound, more workers should buy back wall clock, and
if it is not, they should change nothing.

WHAT THE TIF ARM HAS TO PAY THAT THE NPY ARM DOES NOT
-----------------------------------------------------
The npy files are not a raw dump of the tif -- scripts/prepare_full_npy.py
baked preprocessing into them:

  inputs      astype(float32)
  dem         z-score with the dataset-wide mean/std
  worldcover  class remap {10:0, 30:1, ... 20:8} -> uint8
  jrc_water   nodata -128 -> 255

Reading tif at training time means paying that per sample, every epoch. Skipping
it would not just be unfair, it would crash: raw worldcover values (10..100) fed
to cross_entropy with num_classes=9 is an out-of-range index. So the tif arm
replicates all of it, using a 256-entry LUT for the categorical remaps -- the
fast way, i.e. the benefit of the doubt.

The tif arm measures cost, not correctness: DEM stats come from the recorded
preprocess_meta.json rather than being recomputed, since a real tif-native
pipeline would cache them too.

MEASURED PER ARM
----------------
  step        fixed pre-fetched batch: compute+comm only, no input pipeline.
              Should be IDENTICAL across formats WITHIN a training arm -- that is
              the control. If it moves, something other than the loader changed
              and the comparison is void. (It legitimately differs BETWEEN naive
              and current -- that is the acceleration work.)
  epoch       plain pass over the real DataLoader, no extra syncs.
  loader      pass that only pulls batches and drops them: the input side with
              compute removed, showing how much cost exists before overlap.
  200 epoch   the projection a report actually wants.

The gap between `loader` and how much `epoch` actually grows is the point: slow
input only costs wall clock where it fails to hide behind compute.

Usage:
  bash scripts/run_bench_tif_training.sh
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.losses import compute_total_loss

from bench_ddp8 import allreduce_max, fwd_kwargs, move, ns
from bench_naive_ddp import ARMS

RAW_ROOT = Path("/data/heyuhang/dataset/raw/yajiang")

# From scripts/prepare_full_npy.py -- kept in sync by hand; a mismatch only
# shifts values, not timings, but a wrong worldcover map would crash the loss.
WORLDCOVER_MAP = {10: 0, 30: 1, 40: 2, 50: 3, 60: 4, 70: 5, 80: 6, 100: 7, 20: 8}
WORLDCOVER_IGNORE = 255
JRC_NODATA = -128
JRC_IGNORE = 255


def _worldcover_lut() -> np.ndarray:
    lut = np.full(256, WORLDCOVER_IGNORE, dtype=np.uint8)
    for raw, cls in WORLDCOVER_MAP.items():
        lut[raw] = cls
    return lut


class TifDataset(YajiangAEFDataset):
    """Same dataset, but every path is redirected to its GeoTIFF original.

    Path mapping (inverse of prepare_full_npy.py / prepare_landsat_npy.py):
      full_npy/<patch>/inputs/<src>/<stem>.npy -> raw/<src>/<patch>/<stem>.tif
      full_npy/<patch>/targets/<name>.npy      -> raw/<name>/<patch>/static.tif
    """

    def __init__(self, *a, dem_mean: float, dem_std: float, **kw):
        super().__init__(*a, **kw)
        self.dem_mean = dem_mean
        self.dem_std = dem_std
        self.wc_lut = _worldcover_lut()

    @staticmethod
    def _to_tif(npy_path: Path) -> tuple[Path, str]:
        """Return (tif path, kind) where kind drives the preprocessing branch."""
        parts = npy_path.parts
        stem = npy_path.stem
        if "inputs" in parts:
            i = parts.index("inputs")
            patch, src = parts[i - 1], parts[i + 1]
            return RAW_ROOT / src / patch / f"{stem}.tif", "input"
        if "targets" in parts:
            i = parts.index("targets")
            patch = parts[i - 1]
            return RAW_ROOT / stem / patch / "static.tif", stem
        raise ValueError(f"Cannot map to tif: {npy_path}")

    def _load_array(self, path):
        import rasterio  # worker-local: GDAL state must not be inherited by fork

        tif_path, kind = self._to_tif(Path(path))
        with rasterio.open(tif_path) as src:
            arr = src.read()
        if arr.shape[0] == 1:
            arr = arr[0]

        if kind == "input":
            arr = arr.astype(np.float32)
        elif kind == "dem":
            arr = ((arr.astype(np.float32) - self.dem_mean) / self.dem_std).astype(np.float32)
        elif kind == "worldcover":
            arr = self.wc_lut[arr.astype(np.uint8)]
        elif kind == "jrc_water":
            out = np.full(arr.shape, JRC_IGNORE, dtype=np.uint8)
            valid = arr != JRC_NODATA
            out[valid] = arr[valid].astype(np.uint8)
            arr = out
        else:
            raise ValueError(f"Unknown target kind '{kind}' for {tif_path}")

        return torch.from_numpy(np.ascontiguousarray(arr))


def build(cfg, manifest, rank, world, fmt):
    if fmt == "npy":
        ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    else:
        meta = json.loads((Path(manifest).parent / "preprocess_meta.json").read_text())
        ds = TifDataset(
            cfg=cfg, manifest_path=manifest, split="train",
            dem_mean=float(meta["dem"]["mean"]), dem_std=float(meta["dem"]["std"]),
        )
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
    return ds, loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--format", required=True, choices=["npy", "tif"])
    p.add_argument("--arm", default="naive", choices=list(ARMS))
    p.add_argument("--workers", type=int, default=None,
                   help="override the arm's num_workers (per rank)")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--epochs-projected", type=int, default=200)
    args = p.parse_args()

    arm = ARMS[args.arm]
    workers = arm["workers"] if args.workers is None else args.workers

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")

    # Backend switches follow the arm, not this script -- same table as
    # bench_naive_ddp.py so the two benchmarks stay comparable.
    torch.backends.cuda.matmul.allow_tf32 = arm["tf32"]
    torch.backends.cudnn.allow_tf32 = arm["tf32"]
    torch.backends.cudnn.benchmark = arm["tf32"]
    torch.set_float32_matmul_precision("high" if arm["tf32"] else "highest")
    torch.manual_seed(42)

    cfg = ns(yaml.safe_load(open(args.config)))
    cfg.data.max_frames = arm["frames"]
    cfg.data.num_workers = workers
    cfg.model.stem_norm = arm["norm"]
    cfg.model.fast_attention = arm["fast_attn"]

    ds, loader = build(cfg, args.manifest, rank, world, args.format)
    model = AEFModel(cfg).to(dev)
    model = DDP(model, device_ids=[local],
                find_unused_parameters=bool(cfg.training.find_unused_parameters))
    if arm["compile_mode"] != "eager":
        # Outside DDP, so DDPOptimizer keeps comm/compute overlap.
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

    for _ in range(args.warmup):
        one_step(batch)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    # Control: fixed resident batch, so this must not differ between arms.
    times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        one_step(batch)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    step_ms = allreduce_max(statistics.median(times), dev)
    peak = allreduce_max(torch.cuda.max_memory_allocated() / 1e9, dev)

    # Real epoch, no extra syncs: nothing that would normally overlap is
    # serialized, so this is the number that scales to a real run.
    dist.barrier()
    t0 = time.perf_counter()
    for real in loader:
        one_step(move(real, dev))
    torch.cuda.synchronize()
    epoch_s = allreduce_max(time.perf_counter() - t0, dev)

    # Input side alone: pull every batch and drop it. Compute removed, so this
    # is the cost that has to hide behind compute -- not the cost that is paid.
    dist.barrier()
    t0 = time.perf_counter()
    for _ in loader:
        pass
    loader_s = allreduce_max(time.perf_counter() - t0, dev)

    if rank == 0:
        hours = epoch_s * args.epochs_projected / 3600.0
        label = f"{args.arm}/{args.format}/nw{workers}"
        # compute_s is what the epoch would cost if the input side were free.
        compute_s = step_ms * len(loader) / 1e3
        print(f"RESULT\t{label}\t{step_ms:.1f}\t{peak:.2f}\t{epoch_s:.1f}\t"
              f"{loader_s:.1f}\t{compute_s:.1f}\t{hours:.2f}", flush=True)
        print(f"CFG\t{label}\ttf32={arm['tf32']} amp={arm['amp']} "
              f"fast_attn={arm['fast_attn']} norm={arm['norm']} "
              f"frames={arm['frames']} workers={workers} "
              f"compile={arm['compile_mode']} steps_per_epoch={len(loader)}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
