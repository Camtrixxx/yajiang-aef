"""Does CPU contention explain the loader wait?

Two measurements disagreed:
  isolated, 1 rank x 8 workers, idle box : 3.85 s for one rank's share
  inside real 8-rank training            : 8.08 s blocked in next()

Hypothesis: the isolated test gave one rank all 8 workers on 128 idle logical
cores. Real training runs 8 ranks x 8 workers = 64 workers plus 8 mains on 64
physical cores, so each worker gets roughly one hyperthread rather than a core.

Test: run the same shard-sized loader in N concurrent processes and watch
per-process time as N goes 1 -> 8. No GPU involved. If the 1-rank number holds
at N=8, contention is not the explanation and something else is.
"""
import argparse
import multiprocessing as mp
import time
import types

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import YajiangAEFDataset, aef_collate_fn


def ns(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(v) for v in d]
    return d


def worker(rank, world, cfg_path, manifest, nw, out):
    cfg = ns(yaml.safe_load(open(cfg_path)))
    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    per_rank = len(ds) // world
    idx = list(range(rank * per_rank, (rank + 1) * per_rank))
    loader = DataLoader(
        Subset(ds, idx),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=int(cfg.data.prefetch_factor),
        collate_fn=aef_collate_fn,
        drop_last=True,
    )
    it = iter(loader)
    next(it)                      # worker startup outside the clock
    del it
    t0 = time.perf_counter()
    n = 0
    for _ in loader:
        n += 1
    out.put((rank, n, time.perf_counter() - t0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--world", type=int, default=8,
                   help="shard count, kept fixed so each process does the same work")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--concurrency", default="1,2,4,8",
                   help="comma-separated process counts to test")
    args = p.parse_args()
    levels = [int(x) for x in args.concurrency.split(",")]

    print("=" * 76)
    print(f"loader time per process vs concurrency "
          f"(shard = 1/{args.world} of data, num_workers={args.workers})")
    print("=" * 76)
    print(f"{'concurrent':>10s} {'total workers':>14s} {'slowest s':>10s} "
          f"{'fastest s':>10s} {'vs N=1':>8s}")
    solo = None
    for n_proc in levels:
        q = mp.Queue()
        procs = [
            mp.Process(target=worker,
                       args=(r, args.world, args.config, args.manifest,
                             args.workers, q))
            for r in range(n_proc)
        ]
        for pr in procs:
            pr.start()
        res = [q.get() for _ in range(n_proc)]
        for pr in procs:
            pr.join()
        times = [t for _, _, t in res]
        if solo is None:
            solo = max(times)
        print(f"{n_proc:>10d} {n_proc * args.workers:>14d} {max(times):>10.2f} "
              f"{min(times):>10.2f} {max(times) / solo:>7.2f}x")
    print()
    print("Each process loads the same number of batches at every concurrency, so")
    print("a rising 'slowest' column is contention, not more work.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
