"""Where is the input-pipeline floor, and what is it made of?

Compute is now 355 ms/step on 8 cards while the epoch spends 30% not computing,
and that non-compute share GREW as compute got faster (17% -> 22% -> 30% across
eager -> default -> max-autotune). That is the signature of waiting on data, not
of a fixed overhead. So: measure the loader with no model at all.

Three things, in order:

  1. Loader ceiling -- iterate the real DataLoader, no forward/backward. Gives the
     epoch time the input pipeline alone can support. If that is near the observed
     26.9 s, compute optimizations are done paying.
  2. Per-item cost breakdown -- time __getitem__ in one process and split it into
     file reads vs the CPU transform chain, so the fix targets the right half.
  3. Candidate fixes, measured rather than assumed:
       a. cache the normalize constants instead of rebuilding them 39x per item
       b. batch the per-frame resize into one interpolate call
       c. read with np.load(mmap_mode='r')
     Each is timed against the current code on identical records.

Run WITHOUT a concurrent GPU job: the numbers here are CPU- and IO-bound and 64
dataloader workers from a training run would distort every one of them.
"""
import argparse
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import YajiangAEFDataset, aef_collate_fn


def ns(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(v) for v in d]
    return d


def section_ceiling(cfg, manifest, world):
    """Epoch time the loader alone can sustain, at the real worker count.

    One rank's share is simulated by iterating 1/world of the dataset, which is
    what each of the 8 processes actually does.
    """
    print("=" * 76)
    print("1. loader ceiling (no model)")
    print("=" * 76)
    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    per_rank = len(ds) // world
    subset = torch.utils.data.Subset(ds, list(range(per_rank)))
    for nw in (8, 16):
        loader = DataLoader(
            subset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=int(cfg.data.prefetch_factor),
            collate_fn=aef_collate_fn,
            drop_last=True,
        )
        it = iter(loader)          # pay worker startup outside the clock
        next(it)
        del it
        t0 = time.perf_counter()
        n = 0
        for _ in loader:
            n += 1
        dt = time.perf_counter() - t0
        print(f"  num_workers={nw:2d}  {n} batches in {dt:6.2f} s  "
              f"-> {dt / n * 1e3:6.1f} ms/batch  "
              f"({per_rank / dt:6.1f} patches/s per rank)")
        print(f"                 loader-only epoch floor for one rank: {dt:5.2f} s")
        del loader
    return ds


def section_breakdown(ds, n_items=24):
    """Split one __getitem__ into file reads vs CPU transforms."""
    print()
    print("=" * 76)
    print("2. per-item cost: file reads vs CPU transforms")
    print("=" * 76)
    whole, read_only = [], []
    for i in range(n_items):
        rec = ds.records[i]
        t0 = time.perf_counter()
        ds[i]
        whole.append((time.perf_counter() - t0) * 1e3)

        paths = [f["path"] for v in rec.inputs.values() for f in v.get("frames", [])]
        t0 = time.perf_counter()
        for p in paths:
            np.load(p)
        read_only.append((time.perf_counter() - t0) * 1e3)

    w, r = statistics.median(whole), statistics.median(read_only)
    n_files = len(paths)
    print(f"  files per item          : {n_files}")
    print(f"  full __getitem__        : {w:7.2f} ms")
    print(f"  np.load only            : {r:7.2f} ms  ({r / w * 100:4.1f}%)")
    print(f"  CPU transform chain     : {w - r:7.2f} ms  ({(w - r) / w * 100:4.1f}%)")
    print(f"  per file                : {r / n_files:7.3f} ms read")
    return w, r


def _paths_of(ds, i):
    rec = ds.records[i]
    return [f["path"] for v in rec.inputs.values() for f in v.get("frames", [])]


def section_candidates(ds, n_items=24):
    """Measure the three candidate fixes on identical records."""
    print()
    print("=" * 76)
    print("3. candidate fixes, measured")
    print("=" * 76)

    def timeit(fn):
        ts = []
        for i in range(n_items):
            t0 = time.perf_counter()
            fn(i)
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    base = timeit(lambda i: [np.load(p) for p in _paths_of(ds, i)])
    mmap = timeit(lambda i: [np.load(p, mmap_mode="r") for p in _paths_of(ds, i)])
    # mmap is lazy; force the pages so the comparison is honest
    mmap_touch = timeit(
        lambda i: [np.asarray(np.load(p, mmap_mode="r")) for p in _paths_of(ds, i)]
    )
    print(f"  (c) np.load            : {base:7.2f} ms")
    print(f"      np.load mmap lazy  : {mmap:7.2f} ms  <- not comparable, pages unread")
    print(f"      np.load mmap forced: {mmap_touch:7.2f} ms  "
          f"({base / mmap_touch:.2f}x vs np.load)")

    # (a) normalize constants: rebuilt per frame today
    x = torch.randn(6, 128, 128)
    reps = 39
    t0 = time.perf_counter()
    for _ in range(reps):
        m = torch.as_tensor(4934.0, dtype=x.dtype).view(-1, 1, 1)
        s = torch.as_tensor(1315.0, dtype=x.dtype).view(-1, 1, 1).clamp_min(1e-6)
        _ = (x.clamp(0.0, 10000.0) - m) / s
    rebuilt = (time.perf_counter() - t0) * 1e3
    m = torch.as_tensor(4934.0, dtype=x.dtype).view(-1, 1, 1)
    s = torch.as_tensor(1315.0, dtype=x.dtype).view(-1, 1, 1).clamp_min(1e-6)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = (x.clamp(0.0, 10000.0) - m) / s
    cached = (time.perf_counter() - t0) * 1e3
    print(f"  (a) normalize x{reps}     : {rebuilt:7.2f} ms rebuilt vs "
          f"{cached:7.2f} ms cached  (saves {rebuilt - cached:.2f} ms/item)")

    # (b) per-frame resize vs one batched interpolate
    frames = torch.randn(13, 6, 100, 100)
    t0 = time.perf_counter()
    for f in frames:
        torch.nn.functional.interpolate(f.unsqueeze(0), size=(128, 128),
                                        mode="bilinear", align_corners=False)
    per_frame = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    torch.nn.functional.interpolate(frames, size=(128, 128),
                                    mode="bilinear", align_corners=False)
    batched = (time.perf_counter() - t0) * 1e3
    print(f"  (b) resize 13 frames   : {per_frame:7.2f} ms per-frame vs "
          f"{batched:7.2f} ms batched  ({per_frame / batched:.2f}x)")


def section_cache_state(ds, n_items=24):
    """Are these reads hitting page cache? Decides whether the numbers transfer."""
    print()
    print("=" * 76)
    print("4. is this warm cache?")
    print("=" * 76)
    paths = _paths_of(ds, 0)
    total_mb = sum(Path(p).stat().st_size for p in paths) / 1e6
    print(f"  bytes per item          : {total_mb:6.2f} MB over {len(paths)} files "
          f"({total_mb / len(paths) * 1000:6.1f} KB each)")
    uniq = set()
    for i in range(len(ds.records)):
        uniq.update(_paths_of(ds, i))
    size_gb = sum(Path(p).stat().st_size for p in uniq) / 1e9
    print(f"  unique files in manifest: {len(uniq)}")
    print(f"  dataset size            : {size_gb:6.2f} GB")
    with open("/proc/meminfo") as f:
        mi = {k: v for k, v in (l.split(":", 1) for l in f)}
    cached_gb = int(mi["Cached"].split()[0]) / 1e6
    total_gb = int(mi["MemTotal"].split()[0]) / 1e6
    print(f"  host page cache         : {cached_gb:6.2f} GB of {total_gb:.0f} GB RAM")
    if size_gb < cached_gb:
        print(f"  -> the whole dataset fits in page cache, so section 1-3 numbers")
        print(f"     are WARM-cache numbers and will not transfer to a dataset")
        print(f"     larger than RAM.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--world", type=int, default=8)
    args = p.parse_args()
    cfg = ns(yaml.safe_load(open(args.config)))
    ds = section_ceiling(cfg, args.manifest, args.world)
    section_breakdown(ds)
    section_candidates(ds)
    section_cache_state(ds)


if __name__ == "__main__":
    main()
