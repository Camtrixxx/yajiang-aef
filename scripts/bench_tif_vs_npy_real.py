"""Compare loading times of the ACTUAL tif and npy data, on the same filesystem.

Prior bench_tif_vs_npy.py compared /tmp (local overlay) vs /data (quarkfs
FUSE mount), which skewed results. This version loads real tif files from
/data/heyuhang/dataset/raw/yajiang and real npy files from
data/full_npy, both on the same quarkfs mount, giving a fair format comparison.

Usage:
  PYTHONPATH=. /home/heyuhang/miniconda3/envs/hyh-dl/bin/python \
    scripts/bench_tif_vs_npy_real.py --patches 64
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np
import rasterio


def evict(path: Path) -> None:
    """Drop this file's pages from the page cache (best-effort on FUSE)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


def time_reads(files: list[Path], reader, cold: bool) -> tuple[float, float, int]:
    """Return (total_seconds, median_ms_per_file, bytes_read)."""
    per_file = []
    nbytes = 0
    t_total = 0.0
    for f in files:
        if cold:
            evict(f)
        t0 = time.perf_counter()
        a = reader(f)
        dt = time.perf_counter() - t0
        t_total += dt
        per_file.append(dt * 1e3)
        nbytes += a.nbytes
    return t_total, statistics.median(per_file), nbytes


def collect(root: Path, patches: list[str], subdirs: list[str], ext: str,
            layout: str) -> list[Path]:
    """Gather files for the given patches.

    layout='raw'  -> root/<source>/<patch>/<frame>.tif
    layout='npy'  -> root/<patch>/inputs/<source>/<frame>.npy
    """
    out = []
    for p in patches:
        for s in subdirs:
            d = root / s / p if layout == "raw" else root / p / "inputs" / s
            if d.is_dir():
                out.extend(sorted(d.glob(f"*{ext}")))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="/data/heyuhang/dataset/raw/yajiang")
    p.add_argument("--npy-root", default="data/full_npy")
    p.add_argument("--patches", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    raw = Path(args.raw_root)
    npy = Path(args.npy_root)
    sources = ["s2", "s1", "landsat"]

    # Only patches present in BOTH trees, so the two sides read the same content.
    raw_p = {d.name for d in (raw / "s2").iterdir() if d.is_dir()}
    npy_p = {d.name for d in npy.iterdir() if d.is_dir() and d.name.startswith("patch_")}
    patches = sorted(raw_p & npy_p)[: args.patches]
    if not patches:
        raise SystemExit("no overlapping patches between the two trees")

    tif_files = collect(raw, patches, sources, ".tif", "raw")
    npy_files = collect(npy, patches, sources, ".npy", "npy")

    print(f"patches            : {len(patches)}")
    print(f"tif files          : {len(tif_files)}")
    print(f"npy files          : {len(npy_files)}")
    tb = sum(f.stat().st_size for f in tif_files)
    nb = sum(f.stat().st_size for f in npy_files)
    print(f"tif on-disk        : {tb / 2**20:8.1f} MiB  (deflate, float64)")
    print(f"npy on-disk        : {nb / 2**20:8.1f} MiB  (raw, float32)")
    print(f"npy/tif size ratio : {nb / tb:8.2f}x")
    print("\nBoth trees are on the same quarkfs FUSE mount, so this isolates "
          "format\nfrom filesystem. Note the tif side is float64 and the npy "
          "side float32:\nthat dtype cast is part of what the conversion buys.")

    variants = {"tif": (tif_files, read_tif), "npy": (npy_files, read_npy)}

    print("\n=== COLD (posix_fadvise DONTNEED per file; best-effort on FUSE) ===")
    print(f"{'variant':10s} {'total s':>9s} {'ms/file':>9s} {'MiB/s':>9s}")
    cold = {}
    for name, (files, rd) in variants.items():
        tot, med, got = time_reads(files, rd, cold=True)
        cold[name] = tot
        print(f"{name:10s} {tot:9.3f} {med:9.3f} {got / 2**20 / tot:9.1f}")
    print(f"{'':10s} npy is {cold['tif'] / cold['npy']:.2f}x faster than tif")

    print("\n=== WARM (page cache primed) ===")
    print(f"{'variant':10s} {'total s':>9s} {'ms/file':>9s} {'MiB/s':>9s}")
    warm = {}
    for name, (files, rd) in variants.items():
        time_reads(files, rd, cold=False)  # prime
        runs = [time_reads(files, rd, cold=False) for _ in range(args.repeats)]
        tot = statistics.median(r[0] for r in runs)
        med = statistics.median(r[1] for r in runs)
        spread = max(r[0] for r in runs) - min(r[0] for r in runs)
        warm[name] = tot
        print(f"{name:10s} {tot:9.3f} {med:9.3f} "
              f"{runs[0][2] / 2**20 / tot:9.1f}   (spread {spread * 1e3:.0f} ms)")
    print(f"{'':10s} npy is {warm['tif'] / warm['npy']:.2f}x faster than tif")

    # Scale to one epoch over the full manifest: 1708 patches x 39 input frames.
    n_epoch = 1708 * 39
    scale = n_epoch / len(npy_files)
    print(f"\n=== extrapolated: one epoch of input frames ({n_epoch} files) ===")
    print("Single-process totals, before any num_workers parallelism.\n")
    print(f"{'variant':10s} {'cold s':>10s} {'warm s':>10s}")
    for name in variants:
        print(f"{name:10s} {cold[name] * scale:10.1f} {warm[name] * scale:10.1f}")
    print(f"\nsaving/epoch: cold {(cold['tif'] - cold['npy']) * scale:.1f} s, "
          f"warm {(warm['tif'] - warm['npy']) * scale:.1f} s")
    print("Divide by num_workers for the wall-clock effect on a real run, and "
          "only\nthe part not already hidden behind compute actually shows up.")


if __name__ == "__main__":
    main()
