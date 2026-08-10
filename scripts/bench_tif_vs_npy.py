"""Is .npy actually faster to load than .tif, and by how much?

A colleague trains straight from GeoTIFF and one run takes over a day. The
question is whether converting to .npy is worth it. This measures the IO layer
in isolation, which gives the CEILING on what a format conversion can buy --
not the end-to-end speedup, which Amdahl will cut down (see
docs/experiments/v1.2_training_acceleration.md: our input side is already down
to 0.75 s of loader wait in a 20.1 s epoch, i.e. ~95% compute bound).

Builds a .tif mirror of the real .npy patches, then times per-file reads.

Variants:
  npy            np.load, the current path
  npy_mmap       np.load(mmap_mode='r') + copy
  tif_none       GeoTIFF, uncompressed, striped
  tif_deflate    GeoTIFF, DEFLATE level 6 (a very common default)
  tif_lzw        GeoTIFF, LZW
  tif_tiled_dfl  GeoTIFF, DEFLATE, 128x128 internal tiling

Cold vs warm matters more than the format here, so both are reported. Cold is
emulated per file with posix_fadvise(DONTNEED), which evicts just that file's
pages and needs no root. Warm is a second pass over the same files.

Reference: our dataset is 12.64 GB and fits entirely in a 903 GB page cache, so
the WARM numbers describe our machine's steady state. A colleague on a smaller
box, or with a dataset that does not fit, lives closer to the COLD numbers.

Usage:
  PYTHONPATH=. /home/heyuhang/miniconda3/envs/hyh-dl/bin/python \
    scripts/bench_tif_vs_npy.py --patches 64
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

# Explicit interpreter note: the system /usr/bin/python has an onnx/protobuf
# conflict that breaks torch imports. See the doc's measurement conventions.

TIF_VARIANTS = {
    "tif_none":      dict(compress=None,      tiled=False),
    "tif_deflate":   dict(compress="deflate", tiled=False),
    "tif_lzw":       dict(compress="lzw",     tiled=False),
    "tif_tiled_dfl": dict(compress="deflate", tiled=True),
}


def evict(path: Path) -> None:
    """Drop this file's pages from the page cache. No root needed."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def as_3d(a: np.ndarray) -> np.ndarray:
    """GeoTIFF is always band-major 3D; targets are 2D so promote them."""
    return a[None] if a.ndim == 2 else a


def write_tif(dst: Path, arr: np.ndarray, compress, tiled: bool) -> None:
    import rasterio
    from rasterio.transform import from_origin

    arr = as_3d(arr)
    bands, h, w = arr.shape
    kwargs = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=arr.dtype,
        # Realistic georeferencing -- a bare TIFF with no CRS would understate
        # the metadata parsing cost that a real GeoTIFF pays.
        crs="EPSG:4326",
        transform=from_origin(100.0, 30.0, 1e-4, 1e-4),
    )
    if compress:
        kwargs["compress"] = compress
    if tiled and h >= 128 and w >= 128:
        kwargs.update(tiled=True, blockxsize=128, blockysize=128)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **kwargs) as ds:
        ds.write(arr)


def read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def read_npy_mmap(path: Path) -> np.ndarray:
    return np.array(np.load(path, mmap_mode="r"))


def read_tif(path: Path) -> np.ndarray:
    import rasterio
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/full_npy")
    p.add_argument("--tif-root", default="/tmp/bench_tif_mirror")
    p.add_argument("--patches", type=int, default=64,
                   help="How many patches to mirror. 64 patches ~= 2688 files, "
                        "enough for a stable median without copying 12.64 GB.")
    p.add_argument("--repeats", type=int, default=3,
                   help="Warm-pass repeats; the median across repeats is used.")
    p.add_argument("--keep", action="store_true", help="Do not delete the tif mirror")
    args = p.parse_args()

    root = Path(args.data_root)
    tif_root = Path(args.tif_root)

    patches = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("patch_"))
    patches = patches[: args.patches]
    if not patches:
        raise SystemExit(f"no patch_* dirs under {root}")

    npy_files = sorted(f for d in patches for f in d.rglob("*.npy"))
    print(f"patches            : {len(patches)}")
    print(f"npy files          : {len(npy_files)}")
    src_bytes = sum(f.stat().st_size for f in npy_files)
    print(f"npy on-disk        : {src_bytes / 2**20:.1f} MiB")

    # ---- build the tif mirrors -------------------------------------------
    print("\nbuilding tif mirrors ...")
    tif_files: dict[str, list[Path]] = {}
    for name, opt in TIF_VARIANTS.items():
        vroot = tif_root / name
        if vroot.exists():
            shutil.rmtree(vroot)
        out = []
        t0 = time.perf_counter()
        for f in npy_files:
            rel = f.relative_to(root).with_suffix(".tif")
            dst = vroot / rel
            write_tif(dst, np.load(f), opt["compress"], opt["tiled"])
            out.append(dst)
        dt = time.perf_counter() - t0
        size = sum(x.stat().st_size for x in out)
        tif_files[name] = out
        print(f"  {name:14s} {size / 2**20:7.1f} MiB  "
              f"({size / src_bytes:4.2f}x npy)  built in {dt:5.1f} s")

    readers = {
        "npy":      (npy_files, read_npy),
        "npy_mmap": (npy_files, read_npy_mmap),
        **{k: (tif_files[k], read_tif) for k in TIF_VARIANTS},
    }

    # ---- cold ------------------------------------------------------------
    print("\n=== COLD (page cache evicted per file) ===")
    print(f"{'variant':16s} {'total s':>9s} {'ms/file':>9s} {'MiB/s':>9s} {'vs npy':>8s}")
    cold = {}
    for name, (files, rd) in readers.items():
        tot, med, nb = time_reads(files, rd, cold=True)
        cold[name] = (tot, med)
        base = cold["npy"][0]
        print(f"{name:16s} {tot:9.3f} {med:9.3f} {nb / 2**20 / tot:9.1f} "
              f"{base / tot:7.2f}x")

    # ---- warm ------------------------------------------------------------
    print("\n=== WARM (page cache primed) ===")
    print(f"{'variant':16s} {'total s':>9s} {'ms/file':>9s} {'MiB/s':>9s} {'vs npy':>8s}")
    warm = {}
    for name, (files, rd) in readers.items():
        time_reads(files, rd, cold=False)  # prime
        runs = [time_reads(files, rd, cold=False) for _ in range(args.repeats)]
        tot = statistics.median(r[0] for r in runs)
        med = statistics.median(r[1] for r in runs)
        spread = max(r[0] for r in runs) - min(r[0] for r in runs)
        warm[name] = (tot, med, spread)
        base = warm["npy"][0]
        print(f"{name:16s} {tot:9.3f} {med:9.3f} {runs[0][2] / 2**20 / tot:9.1f} "
              f"{base / tot:7.2f}x   (spread {spread * 1e3:.0f} ms)")

    # ---- what it means for an epoch --------------------------------------
    n_all = 1708 * 42  # full manifest: 1708 patches x 42 files
    scale = n_all / len(npy_files)
    print(f"\n=== extrapolated to the full manifest ({n_all} files) ===")
    print("Single-process, so this is loader work before any num_workers "
          "parallelism.\n")
    print(f"{'variant':16s} {'cold s/epoch':>14s} {'warm s/epoch':>14s}")
    for name in readers:
        print(f"{name:16s} {cold[name][0] * scale:14.1f} {warm[name][0] * scale:14.1f}")

    print("\nCaveats:")
    print("  - IO only. Says nothing about whether IO is the actual bottleneck")
    print("    in a given training run; measure GPU utilization for that.")
    print("  - Arrays here are small (6x128x128 and 6x43x43). GeoTIFF fixed")
    print("    per-file overhead dominates at this size and shrinks on big rasters.")
    print("  - posix_fadvise cold is a per-file evict, not a full cache drop;")
    print("    directory metadata stays warm, so real cold reads are slower.")

    if not args.keep:
        shutil.rmtree(tif_root, ignore_errors=True)
        print(f"\nremoved {tif_root} (pass --keep to retain)")


if __name__ == "__main__":
    main()
