"""Where does the tif read time actually go?

bench_tif_vs_npy_real.py says npy loads ~5x faster than the real deflate tif.
That 5x is a bundle of three separable things, and they have different fixes:

  1. rasterio.open()   -- header, CRS, block index parsing (per-file fixed cost)
  2. ds.read()         -- DEFLATE decompression + assembly
  3. dtype             -- tif is float64, npy is float32, so tif moves 2x bytes

If (1) dominates, the fix is fewer/larger files, not a format change.
If (2) dominates, dropping compression is enough and you keep GeoTIFF.
If (3) matters, casting to float32 is a win in ANY format.

Also measures npy float64 vs float32 so the dtype effect is visible on the
npy side too, and an uncompressed-tif variant written to the SAME filesystem
to keep the comparison honest.

Usage:
  PYTHONPATH=. /home/heyuhang/miniconda3/envs/hyh-dl/bin/python \
    scripts/bench_tif_breakdown.py --files 400
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def timeit(fn, files, repeats: int) -> float:
    """Median wall-clock seconds for one pass over `files`."""
    fn(files)  # prime
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(files)
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs)


# ---- the passes we compare -------------------------------------------------

def p_open_only(files):
    for f in files:
        with rasterio.open(f) as ds:
            _ = ds.profile


def p_open_read(files):
    for f in files:
        with rasterio.open(f) as ds:
            _ = ds.read()


def p_open_read_cast(files):
    for f in files:
        with rasterio.open(f) as ds:
            _ = ds.read().astype(np.float32)


def p_npy(files):
    for f in files:
        _ = np.load(f)


def write_variant(src_tif: list[Path], dst_dir: Path, compress, dtype):
    """Mirror the tifs into dst_dir with the given compression and dtype."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, f in enumerate(src_tif):
        with rasterio.open(f) as ds:
            arr = ds.read().astype(dtype)
        bands, h, w = arr.shape
        kw = dict(driver="GTiff", height=h, width=w, count=bands,
                  dtype=arr.dtype, crs="EPSG:4326",
                  transform=from_origin(100.0, 30.0, 1e-4, 1e-4))
        if compress:
            kw["compress"] = compress
        dst = dst_dir / f"{i:06d}.tif"
        with rasterio.open(dst, "w", **kw) as ds:
            ds.write(arr)
        out.append(dst)
    return out


def write_npy(src_tif: list[Path], dst_dir: Path, dtype):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, f in enumerate(src_tif):
        with rasterio.open(f) as ds:
            arr = ds.read().astype(dtype)
        dst = dst_dir / f"{i:06d}.npy"
        np.save(dst, arr)
        out.append(dst)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="/data/heyuhang/dataset/raw/yajiang")
    # Scratch must sit on the same mount as the real data, or the FUSE-vs-local
    # difference swamps the format difference (that mistake cost us a run).
    p.add_argument("--scratch", default="/data/heyuhang/yajiang-aef/.bench_scratch")
    p.add_argument("--files", type=int, default=400)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()

    raw = Path(args.raw_root)
    scratch = Path(args.scratch)

    src = sorted((raw / "s2").rglob("*.tif"))[: args.files]
    if not src:
        raise SystemExit(f"no tifs under {raw / 's2'}")
    with rasterio.open(src[0]) as ds:
        print(f"source tif : {ds.count}x{ds.height}x{ds.width} {ds.dtypes[0]} "
              f"{ds.compression}, tiled={ds.is_tiled}")
    print(f"files      : {len(src)}   scratch: {scratch}")

    print("\nbuilding variants on the same filesystem ...")
    v = {}
    v["tif_deflate_f64"] = src  # the real thing, as-is
    v["tif_none_f64"] = write_variant(src, scratch / "tif_none_f64", None, np.float64)
    v["tif_none_f32"] = write_variant(src, scratch / "tif_none_f32", None, np.float32)
    v["tif_deflate_f32"] = write_variant(src, scratch / "tif_dfl_f32", "deflate", np.float32)
    v["npy_f64"] = write_npy(src, scratch / "npy_f64", np.float64)
    v["npy_f32"] = write_npy(src, scratch / "npy_f32", np.float32)
    for k, files in v.items():
        mb = sum(f.stat().st_size for f in files) / 2**20
        print(f"  {k:18s} {mb:8.1f} MiB")

    print("\n=== read passes (warm, median of "
          f"{args.repeats}) ===")
    print(f"{'pass':28s} {'total s':>9s} {'ms/file':>9s} {'vs npy_f32':>11s}")

    results = {}
    # tif: open-only isolates the fixed per-file parsing cost
    results["tif_deflate_f64 open only"] = timeit(p_open_only, v["tif_deflate_f64"], args.repeats)
    results["tif_deflate_f64 open+read"] = timeit(p_open_read, v["tif_deflate_f64"], args.repeats)
    results["tif_deflate_f64 +cast f32"] = timeit(p_open_read_cast, v["tif_deflate_f64"], args.repeats)
    results["tif_deflate_f32 open+read"] = timeit(p_open_read, v["tif_deflate_f32"], args.repeats)
    results["tif_none_f64    open+read"] = timeit(p_open_read, v["tif_none_f64"], args.repeats)
    results["tif_none_f32    open+read"] = timeit(p_open_read, v["tif_none_f32"], args.repeats)
    results["npy_f64         load"] = timeit(p_npy, v["npy_f64"], args.repeats)
    results["npy_f32         load"] = timeit(p_npy, v["npy_f32"], args.repeats)

    base = results["npy_f32         load"]
    n = len(src)
    for k, t in results.items():
        print(f"{k:28s} {t:9.3f} {t / n * 1e3:9.3f} {t / base:10.2f}x")

    print("\n=== attribution ===")
    o = results["tif_deflate_f64 open only"]
    r = results["tif_deflate_f64 open+read"]
    nf64 = results["npy_f64         load"]
    nf32 = results["npy_f32         load"]
    tn64 = results["tif_none_f64    open+read"]
    print(f"  rasterio.open fixed cost   : {o / r * 100:5.1f}% of the tif read")
    print(f"  decompression (dfl->none)  : {(r - tn64) / r * 100:5.1f}% of the tif read")
    print(f"  dtype f64->f32 on npy      : {(nf64 - nf32) / nf64 * 100:5.1f}% of npy_f64")
    print(f"  total real tif -> npy_f32  : {r / nf32:5.2f}x")
    print(f"  tif_none_f32 -> npy_f32    : "
          f"{results['tif_none_f32    open+read'] / nf32:5.2f}x   "
          f"(what remains after fixing compression+dtype)")

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"\nremoved {scratch}")


if __name__ == "__main__":
    main()
