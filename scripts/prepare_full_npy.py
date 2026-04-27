from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

DEFAULT_SRC_ROOT = Path("/workspace/raw/yajiang")
DEFAULT_DST_ROOT = Path("/workspace/hyh/yajiang-aef/data/full_npy")

INPUT_SOURCES = ["s2", "s1"]
TARGET_SOURCES = ["dem", "worldcover"]

WORLDCOVER_MAP = {
    10: 0,
    30: 1,
    40: 2,
    50: 3,
    60: 4,
    70: 5,
    80: 6,
    100: 7,
    # Full data contains shrubland (20), which debug_small did not cover.
    # Keep existing ids stable so partially generated data can be resumed.
    20: 8,
}

WORLDCOVER_IGNORE_INDEX = 255
STRICT_WORLDCOVER = True


def read_tif(path: Path) -> np.ndarray:
    """
    rasterio 读出后统一返回:
    - 多波段: [C, H, W]
    - 单波段: [H, W]
    """
    with rasterio.open(path) as src:
        arr = src.read()

    if arr.shape[0] == 1:
        return arr[0]
    return arr


def iter_patch_ids(src_root: Path) -> list[str]:
    s2_root = src_root / "s2"
    if not s2_root.exists():
        raise FileNotFoundError(f"s2 root not found: {s2_root}")

    patch_ids = sorted(
        p.name for p in s2_root.iterdir()
        if p.is_dir() and p.name.startswith("patch_")
    )
    if len(patch_ids) == 0:
        raise RuntimeError(f"No patches found under {s2_root}")

    return patch_ids


def validate_patch_sources(src_root: Path, patch_ids: list[str]) -> None:
    missing: list[str] = []
    for src_name in [*INPUT_SOURCES, *TARGET_SOURCES]:
        src_dir = src_root / src_name
        if not src_dir.exists():
            missing.append(str(src_dir))
            continue
        for patch_id in patch_ids:
            if not (src_dir / patch_id).exists():
                missing.append(str(src_dir / patch_id))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break

    if missing:
        shown = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing source directories, first {min(len(missing), 20)}:\n{shown}")


def compute_dem_stats(src_root: Path, patch_ids: list[str]) -> tuple[float, float]:
    """
    用 sum/sumsq 流式统计全量 DEM，避免把所有 patch 同时放进内存。
    """
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0

    for idx, patch_id in enumerate(patch_ids, start=1):
        dem_tif = src_root / "dem" / patch_id / "static.tif"
        if not dem_tif.exists():
            raise FileNotFoundError(f"Missing DEM tif: {dem_tif}")

        arr = read_tif(dem_tif).astype(np.float64)
        finite = np.isfinite(arr)
        if finite.any():
            vals = arr[finite]
            total_count += int(vals.size)
            total_sum += float(vals.sum())
            total_sumsq += float(np.square(vals).sum())

        if idx % 100 == 0 or idx == len(patch_ids):
            print(f"DEM stats: scanned {idx}/{len(patch_ids)} patches")

    if total_count == 0:
        raise RuntimeError("No finite DEM values found")

    mean = total_sum / total_count
    variance = max(total_sumsq / total_count - mean * mean, 0.0)
    std = variance ** 0.5
    if std <= 0:
        raise ValueError(f"Computed DEM std must be positive, got {std}")

    return mean, std


def convert_input_array(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)


def convert_dem(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    return ((arr - mean) / std).astype(np.float32)


def convert_worldcover(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, fill_value=WORLDCOVER_IGNORE_INDEX, dtype=np.uint8)

    unique_vals = np.unique(arr)
    for v in unique_vals:
        v_int = int(v)
        if v_int in WORLDCOVER_MAP:
            out[arr == v] = WORLDCOVER_MAP[v_int]
        else:
            if STRICT_WORLDCOVER:
                raise ValueError(f"Unknown worldcover class value: {v_int}")
            print(f"[WARN] Unknown worldcover class value: {v_int}, keep as ignore_index")

    if (out == WORLDCOVER_IGNORE_INDEX).any() and STRICT_WORLDCOVER:
        bad_vals = np.unique(arr[out == WORLDCOVER_IGNORE_INDEX])
        raise ValueError(f"Found unmapped worldcover values: {bad_vals}")

    return out


def process_inputs(src_root: Path, patch_id: str, dst_inputs: Path, skip_existing: bool) -> None:
    for src_name in INPUT_SOURCES:
        src_patch_dir = src_root / src_name / patch_id
        if not src_patch_dir.exists():
            raise FileNotFoundError(f"Missing input patch dir: {src_patch_dir}")

        dst_src_dir = dst_inputs / src_name
        dst_src_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(src_patch_dir.glob("*.tif"))
        if len(tif_files) == 0:
            raise FileNotFoundError(f"No tif files found in {src_patch_dir}")

        for tif_path in tif_files:
            out_path = dst_src_dir / f"{tif_path.stem}.npy"
            if skip_existing and out_path.exists():
                continue

            arr = convert_input_array(read_tif(tif_path))
            np.save(out_path, arr)


def process_dem(
    src_root: Path,
    patch_id: str,
    dst_targets: Path,
    dem_mean: float,
    dem_std: float,
    skip_existing: bool,
) -> None:
    out_path = dst_targets / "dem.npy"
    if skip_existing and out_path.exists():
        return

    dem_tif = src_root / "dem" / patch_id / "static.tif"
    if not dem_tif.exists():
        raise FileNotFoundError(f"Missing DEM tif: {dem_tif}")

    dem_arr = convert_dem(read_tif(dem_tif), mean=dem_mean, std=dem_std)
    np.save(out_path, dem_arr)


def process_worldcover(src_root: Path, patch_id: str, dst_targets: Path, skip_existing: bool) -> None:
    out_path = dst_targets / "worldcover.npy"
    if skip_existing and out_path.exists():
        return

    wc_tif = src_root / "worldcover" / patch_id / "static.tif"
    if not wc_tif.exists():
        raise FileNotFoundError(f"Missing WorldCover tif: {wc_tif}")

    wc_arr = convert_worldcover(read_tif(wc_tif))
    np.save(out_path, wc_arr)


def process_patch(
    src_root: Path,
    dst_root: Path,
    patch_id: str,
    dem_mean: float,
    dem_std: float,
    skip_existing: bool,
) -> None:
    dst_patch = dst_root / patch_id
    dst_inputs = dst_patch / "inputs"
    dst_targets = dst_patch / "targets"

    dst_inputs.mkdir(parents=True, exist_ok=True)
    dst_targets.mkdir(parents=True, exist_ok=True)

    process_inputs(src_root, patch_id, dst_inputs, skip_existing=skip_existing)
    process_dem(src_root, patch_id, dst_targets, dem_mean, dem_std, skip_existing=skip_existing)
    process_worldcover(src_root, patch_id, dst_targets, skip_existing=skip_existing)


def save_preprocess_meta(
    src_root: Path,
    dst_root: Path,
    dem_mean: float,
    dem_std: float,
    patch_count: int,
) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": "v0.3-a",
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "patch_count": patch_count,
        "input_sources": INPUT_SOURCES,
        "target_sources": TARGET_SOURCES,
        "dem": {
            "normalization": "zscore",
            "mean": dem_mean,
            "std": dem_std,
        },
        "worldcover": {
            "mapping": WORLDCOVER_MAP,
            "ignore_index": WORLDCOVER_IGNORE_INDEX,
            "strict": STRICT_WORLDCOVER,
        },
    }

    meta_path = dst_root / "preprocess_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved preprocess meta to: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert full Yajiang GeoTIFF data to npy files.")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--dem-mean", type=float, default=None)
    parser.add_argument("--dem-std", type=float, default=None)
    parser.add_argument(
        "--max-patches",
        type=int,
        default=0,
        help="Only process the first N patches; 0 means all patches.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip npy files that already exist under dst-root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = args.src_root
    dst_root = args.dst_root

    if not src_root.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {src_root}")

    patch_ids = iter_patch_ids(src_root)
    validate_patch_sources(src_root, patch_ids)
    if args.max_patches < 0:
        raise ValueError(f"--max-patches must be >= 0, got {args.max_patches}")
    if args.max_patches > 0:
        patch_ids = patch_ids[:args.max_patches]
    print(f"Found {len(patch_ids)} patches")

    if (args.dem_mean is None) != (args.dem_std is None):
        raise ValueError("--dem-mean and --dem-std must be provided together")

    if args.dem_mean is None:
        dem_mean, dem_std = compute_dem_stats(src_root, patch_ids)
    else:
        dem_mean = float(args.dem_mean)
        dem_std = float(args.dem_std)
        if dem_std <= 0:
            raise ValueError(f"--dem-std must be positive, got {dem_std}")

    print(f"DEM mean: {dem_mean}")
    print(f"DEM std: {dem_std}")

    for idx, patch_id in enumerate(patch_ids, start=1):
        if idx % 50 == 1 or idx == len(patch_ids):
            print(f"Processing {idx}/{len(patch_ids)}: {patch_id}")
        process_patch(
            src_root,
            dst_root,
            patch_id,
            dem_mean=dem_mean,
            dem_std=dem_std,
            skip_existing=args.skip_existing,
        )

    save_preprocess_meta(src_root, dst_root, dem_mean, dem_std, patch_count=len(patch_ids))

    print("Done.")
    print(f"Output saved to: {dst_root}")


if __name__ == "__main__":
    main()
