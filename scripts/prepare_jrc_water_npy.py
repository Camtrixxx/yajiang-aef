from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SRC_ROOT = Path("/workspace/raw/yajiang/jrc_water")
DEFAULT_DST_ROOT = Path("/workspace/hyh/yajiang-aef/data/full_npy")

JRC_NODATA = -128
IGNORE_INDEX = 255


def read_jrc_tif(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Failed to read JRC water tif: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected single-band JRC water tif, got shape={arr.shape} at {path}")
    return arr


def convert_jrc_water(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, fill_value=IGNORE_INDEX, dtype=np.uint8)
    valid = arr != JRC_NODATA

    if valid.any():
        valid_values = arr[valid]
        bad = valid_values[(valid_values < 0) | (valid_values > 100)]
        if bad.size > 0:
            unique_bad = np.unique(bad)[:20]
            raise ValueError(f"Unexpected JRC water values: {unique_bad}")
        out[valid] = valid_values.astype(np.uint8)

    return out


def iter_patch_dirs(src_root: Path) -> list[Path]:
    patch_dirs = sorted(
        p for p in src_root.iterdir()
        if p.is_dir() and p.name.startswith("patch_")
    )
    if not patch_dirs:
        raise RuntimeError(f"No patch dirs found under {src_root}")
    return patch_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JRC water GeoTIFF targets to npy files.")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-patches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src_root.exists():
        raise FileNotFoundError(f"JRC source root not found: {args.src_root}")
    if not args.dst_root.exists():
        raise FileNotFoundError(f"Destination full_npy root not found: {args.dst_root}")

    patch_dirs = iter_patch_dirs(args.src_root)
    if args.max_patches < 0:
        raise ValueError(f"--max-patches must be >= 0, got {args.max_patches}")
    if args.max_patches > 0:
        patch_dirs = patch_dirs[: args.max_patches]

    written = 0
    skipped = 0
    for idx, src_patch in enumerate(patch_dirs, start=1):
        dst_targets = args.dst_root / src_patch.name / "targets"
        if not dst_targets.exists():
            raise FileNotFoundError(f"Missing target directory for patch: {dst_targets}")

        src_tif = src_patch / "static.tif"
        if not src_tif.exists():
            raise FileNotFoundError(f"Missing JRC water tif: {src_tif}")

        out_path = dst_targets / "jrc_water.npy"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        arr = convert_jrc_water(read_jrc_tif(src_tif))
        np.save(out_path, arr)
        written += 1

        if idx % 100 == 0 or idx == len(patch_dirs):
            print(f"Processed {idx}/{len(patch_dirs)} patches")

    meta = {
        "source": "jrc_water",
        "src_root": str(args.src_root),
        "dst_root": str(args.dst_root),
        "nodata": JRC_NODATA,
        "ignore_index": IGNORE_INDEX,
        "classes": 101,
        "written": written,
        "skipped": skipped,
    }
    meta_path = args.dst_root / "jrc_water_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Done. written={written}, skipped={skipped}")
    print(f"Saved meta to: {meta_path}")


if __name__ == "__main__":
    main()
