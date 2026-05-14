from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


DEFAULT_SRC_ROOT = Path("/workspace/raw/yajiang/landsat")
DEFAULT_DST_ROOT = Path("/workspace/hyh/yajiang-aef/data/full_npy")


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    return arr.astype(np.float32)


def iter_patch_ids(src_root: Path) -> list[str]:
    patch_ids = sorted(
        p.name for p in src_root.iterdir()
        if p.is_dir() and p.name.startswith("patch_")
    )
    if not patch_ids:
        raise RuntimeError(f"No patch dirs found under {src_root}")
    return patch_ids


def process_patch(src_root: Path, dst_root: Path, patch_id: str, skip_existing: bool) -> tuple[int, int]:
    src_patch = src_root / patch_id
    dst_dir = dst_root / patch_id / "inputs" / "landsat"
    if not src_patch.exists():
        raise FileNotFoundError(f"Missing Landsat patch dir: {src_patch}")

    tif_paths = sorted(src_patch.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No Landsat tif files found in {src_patch}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for tif_path in tif_paths:
        out_path = dst_dir / f"{tif_path.stem}.npy"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        arr = read_tif(tif_path)
        if arr.ndim != 3:
            raise ValueError(f"Expected [C,H,W] array from {tif_path}, got shape={arr.shape}")
        np.save(out_path, arr)
        written += 1
    return written, skipped


def save_meta(src_root: Path, dst_root: Path, patch_count: int, written: int, skipped: int) -> None:
    meta = {
        "version": "v1.1",
        "source": "landsat",
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "patch_count": patch_count,
        "frames_written": written,
        "frames_skipped": skipped,
        "output_layout": "patch_xxxxxx/inputs/landsat/YYYYQn.npy",
        "channels": 6,
        "dtype": "float32",
    }
    out_path = dst_root / "landsat_meta.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved Landsat meta to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Yajiang Landsat GeoTIFF data to full_npy inputs.")
    parser.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    parser.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT)
    parser.add_argument("--max-patches", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src_root.exists():
        raise FileNotFoundError(f"Landsat source root not found: {args.src_root}")
    if not args.dst_root.exists():
        raise FileNotFoundError(f"Destination data root not found: {args.dst_root}")
    if args.max_patches < 0:
        raise ValueError(f"--max-patches must be >= 0, got {args.max_patches}")

    patch_ids = iter_patch_ids(args.src_root)
    if args.max_patches > 0:
        patch_ids = patch_ids[: args.max_patches]

    total_written = 0
    total_skipped = 0
    print(f"Found {len(patch_ids)} Landsat patches")
    for idx, patch_id in enumerate(patch_ids, start=1):
        if idx % 50 == 1 or idx == len(patch_ids):
            print(f"Processing {idx}/{len(patch_ids)}: {patch_id}")
        written, skipped = process_patch(
            src_root=args.src_root,
            dst_root=args.dst_root,
            patch_id=patch_id,
            skip_existing=args.skip_existing,
        )
        total_written += written
        total_skipped += skipped

    save_meta(args.src_root, args.dst_root, len(patch_ids), total_written, total_skipped)
    print(f"Done. frames_written={total_written}, frames_skipped={total_skipped}")


if __name__ == "__main__":
    main()
