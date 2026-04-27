from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

SRC_ROOT = Path("/workspace/hyh/yajiang-aef/data/debug_small")
DST_ROOT = Path("/workspace/hyh/yajiang-aef/data/debug_small_npy")

# 只处理 v0.2-a 当前需要的数据源
INPUT_SOURCES = ["s2", "s1"]
TARGET_SOURCES = ["dem", "worldcover"]

# DEM 标准化参数
# 当前 debug_small_npy 统计结果：
# min: 1794.1581, max: 3596.0574, mean: 2677.106, std: 413.22937
DEM_MEAN = 2677.106
DEM_STD = 413.22937

# worldcover 类别重映射
WORLDCOVER_MAP = {
    10: 0,
    30: 1,
    40: 2,
    50: 3,
    60: 4,
    70: 5,
    80: 6,
    100: 7,
}

# 是否严格要求 worldcover 不出现未知值
STRICT_WORLDCOVER = True


def read_tif(path: Path) -> np.ndarray:
    """
    rasterio 读出后统一返回:
    - 多波段: [C, H, W]
    - 单波段: [H, W]
    """
    with rasterio.open(path) as src:
        arr = src.read()  # [C, H, W]

    if arr.shape[0] == 1:
        return arr[0]

    return arr


def convert_input_array(arr: np.ndarray) -> np.ndarray:
    """
    输入源统一转 float32
    保持 shape:
    - s2/s1: [C, H, W]
    """
    return arr.astype(np.float32)


def convert_dem(arr: np.ndarray) -> np.ndarray:
    """
    DEM 标准化:
    原始高程值 -> 标准化连续值

    输入:
        arr: [H, W], 原始 DEM 高程值

    输出:
        arr: [H, W], float32，标准化后大致落在 [-2, 2] 附近
    """
    arr = arr.astype(np.float32)

    if DEM_STD <= 0:
        raise ValueError(f"DEM_STD must be positive, got {DEM_STD}")

    arr = (arr - DEM_MEAN) / DEM_STD
    return arr.astype(np.float32)


def convert_worldcover(arr: np.ndarray) -> np.ndarray:
    """
    worldcover 做类别重映射

    输入:
        [H, W]，原始 ESA WorldCover 编码，例如 10/30/40/...

    输出:
        [H, W]，uint8，连续类别 id：0~7
    """
    out = np.full_like(arr, fill_value=255, dtype=np.uint8)  # 255 作为临时未知值

    unique_vals = np.unique(arr)
    for v in unique_vals:
        v_int = int(v)
        if v_int in WORLDCOVER_MAP:
            out[arr == v] = WORLDCOVER_MAP[v_int]
        else:
            if STRICT_WORLDCOVER:
                raise ValueError(f"Unknown worldcover class value: {v_int}")
            print(f"[WARN] Unknown worldcover class value: {v_int}, keep as 255")

    if (out == 255).any() and STRICT_WORLDCOVER:
        bad_vals = np.unique(arr[out == 255])
        raise ValueError(f"Found unmapped worldcover values: {bad_vals}")

    return out


def process_patch(patch_id: str) -> None:
    dst_patch = DST_ROOT / patch_id
    dst_inputs = dst_patch / "inputs"
    dst_targets = dst_patch / "targets"
    dst_inputs.mkdir(parents=True, exist_ok=True)
    dst_targets.mkdir(parents=True, exist_ok=True)

    # 处理输入源
    for src_name in INPUT_SOURCES:
        src_patch_dir = SRC_ROOT / src_name / patch_id
        if not src_patch_dir.exists():
            raise FileNotFoundError(f"Missing input patch dir: {src_patch_dir}")

        dst_src_dir = dst_inputs / src_name
        dst_src_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(src_patch_dir.glob("*.tif"))
        if len(tif_files) == 0:
            raise FileNotFoundError(f"No tif files found in {src_patch_dir}")

        for tif_path in tif_files:
            arr = read_tif(tif_path)
            arr = convert_input_array(arr)
            out_path = dst_src_dir / f"{tif_path.stem}.npy"
            np.save(out_path, arr)

    # 处理 DEM
    dem_tif = SRC_ROOT / "dem" / patch_id / "static.tif"
    if not dem_tif.exists():
        raise FileNotFoundError(f"Missing DEM tif: {dem_tif}")

    dem_arr = read_tif(dem_tif)
    dem_arr = convert_dem(dem_arr)
    np.save(dst_targets / "dem.npy", dem_arr)

    # 处理 WorldCover
    wc_tif = SRC_ROOT / "worldcover" / patch_id / "static.tif"
    if not wc_tif.exists():
        raise FileNotFoundError(f"Missing WorldCover tif: {wc_tif}")

    wc_arr = read_tif(wc_tif)
    wc_arr = convert_worldcover(wc_arr)
    np.save(dst_targets / "worldcover.npy", wc_arr)


def save_preprocess_meta() -> None:
    """
    保存预处理元信息，方便之后复现实验。
    """
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    meta = {
        "src_root": str(SRC_ROOT),
        "dst_root": str(DST_ROOT),
        "input_sources": INPUT_SOURCES,
        "target_sources": TARGET_SOURCES,
        "dem": {
            "normalization": "zscore",
            "mean": DEM_MEAN,
            "std": DEM_STD,
        },
        "worldcover": {
            "mapping": WORLDCOVER_MAP,
            "ignore_value": 255,
            "strict": STRICT_WORLDCOVER,
        },
    }

    meta_path = DST_ROOT / "preprocess_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved preprocess meta to: {meta_path}")


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {SRC_ROOT}")

    s2_root = SRC_ROOT / "s2"
    if not s2_root.exists():
        raise FileNotFoundError(f"s2 root not found: {s2_root}")

    patch_ids = sorted(
        p.name for p in s2_root.iterdir()
        if p.is_dir() and p.name.startswith("patch_")
    )

    if len(patch_ids) == 0:
        raise RuntimeError(f"No patches found under {s2_root}")

    print(f"Found {len(patch_ids)} patches")

    for patch_id in patch_ids:
        print(f"Processing {patch_id}")
        process_patch(patch_id)

    save_preprocess_meta()

    print("Done.")
    print(f"Output saved to: {DST_ROOT}")


if __name__ == "__main__":
    main()