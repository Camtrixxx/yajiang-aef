from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

SRC_ROOT = Path("/workspace/hyh/yajiang-aef/data/debug_small")
DST_ROOT = Path("/workspace/hyh/yajiang-aef/data/debug_small_npy")

# v0.2-c 当前使用的数据源
INPUT_SOURCES = ["s2", "s1"]
TARGET_SOURCES = ["dem", "worldcover", "jrc_water"]

# DEM 标准化参数
# 当前 debug_small 8 patch 统计结果：
# min: 1794.1581
# max: 3596.0574
# mean: 2677.106
# std: 413.22937
DEM_MEAN = 2677.106
DEM_STD = 413.22937

# WorldCover 类别重映射
# 原始类别值:
# 10, 30, 40, 50, 60, 70, 80, 100
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

WORLDCOVER_IGNORE_INDEX = 255
STRICT_WORLDCOVER = True

# JRC Water 类别重映射
# 原始统计:
# -128, 2, 3, 4, 5, 6, 7, 9, 20, 21, 23, 27, 30, 33, 40
#
# -128 作为 nodata / ignore
# 有效类别重映射为 0~13
JRC_WATER_NODATA = -128
JRC_WATER_IGNORE_INDEX = 255

JRC_WATER_MAP = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
    9: 6,
    20: 7,
    21: 8,
    23: 9,
    27: 10,
    30: 11,
    33: 12,
    40: 13,
}

STRICT_JRC_WATER = True


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
    输入源统一转 float32。

    s2/s1 输出 shape:
        [C, H, W]
    """
    return arr.astype(np.float32)


def convert_dem(arr: np.ndarray) -> np.ndarray:
    """
    DEM 标准化。

    输入:
        [H, W]，原始高程值

    输出:
        [H, W]，float32，z-score 标准化结果
    """
    arr = arr.astype(np.float32)

    if DEM_STD <= 0:
        raise ValueError(f"DEM_STD must be positive, got {DEM_STD}")

    arr = (arr - DEM_MEAN) / DEM_STD
    return arr.astype(np.float32)


def convert_worldcover(arr: np.ndarray) -> np.ndarray:
    """
    WorldCover 类别重映射。

    输入:
        [H, W]，原始 ESA WorldCover 编码，例如 10/30/40/...

    输出:
        [H, W]，uint8，连续类别 id：0~7
    """
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


def convert_jrc_water(arr: np.ndarray) -> np.ndarray:
    """
    JRC Water 类别重映射，并将 nodata 转换为 ignore_index。

    输入:
        [H, W]，原始 JRC Water 编码，例如 -128/2/3/...

    输出:
        [H, W]，uint8
        有效类别: 0~13
        ignore 区域: 255
    """
    out = np.full(arr.shape, fill_value=JRC_WATER_IGNORE_INDEX, dtype=np.uint8)

    unique_vals = np.unique(arr)
    for v in unique_vals:
        v_int = int(v)

        if v_int == JRC_WATER_NODATA:
            out[arr == v] = JRC_WATER_IGNORE_INDEX
        elif v_int in JRC_WATER_MAP:
            out[arr == v] = JRC_WATER_MAP[v_int]
        else:
            if STRICT_JRC_WATER:
                raise ValueError(f"Unknown jrc_water class value: {v_int}")
            print(f"[WARN] Unknown jrc_water class value: {v_int}, keep as ignore_index")

    return out


def process_inputs(patch_id: str, dst_inputs: Path) -> None:
    """
    处理 s2 / s1 输入时序。
    """
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


def process_dem(patch_id: str, dst_targets: Path) -> None:
    """
    处理 DEM target。
    """
    dem_tif = SRC_ROOT / "dem" / patch_id / "static.tif"
    if not dem_tif.exists():
        raise FileNotFoundError(f"Missing DEM tif: {dem_tif}")

    dem_arr = read_tif(dem_tif)
    dem_arr = convert_dem(dem_arr)
    np.save(dst_targets / "dem.npy", dem_arr)


def process_worldcover(patch_id: str, dst_targets: Path) -> None:
    """
    处理 WorldCover target。
    """
    wc_tif = SRC_ROOT / "worldcover" / patch_id / "static.tif"
    if not wc_tif.exists():
        raise FileNotFoundError(f"Missing WorldCover tif: {wc_tif}")

    wc_arr = read_tif(wc_tif)
    wc_arr = convert_worldcover(wc_arr)
    np.save(dst_targets / "worldcover.npy", wc_arr)


def process_jrc_water(patch_id: str, dst_targets: Path) -> None:
    """
    处理 JRC Water target。
    """
    jrc_tif = SRC_ROOT / "jrc_water" / patch_id / "static.tif"
    if not jrc_tif.exists():
        raise FileNotFoundError(f"Missing JRC Water tif: {jrc_tif}")

    jrc_arr = read_tif(jrc_tif)
    jrc_arr = convert_jrc_water(jrc_arr)
    np.save(dst_targets / "jrc_water.npy", jrc_arr)


def process_patch(patch_id: str) -> None:
    dst_patch = DST_ROOT / patch_id
    dst_inputs = dst_patch / "inputs"
    dst_targets = dst_patch / "targets"

    dst_inputs.mkdir(parents=True, exist_ok=True)
    dst_targets.mkdir(parents=True, exist_ok=True)

    process_inputs(patch_id, dst_inputs)
    process_dem(patch_id, dst_targets)
    process_worldcover(patch_id, dst_targets)
    process_jrc_water(patch_id, dst_targets)


def save_preprocess_meta() -> None:
    """
    保存预处理元信息，方便之后复现实验。
    """
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": "v0.2-c",
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
            "ignore_index": WORLDCOVER_IGNORE_INDEX,
            "strict": STRICT_WORLDCOVER,
        },
        "jrc_water": {
            "mapping": JRC_WATER_MAP,
            "nodata": JRC_WATER_NODATA,
            "ignore_index": JRC_WATER_IGNORE_INDEX,
            "strict": STRICT_JRC_WATER,
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