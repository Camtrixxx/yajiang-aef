from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/workspace/hyh/yajiang-aef/data/full_npy")
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_ROOT / "train.jsonl"

INPUT_SOURCES = ["s2", "s1"]
TARGET_SOURCES = ["dem", "worldcover", "jrc_water"]

# 当前季度输入覆盖 2023Q1 ~ 2026Q1。
VALID_START_MS = 1672531200000  # 2023-01-01 00:00:00 UTC
VALID_END_MS = 1767225600000    # 2026-01-01 00:00:00 UTC

DEFAULT_TARGET_RELATIVE_TIME = 0.5
DEFAULT_METADATA = [0.0, 0.0, 0.0, 0.0]


def quarter_to_timestamp_ms(name: str) -> int:
    """
    例如:
      2023Q1 -> 2023-01-01
      2023Q2 -> 2023-04-01
      2023Q3 -> 2023-07-01
      2023Q4 -> 2023-10-01
    """
    if len(name) != 6 or name[4] != "Q":
        raise ValueError(f"Unexpected quarter name: {name}")

    year = int(name[:4])
    quarter = int(name[5])
    month_map = {
        1: 1,
        2: 4,
        3: 7,
        4: 10,
    }

    if quarter not in month_map:
        raise ValueError(f"Unexpected quarter name: {name}")

    timestamp = dt.datetime(
        year,
        month_map[quarter],
        1,
        0,
        0,
        0,
        tzinfo=dt.timezone.utc,
    ).timestamp()

    return int(timestamp * 1000)


def build_input_frames(patch_dir: Path, src_name: str) -> dict:
    src_dir = patch_dir / "inputs" / src_name
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {src_dir}")

    frames = []
    for npy_path in sorted(src_dir.glob("*.npy")):
        frames.append(
            {
                "path": str(npy_path.resolve()),
                "timestamp_ms": quarter_to_timestamp_ms(npy_path.stem),
            }
        )

    if len(frames) == 0:
        raise RuntimeError(f"No input frames found in {src_dir}")

    return {"frames": frames}


def build_target_record(patch_dir: Path, target_name: str) -> dict:
    target_path = patch_dir / "targets" / f"{target_name}.npy"
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target: {target_path}")

    return {
        "path": str(target_path.resolve()),
        "relative_time": DEFAULT_TARGET_RELATIVE_TIME,
        "metadata": DEFAULT_METADATA,
    }


def build_one_record(patch_dir: Path, split: str) -> dict:
    inputs = {}
    for src_name in INPUT_SOURCES:
        inputs[src_name] = build_input_frames(patch_dir, src_name)

    targets = {}
    for target_name in TARGET_SOURCES:
        targets[target_name] = build_target_record(patch_dir, target_name)

    return {
        "sample_id": patch_dir.name,
        "valid_start_ms": VALID_START_MS,
        "valid_end_ms": VALID_END_MS,
        "inputs": inputs,
        "targets": targets,
        "split": split,
        "meta": {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest for full Yajiang npy data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")

    patch_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and p.name.startswith("patch_")
    )

    if len(patch_dirs) == 0:
        raise RuntimeError(f"No patch dirs found under {data_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for patch_dir in patch_dirs:
            rec = build_one_record(patch_dir, split=args.split)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(patch_dirs)} records to {args.output}")


if __name__ == "__main__":
    main()
