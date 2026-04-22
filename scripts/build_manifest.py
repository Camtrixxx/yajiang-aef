from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.manifest import dump_manifest


def discover_frames(src_dir: Path):
    """
    约定输入文件名形如:
      1712016000000.npy
      1712016000000.npz
      1712016000000.pt
    时间戳部分会被解析为 timestamp_ms
    """
    frames = []
    if not src_dir.exists():
        return frames

    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in {".npy", ".npz", ".pt"}:
            continue
        try:
            timestamp_ms = int(path.stem.split("_")[0])
        except ValueError:
            continue
        frames.append(
            {
                "path": str(path.resolve()),
                "timestamp_ms": timestamp_ms,
            }
        )
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="预组织 patch 根目录")
    parser.add_argument("--output", type=str, required=True, help="输出 jsonl manifest")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--valid-start-ms", type=int, required=True)
    parser.add_argument("--valid-end-ms", type=int, required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)

    """
    目录约定:
    data_root/
      patch_0001/
        inputs/
          s2/
            1712016000000.npy
            1712880000000.npy
          s1/
            1712102400000.npy
        targets/
          dem.npy
          worldcover.npy
          jrc_water.npy
      patch_0002/
        ...
    """
    records = []

    for sample_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        inputs = {}
        for src_name in ["s2", "s1", "hls"]:
            frames = discover_frames(sample_dir / "inputs" / src_name)
            if len(frames) > 0:
                inputs[src_name] = {"frames": frames}

        targets = {}
        target_dir = sample_dir / "targets"
        for tgt_name in ["dem", "worldcover", "jrc_water"]:
            for suffix in [".npy", ".npz", ".pt"]:
                p = target_dir / f"{tgt_name}{suffix}"
                if p.exists():
                    targets[tgt_name] = {
                        "path": str(p.resolve()),
                        "relative_time": 0.5,
                        "metadata": [0.0, 0.0, 0.0, 0.0],
                    }
                    break

        if len(inputs) == 0 or len(targets) == 0:
            continue

        records.append(
            {
                "sample_id": sample_dir.name,
                "valid_start_ms": args.valid_start_ms,
                "valid_end_ms": args.valid_end_ms,
                "inputs": inputs,
                "targets": targets,
                "split": args.split,
                "meta": {},
            }
        )

    dump_manifest(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
