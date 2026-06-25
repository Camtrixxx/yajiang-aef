from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models.model import AEFModel


def config_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {k: config_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [config_to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a training checkpoint to AEF deploy format.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model = AEFModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config_to_dict(cfg),
            "epoch": int(ckpt.get("epoch", 0)),
            "global_step": int(ckpt.get("global_step", 0)),
            "format": "aef_deploy_v1",
        },
        output_path,
    )
    print(f"Exported deploy model to {output_path}")


if __name__ == "__main__":
    main()
