from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(getattr(cfg.experiment, "seed", 42))

    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(args.split == "train"),
        num_workers=getattr(cfg.data, "num_workers", 0),
        pin_memory=True,
        collate_fn=aef_collate_fn,
    )

    model = AEFModel(cfg).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scaler = None
    if args.device.startswith("cuda") and getattr(cfg.training, "amp_dtype", "bf16") == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        cfg=cfg,
        device=args.device,
        scaler=scaler,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
