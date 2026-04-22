from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import load_config
from src.models.model import AEFModel
from src.training.trainer import Trainer


class DummyYajiangDataset(Dataset):
    """
    仅用于先打通主链路。
    你把真实 dataset 接进来后，保持输出字段一致即可。
    """
    SOURCE_NAME_TO_ID = {"s2": 0, "s1": 1, "hls": 2}

    def __init__(self, cfg, length: int = 32):
        self.cfg = cfg
        self.length = length
        self.image_size = cfg.data.image_size
        self.max_frames = cfg.data.max_frames
        self.input_sources = list(cfg.data.input_sources)
        self.target_sources = list(cfg.data.target_sources)

        source_channels = getattr(cfg.model, "source_channels", {})
        if isinstance(source_channels, dict):
            self.source_channels = source_channels
        else:
            self.source_channels = vars(source_channels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        h = w = self.image_size
        s = len(self.input_sources)
        t = self.max_frames
        max_in_ch = max(self.source_channels.values())

        source_frames = torch.randn(s, t, max_in_ch, h, w)
        source_timestamps_ms = torch.randint(1_600_000_000_000, 1_700_000_000_000, (s, t))
        source_frame_mask = torch.ones(s, t, dtype=torch.bool)
        source_input_mask = torch.ones(s, dtype=torch.bool)
        source_type_ids = torch.tensor([self.SOURCE_NAME_TO_ID[name] for name in self.input_sources], dtype=torch.long)

        valid_start_ms = torch.tensor(1_640_000_000_000, dtype=torch.long)
        valid_end_ms = torch.tensor(1_670_000_000_000, dtype=torch.long)

        num_tgt = len(self.target_sources)
        target_relative_time = torch.linspace(0.2, 0.8, steps=num_tgt).float()
        target_metadata = torch.zeros(num_tgt, self.cfg.data.metadata_dim).float()

        targets = {}
        target_masks = {}

        out_h = h // 2
        out_w = w // 2
        for tgt in self.target_sources:
            name = tgt.name if hasattr(tgt, "name") else tgt["name"]
            loss_type = tgt.loss_type if hasattr(tgt, "loss_type") else tgt["loss_type"]
            out_channels = tgt.out_channels if hasattr(tgt, "out_channels") else tgt["out_channels"]

            if loss_type == "categorical":
                targets[name] = torch.randint(0, out_channels, (1, out_h, out_w), dtype=torch.long)
                target_masks[name] = torch.ones(1, out_h, out_w, dtype=torch.bool)
            else:
                targets[name] = torch.randn(out_channels, out_h, out_w)
                target_masks[name] = torch.ones(out_channels, out_h, out_w, dtype=torch.bool)

        return {
            "source_frames": source_frames,
            "source_timestamps_ms": source_timestamps_ms,
            "source_frame_mask": source_frame_mask,
            "source_input_mask": source_input_mask,
            "source_type_ids": source_type_ids,
            "valid_start_ms": valid_start_ms,
            "valid_end_ms": valid_end_ms,
            "target_relative_time": target_relative_time,
            "target_metadata": target_metadata,
            "targets": targets,
            "target_masks": target_masks,
        }


def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if isinstance(batch[0][k], dict):
            out[k] = {kk: torch.stack([x[k][kk] for x in batch], dim=0) for kk in batch[0][k].keys()}
        else:
            out[k] = torch.stack([x[k] for x in batch], dim=0)
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(getattr(cfg.experiment, "seed", 42))

    model = AEFModel(cfg)
    model.to(args.device)

    dataset = DummyYajiangDataset(cfg, length=max(getattr(cfg.data, "num_samples", 0), 32))
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

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
