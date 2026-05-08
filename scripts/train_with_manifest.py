from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.trainer import Trainer
from src.utils.device import build_grad_scaler, resolve_device, set_seed, should_pin_memory
from src.utils.distributed import cleanup_distributed, init_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(getattr(cfg.experiment, "seed", 42))
    device = resolve_device(args.device)
    distributed = init_distributed(device)
    if distributed.is_main_process:
        print(f"Using device: {device}")
        if distributed.enabled:
            print(f"Distributed training: world_size={distributed.world_size}")

    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split=args.split)
    sampler = None
    if distributed.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed.world_size,
            rank=distributed.rank,
            shuffle=(args.split == "train"),
            drop_last=False,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(args.split == "train" and sampler is None),
        num_workers=getattr(cfg.data, "num_workers", 0),
        pin_memory=should_pin_memory(device),
        sampler=sampler,
        collate_fn=aef_collate_fn,
    )

    model = AEFModel(cfg).to(device)
    if distributed.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type in {"cuda", "npu"} else None,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scaler = build_grad_scaler(device, getattr(cfg.training, "amp_dtype", "bf16"))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        cfg=cfg,
        device=device,
        scaler=scaler,
        distributed=distributed,
    )
    try:
        trainer.fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
