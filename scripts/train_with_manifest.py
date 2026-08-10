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


def configure_cuda_backend(cfg) -> None:
    if not torch.cuda.is_available():
        return

    training_cfg = getattr(cfg, "training", None)
    if getattr(training_cfg, "allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(getattr(training_cfg, "matmul_precision", "high"))
    torch.backends.cudnn.benchmark = bool(getattr(training_cfg, "cudnn_benchmark", True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-no-optimizer", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_cuda_backend(cfg)
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

    num_workers = int(getattr(cfg.data, "num_workers", 0))
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(getattr(cfg.data, "persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(getattr(cfg.data, "prefetch_factor", 4))

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=(args.split == "train" and sampler is None),
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        sampler=sampler,
        collate_fn=aef_collate_fn,
        drop_last=bool(getattr(cfg.data, "drop_last", distributed.enabled and args.split == "train")),
        **loader_kwargs,
    )

    model = AEFModel(cfg).to(device)

    # Fix DDP grad-stride warning: convert 1x1 conv weights to channels_last so
    # DDP buckets with the stride inductor will produce, eliminating the copy.
    # Measured numerically inert (5.960e-08 diff vs 3.836e-01 scale, kappa=0)
    # and zero-cost (step 349.5-352.8ms all within spread, peak 20.86GB fixed).
    if bool(getattr(cfg.training, "channels_last_1x1_conv", True)):
        n_cl = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1):
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
                n_cl += 1
        if distributed.is_main_process and n_cl > 0:
            print(f"Converted {n_cl} 1x1 conv weights to channels_last for DDP bucketing")

    if distributed.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type in {"cuda", "npu"} else None,
            find_unused_parameters=bool(getattr(cfg.training, "find_unused_parameters", False)),
        )

    # torch.compile goes OUTSIDE DDP so DDPOptimizer can split the graph at
    # gradient-bucket boundaries and keep comm/compute overlap. Measured on one
    # A800 at batch_size=4: 597.9 ms -> 362.2 ms (1.65x) and 41.4 -> 26.2 GB.
    if bool(getattr(cfg.training, "compile", False)):
        if device.type != "cuda":
            print(f"[compile] skipped: device is {device.type}, not cuda")
        else:
            mode = str(getattr(cfg.training, "compile_mode", "default"))
            # Shapes are static here (fixed batch_size + drop_last + padded
            # max_frames), so recompilation should not trigger. If it does,
            # TORCH_LOGS=recompiles will say why.
            model = torch.compile(model, mode=mode)
            print(f"[compile] enabled, mode={mode}")

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
    if args.resume:
        trainer.load_checkpoint(
            args.resume,
            load_optimizer=not args.resume_no_optimizer,
            override_optimizer_lr=True,
        )
    try:
        trainer.fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
