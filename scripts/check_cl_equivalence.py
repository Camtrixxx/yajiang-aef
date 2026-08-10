"""Is forcing channels_last on the 1x1 convs numerically inert?

The DDP grad-stride fix flips the memory format of all eight 1x1 conv weights.
For kernel_size=1 the two layouts address identical bytes (H and W indices are
always 0), so the math should be untouched -- but the layout can change which
kernel cuDNN/inductor picks, and a different kernel can accumulate in a
different order. That has to be measured, not assumed.

Single card, eval mode (so GroupNorm/BN behave deterministically), same seed,
same batch. Reports max abs diff on the embedding map and on the loss.

Reference scale: bf16 has an 8-bit mantissa, so its relative resolution is
2^-8 = 3.9e-3. A diff far below that is float noise, not a behavior change.

Usage:
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=.:scripts \
    python scripts/check_cl_equivalence.py
"""
import argparse
import types

import torch
import yaml

from bench_ddp8 import fwd_kwargs, move, ns
from bench_grad_stride import to_channels_last_1x1
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.losses import compute_total_loss
from torch.utils.data import DataLoader


def run(cfg, batch, dev, apply_cl, seed=42):
    torch.manual_seed(seed)
    model = AEFModel(cfg).to(dev)
    n = to_channels_last_1x1(model) if apply_cl else 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**fwd_kwargs(batch))
        loss = compute_total_loss(out, batch, cfg)
    return out.embedding_map.float(), float(loss.total), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--kappa", type=float, default=0.0,
                   help="0 disables the vMF randn injection, isolating float "
                        "accumulation order from RNG divergence. At the config "
                        "default (2000) the sampling noise dominates and the "
                        "comparison says nothing about layout.")
    args = p.parse_args()

    dev = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = ns(yaml.safe_load(open(args.config)))
    cfg.model.vmf_kappa = args.kappa
    print(f"vmf_kappa           : {args.kappa}")
    ds = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    loader = DataLoader(ds, batch_size=int(cfg.data.batch_size), shuffle=False,
                        num_workers=2, collate_fn=aef_collate_fn)
    batch = move(next(iter(loader)), dev)

    emb_a, loss_a, _ = run(cfg, batch, dev, apply_cl=False)
    emb_b, loss_b, n_cl = run(cfg, batch, dev, apply_cl=True)

    d_emb = (emb_a - emb_b).abs().max().item()
    d_loss = abs(loss_a - loss_b)
    scale = emb_a.abs().max().item()
    print(f"convs converted     : {n_cl}")
    print(f"loss baseline / cl  : {loss_a:.8f} / {loss_b:.8f}")
    print(f"max|d embedding_map|: {d_emb:.3e}   (embedding scale {scale:.3e})")
    print(f"|d loss|            : {d_loss:.3e}")
    print(f"bf16 resolution     : 3.9e-03  <- compare against this")
    print("VERDICT:", "inert" if d_emb < 1e-4 and d_loss < 1e-4 else "CHANGES MATH")


if __name__ == "__main__":
    main()
