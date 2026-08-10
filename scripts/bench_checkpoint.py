"""Attribute the per-epoch non-step wall time seen in train.log.

Log timestamps imply ~12 s/epoch between "Epoch done" and the second
"Saved checkpoint" line, but torch.save of an already-on-CPU payload measures
only 0.73 s. This script reproduces the trainer's exact save path with
GPU-resident params + AdamW state to find where the time actually goes.

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/bench_checkpoint.py \
      --config configs/yajiang_v1_2.yaml --manifest data/full_npy/train.jsonl
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from src.config import load_config
from src.models.model import AEFModel


def timed(fn, repeats: int = 3):
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = load_config(args.config)
    model = AEFModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n = sum(p.numel() for p in model.parameters())
    print(f"params = {n/1e6:.2f} M")

    # Materialize AdamW exp_avg / exp_avg_sq so the payload matches a real run.
    loss = sum(p.float().pow(2).sum() for p in model.parameters())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    sd_med, sd_all = timed(lambda: model.state_dict(), args.repeats)
    print(f"model.state_dict()            median={sd_med*1e3:8.1f} ms  {[f'{x*1e3:.0f}' for x in sd_all]}")
    op_med, op_all = timed(lambda: opt.state_dict(), args.repeats)
    print(f"optimizer.state_dict()        median={op_med*1e3:8.1f} ms  {[f'{x*1e3:.0f}' for x in op_all]}")

    def build():
        return {
            "epoch": 1,
            "global_step": 1,
            "best_loss": 0.0,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
        }

    targets = [
        ("quarkfs (current)", "outputs/_bench_ckpt.pt"),
        ("/dev/shm (RAM)", "/dev/shm/_bench_ckpt.pt"),
    ]
    for tag, path in targets:
        payload = build()
        med, allv = timed(lambda: torch.save(payload, path), args.repeats)
        size = os.path.getsize(path) / 2**20
        print(
            f"torch.save GPU-resident -> {tag:18s} median={med:6.2f} s  "
            f"{[f'{x:.2f}' for x in allv]}  {size:.0f} MB  {size/med:.0f} MB/s"
        )
        os.remove(path)

    # What the trainer actually pays per epoch: best.pt + latest.pt
    payload = build()
    med, _ = timed(lambda: (torch.save(payload, "outputs/_b1.pt"),
                            torch.save(payload, "outputs/_b2.pt")), args.repeats)
    print(f"\ntrainer per-epoch cost (2 saves to quarkfs) median={med:.2f} s")
    for p in ("outputs/_b1.pt", "outputs/_b2.pt"):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
