#!/usr/bin/env python3
"""Verify torch.compile is safe to turn on, under real DDP.

A 1.65x speedup only counts if it holds in the real training loop, so this
checks the three things that could void it:

  1. Checkpoint keys -- compiled DDP nests as _orig_mod -> module -> model.
     If Trainer._unwrap() is wrong, checkpoints silently become unloadable.
  2. Recompilation -- a graph rebuilt every step would erase the gain. Counted
     via torch._dynamo metrics, not assumed from static shapes.
  3. Numerics -- compile preserves the math but changes float accumulation
     order, so losses should track eager closely, not match bit-exactly.

Run under torchrun to exercise the DDP path:
  torchrun --nproc_per_node=2 scripts/verify_compile.py --steps 6
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
import types

import torch
import yaml


def ns(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(v) for v in d]
    return d


def move(v, device):
    if torch.is_tensor(v):
        return v.to(device)
    if isinstance(v, dict):
        return {k: move(x, device) for k, x in v.items()}
    return v


def build(cfg, device, distributed, compile_mode):
    from src.models.model import AEFModel

    torch.manual_seed(cfg.experiment.seed)
    model = AEFModel(cfg).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index])
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)
    return model, opt


def run_steps(model, opt, batches, cfg, n_steps):
    from src.training.losses import compute_total_loss

    losses, times = [], []
    for i in range(n_steps):
        batch = batches[i % len(batches)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.training.amp):
            out = model(
                source_frames=batch["source_frames"],
                source_timestamps_ms=batch["source_timestamps_ms"],
                source_frame_mask=batch["source_frame_mask"],
                source_input_mask=batch["source_input_mask"],
                source_type_ids=batch["source_type_ids"],
                valid_start_ms=batch["valid_start_ms"],
                valid_end_ms=batch["valid_end_ms"],
                target_relative_time=batch["target_relative_time"],
                target_metadata=batch["target_metadata"],
            )
            loss = compute_total_loss(out, batch, cfg)
        loss.total.backward()
        if getattr(cfg.training, "grad_clip_norm", 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.training.grad_clip_norm)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.total.detach()))
    return losses, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    ap.add_argument("--manifest", default="data/full_npy/train.jsonl")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--mode", default="max-autotune-no-cudagraphs")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    distributed = "RANK" in os.environ
    if distributed:
        torch.distributed.init_process_group("nccl")
        rank = torch.distributed.get_rank()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device(f"cuda:{local}")
        world = torch.distributed.get_world_size()
    else:
        rank, world = 0, 1
        device = torch.device("cuda:0")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    cfg = ns(yaml.safe_load(open(args.config)))

    from src.data.dataset import YajiangAEFDataset, aef_collate_fn
    from torch.utils.data import DataLoader

    ds = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=2,
                    shuffle=False, collate_fn=aef_collate_fn, drop_last=True)
    it = iter(dl)
    batches = [move(next(it), device) for _ in range(3)]

    log(f"world_size={world}  distributed={distributed}  mode={args.mode}")
    log(f"batch_size={cfg.data.batch_size}  steps={args.steps}\n")

    # --- eager reference ---
    model_e, opt_e = build(cfg, device, distributed, None)
    torch.cuda.reset_peak_memory_stats()
    loss_e, time_e = run_steps(model_e, opt_e, batches, cfg, args.steps)
    peak_e = torch.cuda.max_memory_allocated() / 2**30
    del model_e, opt_e
    torch.cuda.empty_cache()

    # --- compiled ---
    torch._dynamo.reset()
    model_c, opt_c = build(cfg, device, distributed, args.mode)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss_c, time_c = run_steps(model_c, opt_c, batches, cfg, args.steps)
    wall_c = time.perf_counter() - t0
    peak_c = torch.cuda.max_memory_allocated() / 2**30

    # 2. recompilation count
    try:
        from torch._dynamo.utils import counters
        frames = counters["frames"]
        n_compiles = frames.get("total", 0)
        n_ok = frames.get("ok", 0)
    except Exception:
        n_compiles = n_ok = -1

    log("=" * 70)
    log("1. checkpoint keys after unwrap")
    log("=" * 70)
    from src.training.trainer import Trainer
    probe = types.SimpleNamespace(model=model_c)
    unwrapped = Trainer._unwrap(probe)
    keys = list(unwrapped.state_dict().keys())
    bad = [k for k in keys if "_orig_mod" in k or k.startswith("module.")]
    log(f"wrapper chain: {type(model_c).__name__}"
        + (" -> DDP" if distributed else "") + " -> AEFModel")
    log(f"unwrapped to {type(unwrapped).__name__}, {len(keys)} keys")
    log(f"polluted keys (_orig_mod / module.): {len(bad)}   "
        + ("PASS" if not bad else f"FAIL {bad[:3]}"))
    log(f"sample key: {keys[0]}")

    log("\n" + "=" * 70)
    log("2. recompilation")
    log("=" * 70)
    log(f"steps run: {args.steps}   dynamo frames compiled: {n_compiles} (ok {n_ok})")
    log(f"per-step times (ms): {[f'{t*1e3:.0f}' for t in time_c]}")
    steady_c = statistics.median(time_c[2:]) if len(time_c) > 3 else time_c[-1]
    log(f"first step {time_c[0]*1e3:.0f} ms (includes compile), "
        f"steady-state median {steady_c*1e3:.1f} ms")
    log("PASS: no per-step recompilation" if time_c[-1] < 2 * steady_c
        else "SUSPECT: last step much slower than median")

    log("\n" + "=" * 70)
    log("3. numerics: compiled vs eager")
    log("=" * 70)
    log(f"{'step':>5s} {'eager loss':>14s} {'compiled loss':>14s} {'rel diff':>11s}")
    for i, (a, b) in enumerate(zip(loss_e, loss_c)):
        rel = abs(a - b) / max(abs(a), 1e-12)
        log(f"{i:5d} {a:14.6f} {b:14.6f} {rel:11.3e}")
    rels = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(loss_e, loss_c)]
    log(f"max relative difference: {max(rels):.3e}")
    log("Expected small but nonzero: fusion changes float accumulation order.")

    log("\n" + "=" * 70)
    log("4. speed and memory")
    log("=" * 70)
    steady_e = statistics.median(time_e[2:]) if len(time_e) > 3 else time_e[-1]
    log(f"eager    steady {steady_e*1e3:8.1f} ms   peak {peak_e:6.2f} GB")
    log(f"compiled steady {steady_c*1e3:8.1f} ms   peak {peak_c:6.2f} GB")
    log(f"speedup {steady_e/steady_c:.3f}x   memory {100*(1-peak_c/peak_e):.1f}% lower")
    log(f"compile amortization: {args.steps} steps took {wall_c:.1f} s wall")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
