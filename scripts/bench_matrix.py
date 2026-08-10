#!/usr/bin/env python3
"""Four-way matrix: eager / compile / compile+fast_attn / +group_norm.

Two things this answers that the individual measurements could not:

  1. Do compile and fast_attention stack, or overlap? torch.compile may already
     fuse away nn.MultiheadAttention's plumbing, in which case the 2.5-5x
     isolated attention win is not additive with compile's 1.65x.
  2. Is the compiled-vs-eager loss gap float accumulation order, or RNG
     divergence? The vMF bottleneck injects randn in both train and eval
     branches, so kappa=0 is used to remove sampling as a confound.
"""
from __future__ import annotations

import argparse
import copy
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


def move(v):
    if torch.is_tensor(v):
        return v.cuda()
    if isinstance(v, dict):
        return {k: move(x) for k, x in v.items()}
    return v


def fwd_kwargs(b):
    keys = ("source_frames", "source_timestamps_ms", "source_frame_mask",
            "source_input_mask", "source_type_ids", "valid_start_ms",
            "valid_end_ms", "target_relative_time", "target_metadata")
    return {k: b[k] for k in keys}


def timed(fn, warmup=3, repeats=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out), min(out), max(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    ap.add_argument("--manifest", default="data/full_npy/train.jsonl")
    ap.add_argument("--mode", default="default",
                    help="compile mode; 'default' keeps the matrix affordable")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    base = yaml.safe_load(open(args.config))
    cfg = ns(base)

    from src.models.model import AEFModel
    from src.training.losses import compute_total_loss
    from src.data.dataset import YajiangAEFDataset, aef_collate_fn
    from torch.utils.data import DataLoader

    ds = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=2,
                    shuffle=False, collate_fn=aef_collate_fn, drop_last=True)
    batch = move(next(iter(dl)))

    def build(compile_mode=None, **over):
        c = copy.deepcopy(base)
        c["model"].update(over)
        cc = ns(c)
        torch.manual_seed(cc.experiment.seed)
        m = AEFModel(cc).cuda().train()
        if compile_mode:
            m = torch.compile(m, mode=compile_mode)
        opt = torch.optim.AdamW(m.parameters(), lr=cc.training.lr,
                                weight_decay=cc.training.weight_decay)
        return m, opt, cc

    def step(m, opt, cc):
        def run():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=cc.training.amp):
                out = m(**fwd_kwargs(batch))
                loss = compute_total_loss(out, batch, cc)
            loss.total.backward()
            opt.step()
        return run

    # ---------------- part A: numerics, RNG removed ----------------
    print("=" * 76)
    print("A. compiled vs eager numerics, with and without vMF sampling")
    print("=" * 76)
    print("kappa=0 disables the randn injection in bottleneck.py, leaving only")
    print("float accumulation order as a source of difference.\n")
    for kappa, tag in ((0.0, "kappa=0   deterministic"),
                       (base["model"]["vmf_kappa"], "kappa=2000 stochastic")):
        c = copy.deepcopy(base)
        c["model"]["vmf_kappa"] = kappa
        cc = ns(c)
        torch.manual_seed(cc.experiment.seed)
        m = AEFModel(cc).cuda().eval()
        sd = copy.deepcopy(m.state_dict())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(7)
            torch.cuda.manual_seed_all(7)
            oe = m(**fwd_kwargs(batch))
            le = float(compute_total_loss(oe, batch, cc).total)

        torch._dynamo.reset()
        torch.manual_seed(cc.experiment.seed)
        m2 = AEFModel(cc).cuda().eval()
        m2.load_state_dict(sd)          # load BEFORE compiling
        m2 = torch.compile(m2, mode=args.mode)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(7)
            torch.cuda.manual_seed_all(7)
            oc = m2(**fwd_kwargs(batch))
            lc = float(compute_total_loss(oc, batch, cc).total)

        rel = abs(le - lc) / max(abs(le), 1e-12)
        demb = (oe.embedding_map - oc.embedding_map).abs().max().item()
        print(f"{tag:26s} eager {le:9.6f}  compiled {lc:9.6f}  "
              f"rel {rel:.3e}  emb max {demb:.3e}")
        del m, m2, oe, oc
        torch.cuda.empty_cache()
    print("\nbf16 has an 8-bit mantissa -> relative resolution 2^-8 = 3.9e-3")

    # ---------------- part B: speed matrix ----------------
    print("\n" + "=" * 76)
    print(f"B. speed and memory matrix (compile mode = {args.mode})")
    print("=" * 76)
    variants = [
        ("eager baseline",            dict(), None),
        ("eager + fast_attn",         dict(fast_attention=True), None),
        ("compile",                   dict(), args.mode),
        ("compile + fast_attn",       dict(fast_attention=True), args.mode),
        ("compile + fast + group",    dict(fast_attention=True,
                                           stem_norm="group"), args.mode),
    ]
    print(f"{'variant':26s} {'step ms':>9s} {'spread':>8s} {'peak GB':>9s} "
          f"{'vs eager':>9s} {'vs compile':>11s}")
    base_t = comp_t = None
    for label, over, mode in variants:
        torch._dynamo.reset()
        m, opt, cc = build(compile_mode=mode, **over)
        torch.cuda.reset_peak_memory_stats()
        run = step(m, opt, cc)
        try:
            med, lo, hi = timed(run, warmup=5 if mode else 3, repeats=3)
        except Exception as exc:
            print(f"{label:26s} FAILED {type(exc).__name__}: {str(exc)[:60]}")
            del m, opt
            torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_allocated() / 2**30
        if label == "eager baseline":
            base_t = med
        if label == "compile":
            comp_t = med
        v1 = f"{base_t/med:8.3f}x" if base_t else "        -"
        v2 = f"{comp_t/med:10.3f}x" if comp_t else "          -"
        print(f"{label:26s} {med*1e3:9.1f} {(hi-lo)*1e3:8.1f} {peak:9.2f} "
              f"{v1:>9s} {v2:>11s}")
        del m, opt
        torch.cuda.empty_cache()
    print("\nspread is max-min of 3 repeats; a difference smaller than the")
    print("spread of either arm carries no information.")


if __name__ == "__main__":
    main()
