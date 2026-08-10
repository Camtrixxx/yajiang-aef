#!/usr/bin/env python3
"""Deep observation of one AEF training step.

Four questions the earlier profiling left open:
  1. MFU -- how far is the step from the A800 roofline, and how much of the
     kernel time does zero-FLOP data movement eat?
  2. Which STPBlock path (precision / time / space / fusion) costs what, at
     the real tensor shapes rather than in the abstract?
  3. Where do the `copy_` kernels (22.75% of kernel time) actually come from?
  4. Does torch.compile help? Never measured on this model.

Every timing does warmup + synchronize + 3 repeats and reports the median,
because a single un-synchronized reading measures queue depth, not work.
"""
from __future__ import annotations

import argparse
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


def load_cfg(path):
    return ns(yaml.safe_load(open(path)))


def timed(fn, warmup=3, repeats=3, inner=1):
    """Median wall time of `fn`, with the GPU actually drained first."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) / inner)
    return statistics.median(out), min(out), max(out)


def get_batch(cfg, manifest, device):
    from src.data.dataset import YajiangAEFDataset
    from torch.utils.data import DataLoader

    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=2, shuffle=False)
    batch = next(iter(dl))

    def move(v):
        if torch.is_tensor(v):
            return v.to(device)
        if isinstance(v, dict):
            return {k: move(x) for k, x in v.items()}
        return v

    return move(batch)


def fwd_kwargs(batch):
    return dict(
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


def step_fn(model, batch, cfg, opt):
    """One full training step, matching the trainer's autocast setting."""
    from src.training.losses import compute_total_loss

    def run():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.training.amp):
            out = model(**fwd_kwargs(batch))
            loss = compute_total_loss(out, batch, cfg)
        loss.total.backward()
        opt.step()

    return run


# --------------------------------------------------------------------------
# 1. FLOP accounting and MFU
# --------------------------------------------------------------------------
A800_BF16_TFLOPS = 312.0  # dense tensor-core peak, no sparsity


def section_flops(model, batch, cfg, opt):
    from torch.utils.flop_counter import FlopCounterMode
    from src.training.losses import compute_total_loss

    print("\n" + "=" * 74)
    print("1. FLOP accounting / MFU")
    print("=" * 74)

    # Count under autocast, matching the real step: FLOP counts are
    # dtype-independent, but fp32 activations would double the footprint and OOM.
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.training.amp)

    ctr = FlopCounterMode(display=False, depth=None)
    with torch.no_grad(), ctr, ac:
        out = model(**fwd_kwargs(batch))
        compute_total_loss(out, batch, cfg)
    fwd_flops = ctr.get_total_flops()
    del out
    torch.cuda.empty_cache()

    ctr2 = FlopCounterMode(display=False, depth=None)
    with ctr2:
        with ac:
            out = model(**fwd_kwargs(batch))
            loss = compute_total_loss(out, batch, cfg)
        loss.total.backward()
    tot_flops = ctr2.get_total_flops()
    del out, loss
    opt.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    med, lo, hi = timed(step_fn(model, batch, cfg, opt))
    achieved = tot_flops / med / 1e12
    print(f"forward FLOPs        {fwd_flops/1e12:8.3f} T")
    print(f"fwd+bwd FLOPs        {tot_flops/1e12:8.3f} T   "
          f"(bwd/fwd = {(tot_flops-fwd_flops)/max(fwd_flops,1):.2f}x)")
    print(f"step time            {med*1e3:8.1f} ms  (min {lo*1e3:.1f} max {hi*1e3:.1f})")
    print(f"achieved             {achieved:8.1f} TFLOPS")
    print(f"MFU vs A800 bf16     {100*achieved/A800_BF16_TFLOPS:8.1f} %  "
          f"(peak {A800_BF16_TFLOPS:.0f} TFLOPS, dense)")
    return {"flops": tot_flops, "step": med, "counts": ctr2.get_flop_counts()}


# --------------------------------------------------------------------------
# 2. Path-level cost inside one STPBlock, at the real shapes
# --------------------------------------------------------------------------
def section_paths(model, batch, cfg):
    print("\n" + "=" * 74)
    print("2. STPBlock path-level cost (real shapes, fwd and fwd+bwd)")
    print("=" * 74)

    blk = model.stp_blocks[0]
    b = cfg.data.batch_size
    t_all = len(cfg.data.input_sources) * cfg.data.max_frames
    c = cfg.model.precision_dim
    h = w = cfg.data.image_size // 2
    x = torch.randn(b, t_all, c, h, w, device="cuda")
    mask = torch.ones(b, t_all, dtype=torch.bool, device="cuda")
    print(f"input to each block: {tuple(x.shape)}  "
          f"= {x.numel()*4/2**30:.2f} GB in fp32\n")

    x_flat = x.reshape(b * t_all, c, h, w)

    def prec():
        return blk.precision_conv(x_flat)

    def time_path():
        y = blk.time_down(x_flat)
        _, ct, ht, wt = y.shape
        y = y.reshape(b, t_all, ct, ht * wt).permute(0, 3, 1, 2).reshape(b * ht * wt, t_all, ct)
        kpm = (~mask).repeat_interleave(ht * wt, dim=0)
        y, _ = blk.time_attn(y, y, y, key_padding_mask=kpm)
        y = blk.time_norm(y)
        y = y.reshape(b, ht * wt, t_all, ct).permute(0, 2, 3, 1).reshape(b * t_all, ct, ht, wt)
        return blk.time_up(y)

    def space_path():
        y = blk.space_down(x_flat)
        _, cs, hs, ws = y.shape
        f = y.reshape(b, t_all, cs, hs * ws).permute(0, 1, 3, 2).reshape(b * t_all, hs * ws, cs)
        f, _ = blk.space_attn(f, f, f)
        f = blk.space_norm(f)
        f = f.reshape(b, t_all, hs, ws, cs).permute(0, 1, 4, 2, 3).reshape(b * t_all, cs, hs, ws)
        return blk.space_up(f)

    def fusion():
        return blk.fusion(torch.cat([x_flat, x_flat, x_flat], dim=1))

    def full_block():
        return blk(x, mask)

    def bwd_of(fn):
        def run():
            out = fn()
            out.sum().backward()
            blk.zero_grad(set_to_none=True)
        return run

    xg = x.detach().requires_grad_(True)
    rows = [("precision_conv", prec), ("time path", time_path),
            ("space path", space_path), ("fusion(1x1)", fusion),
            ("FULL BLOCK", full_block)]
    print(f"{'path':18s} {'fwd ms':>9s} {'fwd+bwd ms':>11s} {'bwd/fwd':>8s} {'% of block':>11s}")
    fwd_all = {}
    tot_all = {}
    for name, fn in rows:
        with torch.no_grad():
            f_med, _, _ = timed(fn)
        t_med, _, _ = timed(bwd_of(fn))
        fwd_all[name] = f_med
        tot_all[name] = t_med
    blk_tot = tot_all["FULL BLOCK"]
    for name, _ in rows:
        f_med, t_med = fwd_all[name], tot_all[name]
        share = "" if name == "FULL BLOCK" else f"{100*t_med/blk_tot:10.1f}%"
        print(f"{name:18s} {f_med*1e3:9.2f} {t_med*1e3:11.2f} "
              f"{t_med/f_med:8.2f} {share:>11s}")
    n_blocks = cfg.model.num_blocks
    print(f"\nsum of 4 paths      {sum(tot_all[n] for n,_ in rows[:-1])*1e3:9.2f} ms "
          f"vs full block {blk_tot*1e3:.2f} ms")
    print(f"x{n_blocks} blocks           {blk_tot*n_blocks*1e3:9.2f} ms")
    del x, xg
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 3. Attribute the data-movement kernels to source lines
# --------------------------------------------------------------------------
def section_copies(model, batch, cfg, opt):
    from torch.profiler import profile, ProfilerActivity
    from collections import defaultdict

    print("\n" + "=" * 74)
    print("3. Where the zero-FLOP kernel time goes")
    print("=" * 74)

    run = step_fn(model, batch, cfg, opt)
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        run()
        torch.cuda.synchronize()

    evts = prof.key_averages()
    # aten:: entries and raw CUDA kernel names are two views of the same time.
    # Summing both double-counts, so keep them apart.
    kern = [e for e in evts if not e.key.startswith(("aten::", "autograd::",
                                                     "cudaLaunch", "Optimizer",
                                                     "torch/", "nn.Module",
                                                     "ProfilerStep"))]
    kern = [e for e in kern if e.self_device_time_total > 0]
    kern_total = sum(e.self_device_time_total for e in kern) or 1.0

    # Kernels that only move bytes. Names verified against this run's output:
    # cutlass_*/sm80_xmma_* are real GEMMs and must NOT be counted here.
    MOVE = ("nchwtonhwc", "nhwctonchw", "elementwise_copy", "direct_copy",
            "copy_device_to_device", "catarraybatchedcopy", "transpose",
            "permute", "unrolled_elementwise")

    def is_move(name: str) -> bool:
        n = name.lower()
        if "cutlass" in n or "xmma" in n or "implicit_gemm" in n or "wgrad" in n:
            return False
        return any(m in n for m in MOVE)

    print(f"--- CUDA kernel view (self time, no double counting) ---")
    print(f"{'kernel':52s} {'ms':>9s} {'%':>7s} {'calls':>7s}")
    move_ms = 0.0
    for e in kern:
        if is_move(e.key):
            move_ms += e.self_device_time_total / 1e3
    for e in sorted(kern, key=lambda x: -x.self_device_time_total)[:16]:
        tag = " <-- move" if is_move(e.key) else ""
        print(f"{e.key[:52]:52s} {e.self_device_time_total/1e3:9.2f} "
              f"{100*e.self_device_time_total/kern_total:6.2f}% {e.count:7d}{tag}")
    print(f"\nkernel self time total  {kern_total/1e3:8.1f} ms")
    print(f"pure data movement      {move_ms:8.1f} ms = "
          f"{100*move_ms/(kern_total/1e3):.1f}% of kernel time")

    print(f"\n--- aten:: op view ---")
    aten = [e for e in evts if e.key.startswith("aten::") and e.self_device_time_total > 0]
    aten_total = sum(e.self_device_time_total for e in aten) or 1.0
    print(f"{'op':40s} {'ms':>9s} {'%':>7s} {'calls':>7s}")
    for e in sorted(aten, key=lambda x: -x.self_device_time_total)[:14]:
        print(f"{e.key[:40]:40s} {e.self_device_time_total/1e3:9.2f} "
              f"{100*e.self_device_time_total/aten_total:6.2f}% {e.count:7d}")

    # Attribute copy_ to source lines via the recorded python stack.
    by_src = defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.key != "aten::copy_" or not getattr(e, "stack", None):
            continue
        frames = [f for f in e.stack if "yajiang-aef" in f and "observe_" not in f]
        site = frames[0] if frames else "<no project frame>"
        dt = getattr(e, "self_device_time_total", 0) or 0
        by_src[site.split("yajiang-aef/")[-1]][0] += dt / 1e3
        by_src[site.split("yajiang-aef/")[-1]][1] += 1
    if by_src:
        print("\n--- aten::copy_ attributed to source line ---")
        print(f"{'site':58s} {'ms':>8s} {'calls':>7s}")
        for k, (ms, n) in sorted(by_src.items(), key=lambda kv: -kv[1][0])[:14]:
            print(f"{k[:58]:58s} {ms:8.2f} {n:7d}")
    return prof


# --------------------------------------------------------------------------
# 5. Micro-bench every candidate copy site at the real shapes.
#    torch 2.13 records no python stack for aten::copy_, so attribute by
#    direct measurement instead of by inference.
# --------------------------------------------------------------------------
def section_copy_sites(cfg):
    print("\n" + "=" * 74)
    print("5. Candidate copy sites, measured directly")
    print("=" * 74)

    b = cfg.data.batch_size
    s = len(cfg.data.input_sources)
    t = cfg.data.max_frames
    t_all = s * t
    c = cfg.model.precision_dim
    h = w = cfg.data.image_size // 2
    raw_c = max(cfg.model.source_channels.__dict__.values())
    dev = "cuda"

    # sensor_encoders.py: boolean-mask index + channel slice + fold t into batch
    frames = torch.randn(b, s, t, raw_c, cfg.data.image_size,
                         cfg.data.image_size, device=dev)
    bmask = torch.tensor([True, True, False, False], device=dev)
    in_ch = cfg.model.source_channels.s2

    # blocks.py: the permute -> reshape round trips
    xt = torch.randn(b, t_all, c, h // 2, w // 2, device=dev)          # time path
    xs = torch.randn(b, t_all, c, h // 4, w // 4, device=dev)          # space path
    xflat = torch.randn(b * t_all, c, h, w, device=dev)                # fusion input
    fmask = torch.ones(b, t_all, dtype=torch.bool, device=dev)
    ht = wt = h // 2
    hs = ws = h // 4

    sites = [
        ("sensor: frames[bmask, s_idx]",
         lambda: frames[bmask, 0]),
        ("sensor: [:, :, :in_ch] slice",
         lambda: frames[:, 0, :, :in_ch].contiguous()),
        ("sensor: reshape(b*t,c,h,w)",
         lambda: frames[:, 0].reshape(b * t, raw_c, cfg.data.image_size,
                                      cfg.data.image_size)),
        ("blocks:91 time permute->reshape",
         lambda: xt.reshape(b, t_all, c, ht * wt).permute(0, 3, 1, 2)
                   .reshape(b * ht * wt, t_all, c)),
        ("blocks:104 time permute back",
         lambda: xt.reshape(b, ht * wt, t_all, c).permute(0, 2, 3, 1)
                   .reshape(b * t_all, c, ht, wt)),
        ("blocks:97 repeat_interleave mask",
         lambda: (~fmask).repeat_interleave(ht * wt, dim=0)),
        ("blocks:111 space permute->reshape",
         lambda: xs.reshape(b, t_all, c, hs * ws).permute(0, 1, 3, 2)
                   .reshape(b * t_all, hs * ws, c)),
        ("blocks:115 space permute back",
         lambda: xs.reshape(b, t_all, hs, ws, c).permute(0, 1, 4, 2, 3)
                   .reshape(b * t_all, c, hs, ws)),
        ("blocks:119 cat 3x on dim=1",
         lambda: torch.cat([xflat, xflat, xflat], dim=1)),
        ("model:134 reshape(b,s*t,...)",
         lambda: torch.randn(b, s, t, c, h, w, device=dev).reshape(b, t_all, c, h, w)),
    ]

    print(f"{'site':36s} {'ms':>8s} {'x/block':>8s} {'GB moved':>9s}")
    per_block = 0.0
    for name, fn in sites:
        med, _, _ = timed(fn, warmup=5, repeats=5)
        out = fn()
        gb = (out.numel() * out.element_size()) / 2**30 if torch.is_tensor(out) else 0
        n_per_block = 4 if name.startswith("blocks") else 0
        if n_per_block:
            per_block += med
        print(f"{name:36s} {med*1e3:8.3f} {n_per_block if n_per_block else '-':>8} {gb:9.3f}")
    nb = cfg.model.num_blocks
    print(f"\nblocks.py sites, x{nb} blocks: {per_block*nb*1e3:.2f} ms per step")


# --------------------------------------------------------------------------
# 6. The two remaining suspects for aten::copy_:
#    MultiheadAttention internal plumbing, and autocast dtype casts.
# --------------------------------------------------------------------------
def section_copy_suspects(model, batch, cfg, opt):
    from torch.profiler import profile, ProfilerActivity

    print("\n" + "=" * 74)
    print("6. MultiheadAttention plumbing vs autocast casts")
    print("=" * 74)

    b = cfg.data.batch_size
    t_all = len(cfg.data.input_sources) * cfg.data.max_frames
    c = cfg.model.precision_dim
    hw_t = (cfg.data.image_size // 4) ** 2      # time attn: one seq per pixel
    hw_s = (cfg.data.image_size // 8) ** 2      # space attn: pixels are the seq
    mha = model.stp_blocks[0].time_attn

    print("attention shapes actually used:")
    print(f"  time_attn : {b*hw_t:6d} seqs x len {t_all:4d} x dim {c}")
    print(f"  space_attn: {b*t_all:6d} seqs x len {hw_s:4d} x dim {c}")

    qt = torch.randn(b * hw_t, t_all, c, device="cuda")
    qs = torch.randn(b * t_all, hw_s, c, device="cuda")
    for label, q in (("time_attn fwd", qt), ("space_attn fwd", qs)):
        with torch.no_grad():
            med, _, _ = timed(lambda: mha(q, q, q))
        # Same math via explicit SDPA, no nn.MultiheadAttention plumbing.
        nh = cfg.model.num_heads
        n, L, _ = q.shape
        qh = q.reshape(n, L, nh, c // nh).transpose(1, 2)
        with torch.no_grad():
            med2, _, _ = timed(
                lambda: torch.nn.functional.scaled_dot_product_attention(qh, qh, qh))
        print(f"{label:16s} nn.MHA {med*1e3:7.2f} ms   raw SDPA {med2*1e3:7.2f} ms   "
              f"plumbing overhead {med/max(med2,1e-9):5.1f}x")

    # How much copy_ time is autocast casting? Compare amp on vs off.
    print("\namp on/off split of aten::copy_ (same step, same shapes):")
    for amp_on in (True, False):
        run = step_fn(model, batch, cfg, opt) if amp_on else None
        if not amp_on:
            from src.training.losses import compute_total_loss

            def run():
                opt.zero_grad(set_to_none=True)
                out = model(**fwd_kwargs(batch))
                compute_total_loss(out, batch, cfg).total.backward()
                opt.step()
        try:
            for _ in range(2):
                run()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
                run()
                torch.cuda.synchronize()
            ev = {e.key: e for e in p.key_averages()}
            cp = ev.get("aten::copy_")
            tc = ev.get("aten::_to_copy")
            step, _, _ = timed(run, warmup=2, repeats=3)
            print(f"  amp={'bf16' if amp_on else 'fp32':4s} step {step*1e3:7.1f} ms  "
                  f"copy_ {(cp.self_device_time_total/1e3 if cp else 0):7.2f} ms "
                  f"({cp.count if cp else 0:4d} calls)  "
                  f"_to_copy {(tc.self_device_time_total/1e3 if tc else 0):6.2f} ms")
        except torch.OutOfMemoryError:
            print(f"  amp={'bf16' if amp_on else 'fp32'}: OOM (fp32 activations "
                  f"do not fit -- itself a finding)")
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 4. torch.compile A/B -- never measured on this model
# --------------------------------------------------------------------------
def section_compile(cfg, batch, manifest, args):
    from src.models.model import AEFModel

    print("\n" + "=" * 74)
    print("4. torch.compile A/B")
    print("=" * 74)

    results = {}
    for label, mode in (("eager", None), ("compile default", "default"),
                        ("compile max-autotune", "max-autotune-no-cudagraphs")):
        torch.manual_seed(cfg.experiment.seed)
        m = AEFModel(cfg).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=cfg.training.lr)
        if mode is not None:
            m = torch.compile(m, mode=mode)
        torch.cuda.reset_peak_memory_stats()
        run = step_fn(m, batch, cfg, opt)
        try:
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            warm = time.perf_counter() - t0
            med, lo, hi = timed(run, warmup=5, repeats=3)
        except Exception as exc:  # a compile failure is a result, not a crash
            print(f"{label:22s} FAILED: {type(exc).__name__}: {str(exc)[:90]}")
            del m, opt
            torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_allocated() / 2**30
        results[label] = (med, peak)
        base = results.get("eager", (med, peak))[0]
        print(f"{label:22s} {med*1e3:8.1f} ms  spread {(hi-lo)*1e3:5.1f} ms  "
              f"peak {peak:6.2f} GB  speedup {base/med:5.3f}x  "
              f"first-call {warm:6.1f} s")
        del m, opt
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    ap.add_argument("--manifest", default="data/full_npy/train.jsonl")
    ap.add_argument("--sections", default="1,2,3,4")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    cfg = load_cfg(args.config)
    want = set(args.sections.split(","))

    from src.models.model import AEFModel
    batch = get_batch(cfg, args.manifest, "cuda")
    torch.manual_seed(cfg.experiment.seed)
    model = AEFModel(cfg).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    print(f"config {args.config}  batch_size={cfg.data.batch_size}  "
          f"max_frames={cfg.data.max_frames}  image_size={cfg.data.image_size}")
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f} M  "
          f"T_all={len(cfg.data.input_sources)*cfg.data.max_frames}")

    if "1" in want:
        section_flops(model, batch, cfg, opt)
    if "2" in want:
        section_paths(model, batch, cfg)
    if "3" in want:
        section_copies(model, batch, cfg, opt)
    if "5" in want:
        section_copy_sites(cfg)
    if "6" in want:
        section_copy_suspects(model, batch, cfg, opt)
    if "4" in want:
        del model, opt
        torch.cuda.empty_cache()
        section_compile(cfg, batch, args.manifest, args)


if __name__ == "__main__":
    main()
