"""How much speed does max_frames 16 -> 13 actually buy?

The knob is only sound under stem_norm=group: with BatchNorm the padded
all-zero frame slots enter the batch statistics, so dropping slots changes the
math (measured |dloss| 8.3e-02) even when the dropped slots are provably empty.
Under GroupNorm the same change moves the loss by 9.5e-07, i.e. it becomes a
pure performance knob.

This script measures the prize, so the stem_norm decision can be made against
a number rather than a hunch. Both arms use compile + fast_attention, i.e. the
configuration that is now the default, so the gain reported here is what would
be added on top of what is already landed -- not a standalone figure.
"""
import argparse
import statistics
import time
import types

import torch
import yaml

from src.data.dataset import YajiangAEFDataset
from src.models.model import AEFModel
from src.training.losses import compute_total_loss


def ns(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [ns(v) for v in d]
    return d


def load_cfg(path):
    return ns(yaml.safe_load(open(path)))


def move(x, dev):
    if torch.is_tensor(x):
        return x.to(dev, non_blocking=True)
    if isinstance(x, dict):
        return {k: move(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move(v, dev) for v in x)
    return x


def get_batch(cfg, dev, manifest):
    from torch.utils.data import DataLoader

    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    dl = DataLoader(ds, batch_size=cfg.data.batch_size, num_workers=2, shuffle=False)
    return move(next(iter(dl)), dev)


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
    def run():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**fwd_kwargs(batch))
            loss = compute_total_loss(out, batch, cfg)
        loss.total.backward()
        opt.step()
    return run


def timed(fn, warmup, repeats, inner):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / inner * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def occupancy(cfg, manifest, n=256):
    """How many of the max_frames slots per source are actually filled?

    Answers whether 13 is even the right number, and what the ceiling is.
    """
    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    n = min(n, len(ds))
    per_src = {}
    for i in range(n):
        fm = ds[i]["source_frame_mask"]    # [S, T]
        for s, name in enumerate(cfg.data.input_sources):
            per_src.setdefault(name, []).append(int(fm[s].sum()))
    print(f"frame occupancy over {n} patches (max_frames={cfg.data.max_frames}):")
    worst = 0
    for name, v in per_src.items():
        print(f"  {name:9s} min {min(v):3d}  median {int(statistics.median(v)):3d}  "
              f"max {max(v):3d}  mean {statistics.mean(v):5.1f}")
        worst = max(worst, max(v))
    print(f"  -> no patch needs more than {worst} slots; "
          f"{cfg.data.max_frames - worst} are always empty")
    return worst


def arm(cfg_path, manifest, frames, norm, compile_mode, dev):
    cfg = load_cfg(cfg_path)
    cfg.data.max_frames = frames
    cfg.model.stem_norm = norm
    batch = get_batch(cfg, dev, manifest)
    torch.manual_seed(0)
    model = AEFModel(cfg).to(dev)
    if compile_mode:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    med, lo, hi = timed(step_fn(model, batch, cfg, opt), warmup=3, repeats=3, inner=2)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, batch
    torch.cuda.empty_cache()
    return med, hi - lo, peak


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--mode", default="default", help="compile mode, or 'eager'")
    args = p.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    base = load_cfg(args.config)
    print("=" * 76)
    print("A. is 13 the right number?")
    print("=" * 76)
    worst = occupancy(base, args.manifest)

    mode = None if args.mode == "eager" else args.mode
    print()
    print("=" * 76)
    print(f"B. what does the knob buy? (compile={args.mode}, "
          f"fast_attention={base.model.fast_attention})")
    print("=" * 76)
    arms = [
        ("max_frames=16  stem_norm=batch  [current default]", 16, "batch"),
        ("max_frames=16  stem_norm=group", 16, "group"),
        ("max_frames=13  stem_norm=group  [the knob]", 13, "group"),
    ]
    if worst < 13:
        arms.append((f"max_frames={worst}   stem_norm=group  [tightest safe]",
                     worst, "group"))
    rows, ref = [], None
    for label, frames, norm in arms:
        med, spread, peak = arm(args.config, args.manifest, frames, norm, mode, dev)
        if ref is None:
            ref = med
        rows.append((label, med, spread, peak, ref / med))
        print(f"  ran {label}")
    print()
    print(f"{'variant':50s} {'step ms':>8s} {'spread':>7s} {'peak GB':>8s} {'speedup':>8s}")
    for label, med, spread, peak, sp in rows:
        print(f"{label:50s} {med:8.1f} {spread:7.1f} {peak:8.2f} {sp:7.3f}x")
    print()
    print("spread is max-min of 3 repeats; a difference smaller than either")
    print("arm's spread carries no information.")


if __name__ == "__main__":
    main()
