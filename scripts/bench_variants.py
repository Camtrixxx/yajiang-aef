"""Measure step time and peak memory for candidate training-speed changes.

Each variant only touches config or tensor layout, so results are directly
comparable against the v1.2 baseline. Loader is kept out of the timing loop:
profile_training.py section C already showed the step is compute-bound.

  CUDA_VISIBLE_DEVICES=5 python scripts/bench_variants.py
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time

import torch

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.losses import compute_total_loss


def _forward(model, batch):
    return model(
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


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def build_batch(cfg, manifest: str, device, batch_size: int):
    """Build one batch synchronously so no worker processes pollute timing."""
    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
    samples = [dataset[i] for i in range(batch_size)]
    return _to_device(aef_collate_fn(samples), device)


def run_variant(
    name: str,
    cfg,
    manifest: str,
    device,
    steps: int,
    warmup: int,
    channels_last: bool = False,
    batch_size: int | None = None,
) -> dict | None:
    batch_size = batch_size or int(cfg.data.batch_size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        batch = build_batch(cfg, manifest, device, batch_size)
        model = AEFModel(cfg).to(device)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
        )
        model.train()

        amp_dtype = (
            torch.bfloat16
            if getattr(cfg.training, "amp_dtype", "bf16") == "bf16"
            else torch.float16
        )
        grad_clip = getattr(cfg.training, "grad_clip_norm", None)
        times: list[float] = []

        for i in range(warmup + steps):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                loss = compute_total_loss(_forward(model, batch), batch, cfg).total
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            torch.cuda.synchronize(device)
            if i >= warmup:
                times.append(time.perf_counter() - start)

        mean = statistics.mean(times)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        result = {
            "name": name,
            "step_ms": mean * 1e3,
            "peak_gb": peak,
            "samples_per_s": batch_size / mean,
            "batch_size": batch_size,
        }
    except torch.cuda.OutOfMemoryError:
        print(f"{name:<42} OOM at batch_size={batch_size}")
        result = None
    finally:
        for obj in ("model", "optimizer", "batch", "loss"):
            if obj in dir():
                pass
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    parser.add_argument("--manifest", default="data/full_npy/train.jsonl")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=4)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    base = load_config(args.config)
    print(f"gpu={torch.cuda.get_device_name(device)}  baseline batch_size={base.data.batch_size}")
    print(f"{'variant':<42}{'step':>10}{'peak mem':>11}{'thruput':>12}{'speedup':>9}")
    print("-" * 84)

    results = []

    def record(label, cfg, **kw):
        out = run_variant(label, cfg, args.manifest, device, args.steps, args.warmup, **kw)
        if out is None:
            return
        results.append(out)
        speed = results[0]["step_ms"] / out["step_ms"] if results else 1.0
        print(
            f"{label:<42}{out['step_ms']:8.1f}ms{out['peak_gb']:9.1f}GB"
            f"{out['samples_per_s']:9.2f}/s{speed:8.2f}x"
        )

    record("baseline (max_frames=16, NCHW)", base)

    cfg = copy.deepcopy(base)
    cfg.data.max_frames = 13
    record("max_frames=13 (no padding waste)", cfg)

    record("channels_last", base, channels_last=True)

    cfg = copy.deepcopy(base)
    cfg.data.max_frames = 13
    record("max_frames=13 + channels_last", cfg, channels_last=True)

    cfg = copy.deepcopy(base)
    cfg.model.gradient_checkpointing = True
    record("grad_checkpointing", cfg)

    cfg = copy.deepcopy(base)
    cfg.data.max_frames = 13
    cfg.model.gradient_checkpointing = True
    record("max_frames=13 + grad_ckpt, bs=8", cfg, batch_size=8)

    cfg = copy.deepcopy(base)
    cfg.data.max_frames = 13
    cfg.model.gradient_checkpointing = True
    record("max_frames=13 + grad_ckpt, bs=16", cfg, batch_size=16)

    if len(results) > 1:
        best = max(results, key=lambda r: r["samples_per_s"])
        print("-" * 84)
        print(
            f"best throughput: {best['name']} at {best['samples_per_s']:.2f} samples/s "
            f"({best['samples_per_s'] / results[0]['samples_per_s']:.2f}x baseline)"
        )


if __name__ == "__main__":
    main()
