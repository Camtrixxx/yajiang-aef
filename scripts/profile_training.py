"""Profile the v1.2 training step to locate throughput bottlenecks.

Sections:
  A. dataset __getitem__ breakdown (file IO vs tensor preprocessing)
  B. DataLoader throughput at several num_workers settings
  C. single-GPU step breakdown (H2D / forward / backward / optimizer / sync)
  D. torch.profiler kernel-level top-N

Run on an idle GPU, e.g.:
  CUDA_VISIBLE_DEVICES=5 python scripts/profile_training.py \
      --config configs/yajiang_v1_2.yaml --manifest data/full_npy/train.jsonl
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.losses import compute_total_loss
from src.utils.device import should_pin_memory


def _fmt(seconds: float) -> str:
    return f"{seconds * 1e3:8.2f} ms"


def _pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:5.1f}%" if whole > 0 else "   n/a"


@contextmanager
def _timer(sink: list[float]):
    start = time.perf_counter()
    yield
    sink.append(time.perf_counter() - start)


def _header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def profile_getitem(dataset, num_samples: int) -> None:
    """Split per-sample cost into raw file reads and everything else."""
    _header("A. dataset.__getitem__ breakdown (single process, no workers)")

    io_times: list[float] = []
    io_bytes = 0
    read_counts: list[int] = []
    original_load = dataset._load_array

    state = {"count": 0}

    def timed_load(path):
        nonlocal io_bytes
        start = time.perf_counter()
        out = original_load(path)
        io_times.append(time.perf_counter() - start)
        io_bytes += out.numel() * out.element_size()
        state["count"] += 1
        return out

    dataset._load_array = timed_load
    totals: list[float] = []
    try:
        for idx in range(min(num_samples, len(dataset))):
            state["count"] = 0
            with _timer(totals):
                dataset[idx]
            read_counts.append(state["count"])
    finally:
        dataset._load_array = original_load

    total = sum(totals)
    io_total = sum(io_times)
    n = len(totals)
    print(f"samples profiled        : {n}")
    print(f"reads per sample        : {statistics.mean(read_counts):.1f}")
    print(f"mean __getitem__        : {_fmt(total / n)}")
    print(f"  file IO               : {_fmt(io_total / n)}  ({_pct(io_total, total)})")
    print(f"  preprocess + alloc    : {_fmt((total - io_total) / n)}  ({_pct(total - io_total, total)})")
    print(f"mean per read           : {_fmt(io_total / max(len(io_times), 1))}")
    print(f"bytes read per sample   : {io_bytes / n / 1e6:.2f} MB")
    print(f"single-worker rate      : {n / total:.2f} samples/s")


def _make_loader(dataset, cfg, num_workers: int, device: torch.device) -> DataLoader:
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(cfg.data, "persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(cfg.data, "prefetch_factor", 4))
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        collate_fn=aef_collate_fn,
        drop_last=True,
        **kwargs,
    )


def profile_dataloader(dataset, cfg, device, worker_counts, steps: int) -> dict[int, float]:
    """Measure pure loader throughput: no model, no GPU work."""
    _header("B. DataLoader throughput (no model, batch consumed and discarded)")
    batch_size = int(cfg.data.batch_size)
    results: dict[int, float] = {}

    for workers in worker_counts:
        loader = _make_loader(dataset, cfg, workers, device)
        it = iter(loader)
        next(it)  # exclude worker spin-up
        per_batch: list[float] = []
        for _ in range(steps):
            start = time.perf_counter()
            try:
                next(it)
            except StopIteration:
                break
            per_batch.append(time.perf_counter() - start)
        mean = statistics.mean(per_batch)
        results[workers] = mean
        print(
            f"num_workers={workers:3d}  mean batch wait {_fmt(mean)}"
            f"  p50 {_fmt(statistics.median(per_batch))}"
            f"  max {_fmt(max(per_batch))}"
            f"  {batch_size / mean:7.2f} samples/s"
        )
        del it, loader
    return results


def _move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move(v, device) for v in obj]
    return obj


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


def profile_gpu_step(dataset, cfg, device, steps: int, warmup: int) -> None:
    """Break down one optimizer step with the loader kept out of the critical path."""
    _header("C. GPU step breakdown (batch pre-staged on device, loader excluded)")

    model = AEFModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    model.train()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters              : {params / 1e6:.2f} M total, {trainable / 1e6:.2f} M trainable")

    use_amp = bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(cfg.training, "amp_dtype", "bf16") == "bf16" else torch.float16
    grad_clip = getattr(cfg.training, "grad_clip_norm", None)

    loader = _make_loader(dataset, cfg, int(getattr(cfg.data, "num_workers", 8)), device)
    host_batch = next(iter(loader))
    frames = host_batch["source_frames"]
    payload_mb = sum(
        v.numel() * v.element_size()
        for v in host_batch.values()
        if torch.is_tensor(v)
    ) / 1e6
    print(f"source_frames shape     : {tuple(frames.shape)}  {frames.dtype}")
    print(f"host->device payload    : {payload_mb:.2f} MB per batch")

    stages: dict[str, list[float]] = {
        "h2d": [], "forward": [], "loss": [], "backward": [],
        "clip": [], "step": [], "item_sync": [], "total": [],
    }

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for i in range(warmup + steps):
        record = i >= warmup
        _sync()
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        batch = _move(host_batch, device)
        _sync()
        t_h2d = time.perf_counter() - t0

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            t0 = time.perf_counter()
            output = _forward(model, batch)
            _sync()
            t_fwd = time.perf_counter() - t0

            t0 = time.perf_counter()
            loss_out = compute_total_loss(output, batch, cfg)
            _sync()
            t_loss = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss_out.total.backward()
        _sync()
        t_bwd = time.perf_counter() - t0

        t_clip = 0.0
        if grad_clip is not None:
            t0 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            _sync()
            t_clip = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        _sync()
        t_step = time.perf_counter() - t0

        t0 = time.perf_counter()
        for v in loss_out.components.values():
            float(v.detach().item())
        t_item = time.perf_counter() - t0

        if record:
            stages["h2d"].append(t_h2d)
            stages["forward"].append(t_fwd)
            stages["loss"].append(t_loss)
            stages["backward"].append(t_bwd)
            stages["clip"].append(t_clip)
            stages["step"].append(t_step)
            stages["item_sync"].append(t_item)
            stages["total"].append(time.perf_counter() - t_total)

    total = statistics.mean(stages["total"])
    print(f"\n{'stage':<22}{'mean':>12}{'share':>9}")
    print("-" * 43)
    for name in ("h2d", "forward", "loss", "backward", "clip", "step", "item_sync"):
        mean = statistics.mean(stages[name])
        print(f"{name:<22}{_fmt(mean)}{_pct(mean, total):>9}")
    print("-" * 43)
    print(f"{'total step':<22}{_fmt(total)}{'100.0%':>9}")
    print(f"\ncompute-only rate       : {cfg.data.batch_size / total:.2f} samples/s per GPU")
    if device.type == "cuda":
        print(f"peak memory allocated   : {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

    _padding_waste(cfg, host_batch)
    del loader
    return model, host_batch


def _padding_waste(cfg, batch) -> None:
    """Report how much of the frame axis is pure padding the model still computes on."""
    mask = batch["source_frame_mask"] & batch["source_input_mask"][:, :, None]
    used = int(mask.sum())
    slots = int(mask.numel())
    print(
        f"frame slots             : {slots} allocated, {used} valid "
        f"({_pct(slots - used, slots)} wasted compute, max_frames={cfg.data.max_frames})"
    )


def profile_kernels(model, host_batch, cfg, device, steps: int) -> None:
    """Kernel-level view of where GPU time actually goes."""
    _header("D. torch.profiler top operators by self CUDA time")
    from torch.profiler import ProfilerActivity, profile

    use_amp = bool(getattr(cfg.training, "amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(cfg.training, "amp_dtype", "bf16") == "bf16" else torch.float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    batch = _move(host_batch, device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    for _ in range(3):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            compute_total_loss(_forward(model, batch), batch, cfg).total.backward()
        optimizer.zero_grad(set_to_none=True)

    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = compute_total_loss(_forward(model, batch), batch, cfg).total
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=22))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    parser.add_argument("--manifest", default="data/full_npy/train.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--getitem-samples", type=int, default=12)
    parser.add_argument("--loader-steps", type=int, default=12)
    parser.add_argument("--workers", default="0,8,16,32")
    parser.add_argument(
        "--sections", default="abcd", help="subset of a,b,c,d to run"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if getattr(cfg.training, "allow_tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(getattr(cfg.training, "matmul_precision", "high"))
    torch.backends.cudnn.benchmark = bool(getattr(cfg.training, "cudnn_benchmark", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(getattr(cfg.experiment, "seed", 42)))
    np.random.seed(int(getattr(cfg.experiment, "seed", 42)))

    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split=args.split)
    print(f"device={device}  dataset={len(dataset)} samples  batch_size={cfg.data.batch_size}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    sections = args.sections.lower()
    if "a" in sections:
        profile_getitem(dataset, args.getitem_samples)
    if "b" in sections:
        workers = [int(w) for w in args.workers.split(",") if w.strip()]
        profile_dataloader(dataset, cfg, device, workers, args.loader_steps)
    model = host_batch = None
    if "c" in sections:
        model, host_batch = profile_gpu_step(dataset, cfg, device, args.steps, args.warmup)
    if "d" in sections and model is not None:
        profile_kernels(model, host_batch, cfg, device, max(args.steps // 2, 4))


if __name__ == "__main__":
    main()
