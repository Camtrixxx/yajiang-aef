"""Is the DDP grad-stride warning worth fixing, and which fix actually works?

The warning (reducer.cpp:335) fires on STPBlock.fusion's 1x1 conv weight,
shape [256, 768, 1, 1]:

    grad        strides = [768, 1, 768, 768]   <- channels_last, from inductor
    bucket_view strides = [768, 1,   1,   1]   <- contiguous, how DDP bucketed

With H=W=1 these two describe THE SAME BYTES: the H and W indices are always
0, so their strides never participate in addressing. It is a metadata
disagreement in a degenerate shape, not a real layout difference. DDP's
reducer compares strides elementwise without an equivalence test, so it warns
and takes a copy path. PyTorch itself says "This is not an error".

So the question is not "how do I silence it" but "does it cost anything".
Four arms, one fresh process each:

  none          reproduce, and confirm the two detectors agree
  bucket_view   gradient_as_bucket_view=True -- the usual recommendation. I
                expect this to NOT fix the mismatch: the bucket view is still
                built from the param's contiguous stride, so inductor's
                channels_last grad still needs a copy into it. It should still
                save memory.
  channels_last set the 1x1 conv weights to channels_last BEFORE DDP wraps the
                model, so DDP buckets with the stride inductor will hand back.
                Numerically inert for k=1 (same bytes).
  both          do they compose or conflict?

Two independent detectors, because neither alone is trustworthy:

  1. warning capture. TORCH_WARN_ONCE fires once per process, which is exactly
     enough when each arm is its own torchrun launch.
  2. after backward, compare grad.stride() to the contiguous stride for that
     shape -- what the reducer itself tests. Does not depend on warning state.

Reported per arm: mismatching params, warning seen, steady step ms (max across
ranks so stragglers count), peak GB.
"""
import argparse
import os
import statistics
import time
import warnings

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from bench_ddp8 import allreduce_max, build_loader, fwd_kwargs, move, ns
from src.models.model import AEFModel
from src.training.losses import compute_total_loss


def stride_mismatches(model):
    """Params whose grad stride differs from the bucket view's.

    DDP builds each bucket view as as_strided(param.sizes(), param.strides()),
    so the reference is the PARAM's stride, not the contiguous stride. An
    earlier version of this compared against contiguous and therefore reported
    the channels_last arm -- where param and grad agree and the warning is gone
    -- as 8 mismatches.

    CAVEAT: under gradient_as_bucket_view=True this detector is blind. There
    param.grad IS the bucket view, so its stride trivially equals param's and
    this returns 0 even while the reducer still copies inductor's channels_last
    grad into the bucket. Trust the captured warning in that arm.
    """
    out = []
    for name, p in model.named_parameters():
        if p.grad is None or p.grad.dim() != 4:
            continue
        got, want = tuple(p.grad.stride()), tuple(p.stride())
        if got != want:
            out.append((name, tuple(p.grad.shape), got, want))
    return out


def to_channels_last_1x1(model):
    """Bucket the 1x1 conv weights the way inductor will hand their grads back.

    Only k=1: there the two layouts address identical memory, so this is pure
    metadata. Returns how many it touched, so a silent no-op cannot pass as a
    working fix.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (1, 1):
            m.weight.data = m.weight.data.to(memory_format=torch.channels_last)
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/yajiang_v1_2.yaml")
    p.add_argument("--manifest", default="data/full_npy/train.jsonl")
    p.add_argument("--arm", required=True,
                   choices=["none", "bucket_view", "channels_last", "both"])
    p.add_argument("--mode", default="max-autotune-no-cudagraphs")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("nccl")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)

    # Config defaults are already the accelerated ones (13 frames, group norm,
    # fast attention, nw=2), so this measures the fix where it will actually run.
    cfg = ns(yaml.safe_load(open(args.config)))
    _, loader = build_loader(cfg, args.manifest, rank, world)
    batch = move(next(iter(loader)), dev)

    model = AEFModel(cfg).to(dev)
    n_cl = to_channels_last_1x1(model) if args.arm in ("channels_last", "both") else 0
    model = DDP(model, device_ids=[local],
                find_unused_parameters=bool(cfg.training.find_unused_parameters),
                gradient_as_bucket_view=args.arm in ("bucket_view", "both"))
    if args.mode != "eager":
        model = torch.compile(model, mode=args.mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**fwd_kwargs(batch))
            loss = compute_total_loss(out, batch, cfg)
        loss.total.backward()
        opt.step()

    # Detector 1: catch the reducer's warning on the very first backward, before
    # TORCH_WARN_ONCE has been spent. Compile happens inside this step too.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step()
    warned = any("bucket view strides" in str(w.message) for w in caught)

    for _ in range(args.warmup):
        step()

    # Detector 2: what the reducer compares, read off the live grads.
    bad = stride_mismatches(model)

    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    step_ms = allreduce_max(statistics.median(times), dev)
    peak = allreduce_max(torch.cuda.max_memory_allocated() / 1e9, dev)
    n_bad = int(allreduce_max(len(bad), dev))
    warn_any = int(allreduce_max(1 if warned else 0, dev))

    if rank == 0:
        print(f"RESULT\t{args.arm}\tstep_ms={step_ms:.1f}\tpeak_gb={peak:.2f}"
              f"\tmismatch={n_bad}\twarned={bool(warn_any)}\tcl_convs={n_cl}"
              f"\tspread={max(times) - min(times):.1f}")
        for name, shape, got, want in bad[:4]:
            print(f"  MISMATCH\t{name}\t{shape}\tgot={got}\twant={want}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
