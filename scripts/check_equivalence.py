"""Check whether a config change is numerically equivalent to the baseline.

Motivation: `max_frames` folds into the batch dim and the sensor stem contains
BatchNorm2d, so dropping padded frame slots may shift BN batch statistics.
A perf knob must be proven to leave the math alone; if it does not, it has to
be judged as a convergence knob instead (weaker gate, longer experiment).

Gate classes:
  eval  mode -> mathematical equivalence expected (BN uses running stats)
  train mode -> BN uses batch stats; any diff here means trajectory-only

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/check_equivalence.py \
      --config configs/yajiang_v1_2.yaml --manifest data/full_npy/train.jsonl
"""

from __future__ import annotations

import argparse
import copy

import torch

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.models.model import AEFModel
from src.training.losses import compute_total_loss

BATCH_KEYS = (
    "source_frames",
    "source_frame_mask",
    "source_input_mask",
    "source_type_ids",
    "source_timestamps",
    "valid_start_ms",
    "valid_end_ms",
)


def build_batch(cfg, manifest: str, split: str, batch_size: int):
    ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split=split)
    items = [ds[i] for i in range(batch_size)]
    return aef_collate_fn(items)


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items() if torch.is_tensor(vv)}
        else:
            out[k] = v
    return out


def truncate_frames(batch, max_frames: int):
    """Emulate the max_frames=13 config by slicing the padded frame axis."""
    out = dict(batch)
    out["source_frames"] = batch["source_frames"][:, :, :max_frames].contiguous()
    out["source_frame_mask"] = batch["source_frame_mask"][:, :, :max_frames].contiguous()
    out["source_timestamps_ms"] = batch["source_timestamps_ms"][:, :, :max_frames].contiguous()
    return out


@torch.no_grad()
def run_once(model, batch, cfg, train_mode: bool, seed: int = 1234, bn_eval: bool = False):
    # The vMF bottleneck injects randn noise in BOTH train and eval branches
    # (bottleneck.py:60 and sample_vmf at :67), so the model is stochastic even
    # in eval. Reseed before every forward or the comparison measures noise.
    model.train(train_mode)
    if bn_eval:
        # Isolation arm: keep everything else in train mode but make BN use
        # running stats, so any residual diff cannot be attributed to BN.
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    loss_out = compute_total_loss(out, batch, cfg)
    return out, loss_out


def compare(tag, a, b, loss_a, loss_b):
    emb_a, emb_b = a.embedding_map.float(), b.embedding_map.float()
    d_emb = (emb_a - emb_b).abs()
    scale = emb_a.abs().mean().item() + 1e-12
    d_loss = abs(float(loss_a.total) - float(loss_b.total))
    print(f"\n--- {tag} ---")
    print(f"  loss baseline = {float(loss_a.total):.10f}")
    print(f"  loss variant  = {float(loss_b.total):.10f}")
    print(f"  |dloss|       = {d_loss:.3e}")
    print(f"  embedding_map max abs diff = {d_emb.max().item():.3e}")
    print(f"  embedding_map rel diff     = {(d_emb.mean().item() / scale):.3e}")
    return d_loss, d_emb.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=13)
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    batch = build_batch(cfg, args.manifest, args.split, args.batch_size)
    batch = to_device(batch, device)
    trunc = truncate_frames(batch, args.max_frames)

    valid = batch["source_frame_mask"][:, :, args.max_frames:]
    print(f"frame slots dropped: {tuple(valid.shape)}  any_valid={bool(valid.any())}")
    dropped = batch["source_frames"][:, :, args.max_frames:]
    print(f"dropped payload absmax = {dropped.abs().max().item():.3e} (0 => pure padding)")

    model = AEFModel(cfg).to(device)
    torch.manual_seed(0)

    # BN running stats mutate on every train-mode forward; snapshot so both
    # arms of each comparison see an identical model.
    pristine = copy.deepcopy(model.state_dict())

    results = {}
    # CONTROL first: baseline vs baseline. Must read ~0, else the harness is
    # measuring nondeterminism and every verdict below is worthless.
    for mode, label in ((False, "eval"), (True, "train")):
        model.load_state_dict(pristine)
        out_a, loss_a = run_once(model, batch, cfg, mode)
        model.load_state_dict(pristine)
        out_b, loss_b = run_once(model, batch, cfg, mode)
        results[f"CONTROL {label} (identical input)"] = compare(
            f"CONTROL {label} mode -- identical input, expect 0", out_a, out_b, loss_a, loss_b
        )

    for mode, name in ((False, "eval mode (BN running stats)"), (True, "train mode (BN batch stats)")):
        model.load_state_dict(pristine)
        out_a, loss_a = run_once(model, batch, cfg, mode)
        model.load_state_dict(pristine)
        out_b, loss_b = run_once(model, trunc, cfg, mode)
        results[name] = compare(name, out_a, out_b, loss_a, loss_b)

    # Isolation: train mode with BN frozen. If this collapses to ~0, the
    # train-mode difference is attributable to BN batch stats over padded frames.
    model.load_state_dict(pristine)
    out_a, loss_a = run_once(model, batch, cfg, True, bn_eval=True)
    model.load_state_dict(pristine)
    out_b, loss_b = run_once(model, trunc, cfg, True, bn_eval=True)
    results["train mode, BN frozen (isolation)"] = compare(
        "train mode, BN frozen -- isolates BN as the cause", out_a, out_b, loss_a, loss_b
    )

    print("\n=== verdict ===")
    for name, (d_loss, d_emb) in results.items():
        ok = d_loss < args.tol and d_emb < args.tol
        print(f"  {name:44s}: {'EQUIVALENT' if ok else 'DIFFERENT'} (tol={args.tol:g})")
    print("\nIf a CONTROL row reads DIFFERENT, the harness is unsound and the")
    print("other rows carry no information.")


if __name__ == "__main__":
    main()
