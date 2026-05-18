from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.eval.features import batch_to_device, load_deploy_model
from src.utils.device import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AEF embedding collapse and diversity diagnostics.")
    parser.add_argument("--manifest", type=str, default="data/full_npy/train.jsonl")
    parser.add_argument(
        "--deploy-model",
        type=str,
        default="outputs/aef_hyh_yajiang_v1_1/exports/aef_hyh_yajiang_v1_1_deploy.pt",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/embedding_health/v1_1")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-patches", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-map-pixels-per-patch", type=int, default=64)
    parser.add_argument("--active-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cosine_summary(x: np.ndarray, max_pairs: int = 20000, seed: int = 42) -> dict[str, float | int | None]:
    if x.shape[0] < 2:
        return {
            "num_pairs": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)
    keep = i != j
    i = i[keep]
    j = j[keep]
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sims = (x_norm[i] * x_norm[j]).sum(axis=1)
    return {
        "num_pairs": int(len(sims)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p05": float(np.quantile(sims, 0.05)),
        "p50": float(np.quantile(sims, 0.50)),
        "p95": float(np.quantile(sims, 0.95)),
    }


def matrix_summary(x: np.ndarray, active_threshold: float, seed: int) -> dict[str, Any]:
    if x.size == 0:
        return {
            "num_vectors": 0,
            "dim": 0,
            "active_dims": 0,
            "active_threshold": active_threshold,
        }
    channel_std = x.std(axis=0)
    norms = np.linalg.norm(x, axis=1)
    return {
        "num_vectors": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "active_dims": int((channel_std > active_threshold).sum()),
        "active_threshold": float(active_threshold),
        "std_mean": float(channel_std.mean()),
        "std_min": float(channel_std.min()),
        "std_max": float(channel_std.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "cosine": cosine_summary(x, seed=seed),
    }


def sample_map_vectors(embedding_map: torch.Tensor, max_pixels_per_patch: int, rng: np.random.Generator) -> np.ndarray:
    b, d, h, w = embedding_map.shape
    flat = embedding_map.permute(0, 2, 3, 1).reshape(b, h * w, d).detach().cpu().numpy()
    parts = []
    for patch_vectors in flat:
        take = min(max_pixels_per_patch, patch_vectors.shape[0])
        idx = rng.choice(patch_vectors.shape[0], size=take, replace=False)
        parts.append(patch_vectors[idx])
    if not parts:
        return np.empty((0, d), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def collapse_risk(summary: dict[str, Any]) -> str:
    active = summary["global_embedding"].get("active_dims", 0)
    dim = max(summary["global_embedding"].get("dim", 1), 1)
    cos_mean = summary["global_embedding"].get("cosine", {}).get("mean")
    std_mean = summary["global_embedding"].get("std_mean", 0.0)
    active_ratio = active / dim
    if active_ratio < 0.2 or std_mean < 0.01 or (cos_mean is not None and cos_mean > 0.95):
        return "high"
    if active_ratio < 0.5 or std_mean < 0.03 or (cos_mean is not None and cos_mean > 0.85):
        return "medium"
    return "low"


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    g = summary["global_embedding"]
    m = summary["spatial_embedding_map_sample"]
    lines = [
        "# Embedding Health Report",
        "",
        f"- deploy model: `{summary['deploy_model']}`",
        f"- manifest: `{summary['manifest']}`",
        f"- patches checked: `{summary['num_patches']}`",
        f"- collapse risk: `{summary['collapse_risk']}`",
        "",
        "## Global embedding",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| active dims | {g.get('active_dims')}/{g.get('dim')} |",
        f"| std mean | {g.get('std_mean'):.6f} |",
        f"| norm mean / std | {g.get('norm_mean'):.6f} / {g.get('norm_std'):.6f} |",
        f"| cosine mean / p95 | {g['cosine'].get('mean'):.6f} / {g['cosine'].get('p95'):.6f} |",
        "",
        "## Spatial embedding map sample",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| sampled vectors | {m.get('num_vectors')} |",
        f"| active dims | {m.get('active_dims')}/{m.get('dim')} |",
        f"| std mean | {m.get('std_mean'):.6f} |",
        f"| norm mean / std | {m.get('norm_mean'):.6f} / {m.get('norm_std'):.6f} |",
        f"| cosine mean / p95 | {m['cosine'].get('mean'):.6f} / {m['cosine'].get('p95'):.6f} |",
        "",
        "Interpretation: high cosine close to 1, very low std, or very few active dims indicates embedding collapse.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, cfg = load_deploy_model(args.deploy_model, device=device)
    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    if args.max_patches > 0:
        dataset.records = dataset.records[: args.max_patches]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=aef_collate_fn,
    )

    global_embeddings = []
    spatial_vectors = []
    sample_ids = []
    for batch in loader:
        sample_ids.extend(batch["sample_id"])
        with torch.no_grad():
            device_batch = batch_to_device(batch, device)
            output = model(
                source_frames=device_batch["source_frames"],
                source_timestamps_ms=device_batch["source_timestamps_ms"],
                source_frame_mask=device_batch["source_frame_mask"],
                source_input_mask=device_batch["source_input_mask"],
                source_type_ids=device_batch["source_type_ids"],
                valid_start_ms=device_batch["valid_start_ms"],
                valid_end_ms=device_batch["valid_end_ms"],
                target_relative_time=device_batch["target_relative_time"],
                target_metadata=device_batch["target_metadata"],
            )
        global_embeddings.append(output.embedding.detach().cpu().numpy())
        spatial_vectors.append(sample_map_vectors(output.embedding_map, args.max_map_pixels_per_patch, rng))

    global_matrix = np.concatenate(global_embeddings, axis=0).astype(np.float32)
    spatial_matrix = np.concatenate(spatial_vectors, axis=0).astype(np.float32)
    summary = {
        "deploy_model": str(Path(args.deploy_model).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "num_patches": int(len(dataset)),
        "sample_ids": sample_ids,
        "global_embedding": matrix_summary(global_matrix, args.active_threshold, args.seed),
        "spatial_embedding_map_sample": matrix_summary(spatial_matrix, args.active_threshold, args.seed),
    }
    summary["collapse_risk"] = collapse_risk(summary)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, out_dir)
    print(json.dumps({k: summary[k] for k in ["num_patches", "collapse_risk", "global_embedding", "spatial_embedding_map_sample"]}, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved report to: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
