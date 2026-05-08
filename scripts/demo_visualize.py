from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.eval.features import batch_to_device, load_deploy_model
from src.utils.device import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a visual demo for a trained AEF model.")
    parser.add_argument("--config", type=str, default="configs/yajiang_v0_3_c.yaml")
    parser.add_argument("--manifest", type=str, default="data/full_npy/train.jsonl")
    parser.add_argument(
        "--deploy-model",
        type=str,
        default="outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt",
    )
    parser.add_argument("--sample-index", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="outputs/demo_v0_3_c/demo_patch.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_image(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(x[np.isfinite(x)], [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def make_s2_rgb(source_frames: torch.Tensor) -> np.ndarray:
    # S2 is source 0 in current config. Use a false/visual RGB from the first 3 bands.
    s2 = source_frames[0, 0].float().numpy()
    rgb = np.stack(
        [
            normalize_image(s2[2]),
            normalize_image(s2[1]),
            normalize_image(s2[0]),
        ],
        axis=-1,
    )
    return rgb


def pca_rgb(feature_map: torch.Tensor) -> np.ndarray:
    x = feature_map.detach().cpu()[0]
    c, h, w = x.shape
    flat = x.permute(1, 2, 0).reshape(h * w, c).float().numpy()
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    comps = flat @ vt[:3].T
    comps = comps.reshape(h, w, 3)
    return normalize_image(comps)


def categorical_prediction(logits: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    pred = logits.argmax(dim=1, keepdim=True).float()
    pred = F.interpolate(pred, size=size, mode="nearest")
    return pred[0, 0].detach().cpu().numpy()


def continuous_prediction(pred: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
    return pred[0, 0].detach().cpu().numpy()


def plot_demo(batch: dict, output, out_path: Path) -> None:
    source_frames = batch["source_frames"][0].cpu()
    targets = {k: v[0].cpu() for k, v in batch["targets"].items()}

    dem_true = targets["dem"][0].numpy()
    wc_true = targets["worldcover"][0].numpy()
    jrc_true = targets["jrc_water"][0].numpy()

    dem_pred = continuous_prediction(output.reconstructions["dem"].detach().cpu(), dem_true.shape)
    wc_pred = categorical_prediction(output.reconstructions["worldcover"].detach().cpu(), wc_true.shape)
    jrc_pred = categorical_prediction(output.reconstructions["jrc_water"].detach().cpu(), jrc_true.shape)
    emb_rgb = pca_rgb(output.embedding_map)

    rgb = make_s2_rgb(source_frames)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("S2 RGB composite")
    axes[0, 1].imshow(emb_rgb)
    axes[0, 1].set_title("AEF embedding PCA")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0,
        0.8,
        f"embedding dim: {output.embedding.shape[-1]}\n"
        f"embedding map: {tuple(output.embedding_map.shape[-2:])}\n"
        f"targets: dem / worldcover / jrc_water",
        fontsize=12,
        va="top",
    )

    im = axes[1, 0].imshow(dem_true, cmap="terrain")
    axes[1, 0].set_title("DEM target")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = axes[1, 1].imshow(dem_pred, cmap="terrain")
    axes[1, 1].set_title("DEM reconstruction")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    axes[1, 2].imshow(np.abs(dem_pred - dem_true), cmap="magma")
    axes[1, 2].set_title("DEM abs error")

    axes[2, 0].imshow(wc_true, cmap="tab20", vmin=0, vmax=8)
    axes[2, 0].set_title("WorldCover target")
    axes[2, 1].imshow(wc_pred, cmap="tab20", vmin=0, vmax=8)
    axes[2, 1].set_title("WorldCover reconstruction")
    jrc_show = np.ma.masked_where(jrc_true == 255, jrc_true)
    axes[2, 2].imshow(jrc_show, cmap="Blues", vmin=0, vmax=100)
    axes[2, 2].contour(jrc_pred > 0, levels=[0.5], colors="red", linewidths=0.7)
    axes[2, 2].set_title("JRC target + predicted water contour")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    cfg = load_config(args.config)
    model, _ = load_deploy_model(args.deploy_model, device=device)

    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index out of range: {args.sample_index}, dataset size={len(dataset)}")

    dataset.records = [dataset.records[args.sample_index]]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=aef_collate_fn)
    batch = next(iter(loader))

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

    out_path = Path(args.output)
    plot_demo(batch, output, out_path)
    print(f"Saved demo to: {out_path}")


if __name__ == "__main__":
    main()
