from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.models.model import AEFModel


def dict_to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def load_deploy_model(path: str | Path, device: torch.device) -> tuple[AEFModel, Any]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = dict_to_namespace(ckpt["config"])
    model = AEFModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move(obj):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: move(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [move(v) for v in obj]
        return obj

    return move(batch)


@torch.no_grad()
def extract_aef_embedding_map(model: AEFModel, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    batch = batch_to_device(batch, device)
    output = model(
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
    return output.embedding_map.detach().cpu()


def extract_composite_map(batch: dict[str, Any], cfg: Any) -> torch.Tensor:
    frames = batch["source_frames"].float()
    masks = batch["source_frame_mask"].float()

    source_channels = cfg.model.source_channels
    if not isinstance(source_channels, dict):
        source_channels = vars(source_channels)

    features = []
    for src_idx, src_name in enumerate(cfg.data.input_sources):
        in_ch = int(source_channels[src_name])
        x = frames[:, src_idx, :, :in_ch]
        m = masks[:, src_idx, :, None, None, None]
        denom = m.sum(dim=1).clamp_min(1.0)
        mean = (x * m).sum(dim=1) / denom
        features.append(mean)

    composite = torch.cat(features, dim=1)
    return F.avg_pool2d(composite, kernel_size=2, stride=2)


def random_project_features(features: np.ndarray, out_dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(features.shape[1], 1) ** 0.5
    proj = rng.normal(0.0, scale, size=(features.shape[1], out_dim)).astype(np.float32)
    return features.astype(np.float32) @ proj
