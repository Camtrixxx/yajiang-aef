from __future__ import annotations

from typing import Any

import numpy as np
import torch


IGNORE_INDEX = 255


def target_to_eval_label(targets: dict[str, torch.Tensor], target_name: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    if target_name == "jrc_binary":
        y = targets["jrc_water"][:, 0].long()
        valid = y != IGNORE_INDEX
        return (y > 0).long(), valid, "classification"

    if target_name == "jrc_permanent":
        y = targets["jrc_water"][:, 0].long()
        valid = y != IGNORE_INDEX
        return (y >= 80).long(), valid, "classification"

    if target_name == "jrc_water":
        y = targets["jrc_water"][:, 0].long()
        valid = y != IGNORE_INDEX
        return y, valid, "classification"

    if target_name == "worldcover":
        y = targets["worldcover"][:, 0].long()
        valid = y != IGNORE_INDEX
        return y, valid, "classification"

    if target_name == "dem":
        y = targets["dem"][:, 0].float()
        valid = torch.isfinite(y)
        return y, valid, "regression"

    raise ValueError(f"Unsupported eval target: {target_name}")


def sample_feature_pixels(
    feature_map: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    max_pixels_per_patch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    b, c, h, w = feature_map.shape
    xs = []
    ys = []

    for idx in range(b):
        valid = valid_mask[idx].reshape(-1).cpu().numpy().astype(bool)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        if max_pixels_per_patch > 0 and valid_idx.size > max_pixels_per_patch:
            valid_idx = rng.choice(valid_idx, size=max_pixels_per_patch, replace=False)

        fmap = feature_map[idx].permute(1, 2, 0).reshape(h * w, c).cpu().numpy()
        lab = labels[idx].reshape(-1).cpu().numpy()
        xs.append(fmap[valid_idx])
        ys.append(lab[valid_idx])

    if not xs:
        return np.empty((0, c), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0)


def sample_paired_feature_pixels(
    feature_a: torch.Tensor,
    feature_b: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    max_pixels_per_patch: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, c_a, h, w = feature_a.shape
    _, c_b, h_b, w_b = feature_b.shape
    if (h, w) != (h_b, w_b):
        raise ValueError(f"Feature map shapes do not align: {(h, w)} vs {(h_b, w_b)}")

    xs_a = []
    xs_b = []
    ys = []

    for idx in range(b):
        valid = valid_mask[idx].reshape(-1).cpu().numpy().astype(bool)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue

        if max_pixels_per_patch > 0 and valid_idx.size > max_pixels_per_patch:
            valid_idx = rng.choice(valid_idx, size=max_pixels_per_patch, replace=False)

        fmap_a = feature_a[idx].permute(1, 2, 0).reshape(h * w, c_a).cpu().numpy()
        fmap_b = feature_b[idx].permute(1, 2, 0).reshape(h * w, c_b).cpu().numpy()
        lab = labels[idx].reshape(-1).cpu().numpy()

        xs_a.append(fmap_a[valid_idx])
        xs_b.append(fmap_b[valid_idx])
        ys.append(lab[valid_idx])

    if not xs_a:
        return (
            np.empty((0, c_a), dtype=np.float32),
            np.empty((0, c_b), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.concatenate(xs_a, axis=0).astype(np.float32),
        np.concatenate(xs_b, axis=0).astype(np.float32),
        np.concatenate(ys, axis=0),
    )


def split_indices(num_items: int, train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_items)
    rng.shuffle(indices)
    n_train = max(1, min(num_items - 1, int(round(num_items * train_fraction))))
    return indices[:n_train].tolist(), indices[n_train:].tolist()


def subset_records(dataset: Any, indices: list[int]) -> None:
    dataset.records = [dataset.records[i] for i in indices]
