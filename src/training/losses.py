from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


IGNORE_INDEX = 255


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


def _resize_target_like(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return target

    if target.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ):
        # nearest for labels
        target = target.float()
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
        return target.long()

    return F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)


def continuous_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    target = _resize_target_like(pred, target)
    loss = F.l1_loss(pred, target, reduction="none")

    if mask is not None:
        mask = _resize_target_like(pred[:, :1], mask.float()).float()
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0) * pred.shape[1]
        return loss.sum() / denom

    return loss.mean()


def categorical_recon_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    分类重建损失。

    支持:
    - target: [B, H, W]
    - target: [B, 1, H, W]
    - target: [B, C, H, W] one-hot

    默认 ignore_index=255，用于忽略 jrc_water 中的 nodata 区域。
    """

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    if target.ndim == 4 and target.shape[1] > 1:
        # one-hot -> class index
        target = target.argmax(dim=1)

    # resize label target to prediction spatial size
    target = _resize_target_like(logits[:, :1], target[:, None]).squeeze(1).long()

    if mask is not None:
        mask = _resize_target_like(logits[:, :1], mask.float()).squeeze(1) > 0.5
        target = target.masked_fill(~mask, ignore_index)

    # 如果一个 batch 内全部是 ignore_index，cross_entropy 会变成 nan。
    # 这里直接返回 0 loss，避免 jrc_water 某些 patch 全部无效时训练炸掉。
    valid = target != ignore_index
    if valid.sum() == 0:
        return logits.new_tensor(0.0)

    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def embedding_uniformity_loss(emb: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Wang & Isola style uniformity loss on normalized embeddings.
    emb: [B, D]
    """
    if emb.shape[0] <= 1:
        return emb.new_tensor(0.0)

    emb = F.normalize(emb, dim=-1)
    dist = torch.pdist(emb, p=2).pow(2)
    return torch.log(torch.exp(-t * dist).mean() + 1e-8)


def batch_variance_loss(
    emb: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    if emb.shape[0] <= 1:
        return emb.new_tensor(0.0)

    std = torch.sqrt(emb.var(dim=0, unbiased=False) + eps)
    return F.relu(target_std - std).mean()


def decorrelation_loss(emb: torch.Tensor) -> torch.Tensor:
    if emb.shape[0] <= 1:
        return emb.new_tensor(0.0)

    emb = emb - emb.mean(dim=0, keepdim=True)
    emb = emb / (emb.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
    corr = emb.T @ emb / emb.shape[0]

    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    off_diag = corr - eye
    return off_diag.pow(2).mean()


def orthogonality_loss(feature_map: torch.Tensor) -> torch.Tensor:
    """
    feature_map: [B, C, H, W]
    Encourage channels to be less redundant after spatial pooling.
    """
    b, c, _, _ = feature_map.shape
    x = feature_map.flatten(2)  # [B, C, HW]
    x = F.normalize(x, dim=1)

    gram = torch.matmul(x, x.transpose(1, 2)).mean(dim=0)  # [C, C]
    eye = torch.eye(c, device=gram.device, dtype=gram.dtype)

    return (gram - eye).pow(2).mean()


def compute_total_loss(model_output, batch: dict, cfg) -> LossOutput:
    losses: dict[str, torch.Tensor] = {}
    training_cfg = cfg.training

    recon_weight = getattr(training_cfg, "reconstruction_weight", 1.0)
    uniformity_weight = getattr(training_cfg, "uniformity_weight", 0.0)
    variance_weight = getattr(training_cfg, "variance_weight", 0.0)
    decorrelation_weight = getattr(training_cfg, "decorrelation_weight", 0.0)
    orthogonality_weight = getattr(training_cfg, "orthogonality_weight", 0.0)

    target_tensors = batch.get("targets", {})
    target_masks = batch.get("target_masks", {})

    recon_total = model_output.embedding.new_tensor(0.0)

    for tgt in cfg.data.target_sources:
        name = tgt.name if hasattr(tgt, "name") else tgt["name"]
        loss_type = tgt.loss_type if hasattr(tgt, "loss_type") else tgt["loss_type"]

        pred = model_output.reconstructions[name]
        target = target_tensors[name]
        mask = target_masks.get(name)

        if loss_type == "categorical":
            cur = categorical_recon_loss(
                pred,
                target,
                ignore_index=IGNORE_INDEX,
                mask=mask,
            )
        else:
            cur = continuous_recon_loss(pred, target, mask=mask)

        losses[f"recon/{name}"] = cur
        recon_total = recon_total + cur

    losses["recon/total"] = recon_total

    emb = model_output.embedding
    losses["reg/uniformity"] = embedding_uniformity_loss(emb)
    losses["reg/variance"] = batch_variance_loss(emb)
    losses["reg/decorrelation"] = decorrelation_loss(emb)
    losses["reg/orthogonality"] = orthogonality_loss(model_output.embedding_map)

    total = recon_weight * losses["recon/total"]
    total = total + uniformity_weight * losses["reg/uniformity"]
    total = total + variance_weight * losses["reg/variance"]
    total = total + decorrelation_weight * losses["reg/decorrelation"]
    total = total + orthogonality_weight * losses["reg/orthogonality"]

    losses["loss"] = total
    return LossOutput(total=total, components=losses)