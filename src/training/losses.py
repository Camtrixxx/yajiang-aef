from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    class_weight: torch.Tensor | None = None,
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
        return logits.sum() * 0.0

    if class_weight is not None:
        class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)

    return F.cross_entropy(
        logits,
        target,
        weight=class_weight,
        ignore_index=ignore_index,
    )


def _namespace_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _namespace_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_namespace_to_dict(v) for v in obj]
    return obj


def _get_named_float(mapping: Any, name: str, default: float) -> float:
    mapping = _namespace_to_dict(mapping)
    if not isinstance(mapping, dict):
        return default
    return float(mapping.get(name, default))


def _get_class_weight(training_cfg: Any, target_name: str, device: torch.device) -> torch.Tensor | None:
    weights = _namespace_to_dict(getattr(training_cfg, "class_weights", {}))
    if not isinstance(weights, dict) or target_name not in weights:
        return None
    return torch.tensor(weights[target_name], device=device, dtype=torch.float32)


def embedding_uniformity_loss(emb: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Wang & Isola style uniformity loss on normalized embeddings.
    emb: [B, D]
    """
    if emb.shape[0] <= 1:
        return emb.new_tensor(0.0)

    emb = F.normalize(emb, dim=-1)
    sim = emb @ emb.T
    dist = (2.0 - 2.0 * sim).clamp_min(0.0)
    dist = dist[torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)]
    return torch.log(torch.exp(-t * dist).mean() + 1e-8)


def batch_orthogonal_uniformity_loss(emb: torch.Tensor) -> torch.Tensor:
    """
    AlphaEarth-style batch uniformity proxy.

    Random pairs from a well-spread unit-sphere embedding space should be close
    to orthogonal, so we minimize the absolute dot product after a deterministic
    batch rotation. This is cheap, stable for DDP-local batches, and complements
    the Wang & Isola uniformity term above.
    """
    if emb.shape[0] <= 1:
        return emb.sum() * 0.0

    emb = F.normalize(emb, dim=-1)
    paired = torch.roll(emb, shifts=1, dims=0)
    return (emb * paired).sum(dim=-1).abs().mean()


def embedding_consistency_loss(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage two views of the same sample/time window to produce the same
    embedding. The caller decides whether the teacher side is detached.
    """
    teacher_emb = F.normalize(teacher_emb, dim=-1)
    student_emb = F.normalize(student_emb, dim=-1)
    return ((1.0 - (teacher_emb * student_emb).sum(dim=-1)) * 0.5).mean()


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


def compute_total_loss(
    model_output,
    batch: dict,
    cfg,
    student_output=None,
    teacher_consistency_output=None,
) -> LossOutput:
    losses: dict[str, torch.Tensor] = {}
    training_cfg = cfg.training

    recon_weight = getattr(training_cfg, "reconstruction_weight", 1.0)
    target_loss_weights = getattr(training_cfg, "target_loss_weights", {})
    uniformity_weight = getattr(training_cfg, "uniformity_weight", 0.0)
    batch_uniformity_weight = getattr(training_cfg, "batch_uniformity_weight", 0.0)
    variance_weight = getattr(training_cfg, "variance_weight", 0.0)
    decorrelation_weight = getattr(training_cfg, "decorrelation_weight", 0.0)
    orthogonality_weight = getattr(training_cfg, "orthogonality_weight", 0.0)
    consistency_weight = getattr(training_cfg, "consistency_weight", 0.0)

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
                class_weight=_get_class_weight(training_cfg, name, pred.device),
            )
        else:
            cur = continuous_recon_loss(pred, target, mask=mask)

        target_weight = _get_named_float(target_loss_weights, name, 1.0)
        losses[f"recon/{name}"] = cur
        losses[f"recon_weighted/{name}"] = cur * target_weight
        recon_total = recon_total + cur * target_weight

    losses["recon/total"] = recon_total

    emb = model_output.embedding
    losses["reg/uniformity"] = embedding_uniformity_loss(emb)
    losses["reg/batch_uniformity"] = batch_orthogonal_uniformity_loss(emb)
    losses["reg/variance"] = batch_variance_loss(emb)
    losses["reg/decorrelation"] = decorrelation_loss(emb)
    losses["reg/orthogonality"] = orthogonality_loss(model_output.embedding_map)

    total = recon_weight * losses["recon/total"]
    total = total + uniformity_weight * losses["reg/uniformity"]
    total = total + batch_uniformity_weight * losses["reg/batch_uniformity"]
    total = total + variance_weight * losses["reg/variance"]
    total = total + decorrelation_weight * losses["reg/decorrelation"]
    total = total + orthogonality_weight * losses["reg/orthogonality"]

    if student_output is not None and consistency_weight > 0:
        teacher_ref = teacher_consistency_output if teacher_consistency_output is not None else model_output
        teacher_emb = teacher_ref.embedding.detach()
        losses["consistency/embedding"] = embedding_consistency_loss(
            teacher_emb,
            student_output.embedding,
        )
        total = total + consistency_weight * losses["consistency/embedding"]
    else:
        losses["consistency/embedding"] = emb.sum() * 0.0

    losses["loss"] = total
    return LossOutput(total=total, components=losses)
