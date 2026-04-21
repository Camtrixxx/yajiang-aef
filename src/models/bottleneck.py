from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sample_vmf(mean_direction: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    简化版 vMF 采样近似：
    v = mu + N(0, 1/kappa)，再投影到球面
    """
    if kappa <= 0:
        return mean_direction
    noise = torch.randn_like(mean_direction) * math.sqrt(1.0 / kappa)
    sample = mean_direction + noise
    return F.normalize(sample, p=2, dim=1)


class VMFBottleneck(nn.Module):
    """
    改进版 bottleneck:
    - 训练时可以跳过 L2 norm，保留 pre-norm 幅度
    - 推理时做 L2 norm + vMF noise
    """

    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        kappa: float = 2000.0,
        skip_l2_training: bool = True,
    ) -> None:
        super().__init__()
        self.to_embedding = nn.Conv2d(channels, embedding_dim, kernel_size=1)
        self.embedding_dim = embedding_dim
        self.kappa = kappa
        self.skip_l2_training = skip_l2_training

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W]

        Returns:
            embedding_map: [B, D, H, W]
            embedding_vector: [B, D]
            pre_norm_vector: [B, D]
            pre_norm_map: [B, D, H, W]
        """
        pre_norm_map = self.to_embedding(features)

        if self.training and self.skip_l2_training:
            if self.kappa > 0:
                noise = torch.randn_like(pre_norm_map) * math.sqrt(1.0 / self.kappa)
                embedding_map = pre_norm_map + noise
            else:
                embedding_map = pre_norm_map
        else:
            direction = F.normalize(pre_norm_map, p=2, dim=1)
            if self.kappa > 0:
                embedding_map = sample_vmf(direction, self.kappa)
            else:
                embedding_map = direction

        pre_norm_vector = pre_norm_map.mean(dim=(-2, -1))
        embedding_vector = embedding_map.mean(dim=(-2, -1))

        if not (self.training and self.skip_l2_training):
            embedding_vector = F.normalize(embedding_vector, p=2, dim=1)

        return embedding_map, embedding_vector, pre_norm_vector, pre_norm_map