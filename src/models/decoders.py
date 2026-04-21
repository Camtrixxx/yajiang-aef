from __future__ import annotations

import torch
from torch import nn


class ConditionInjector(nn.Module):
    """
    将 window_code / relative_time / metadata 注入 embedding_map
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
    ) -> None:
        super().__init__()
        total_cond = window_code_dim + relative_time_code_dim + metadata_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(total_cond, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim * 2),
        )
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,   # [B, C, H, W]
        window_code: torch.Tensor,     # [B, W]
        relative_time: torch.Tensor,   # [B, R]
        metadata: torch.Tensor,        # [B, M]
    ) -> torch.Tensor:
        cond = torch.cat([window_code, relative_time, metadata], dim=-1)
        cond_features = self.cond_proj(cond)      # [B, 2C]
        gate = self.gate(cond_features)           # [B, C]

        pooled = embedding_map.mean(dim=(-2, -1))  # [B, C]
        gated = pooled * gate
        return embedding_map + gated[:, :, None, None]


class ContinuousDecoder(nn.Module):
    """
    连续值目标解码器:
    例如 S2/HLS 重建、DEM 重建
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim=embedding_dim,
            window_code_dim=window_code_dim,
            relative_time_code_dim=relative_time_code_dim,
            metadata_dim=metadata_dim,
        )
        hidden = embedding_dim * hidden_mult
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        return self.head(x)


class CategoricalDecoder(nn.Module):
    """
    类别目标解码器:
    例如 WorldCover、JRC Water
    """

    def __init__(
        self,
        embedding_dim: int,
        window_code_dim: int,
        relative_time_code_dim: int,
        metadata_dim: int,
        out_channels: int,
        hidden_mult: int = 1,
    ) -> None:
        super().__init__()
        self.injector = ConditionInjector(
            embedding_dim=embedding_dim,
            window_code_dim=window_code_dim,
            relative_time_code_dim=relative_time_code_dim,
            metadata_dim=metadata_dim,
        )
        hidden = embedding_dim * hidden_mult
        self.head = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        embedding_map: torch.Tensor,
        window_code: torch.Tensor,
        relative_time: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        x = self.injector(embedding_map, window_code, relative_time, metadata)
        return self.head(x)