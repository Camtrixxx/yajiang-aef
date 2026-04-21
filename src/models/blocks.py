from __future__ import annotations

import torch
from torch import nn


class STPBlock(nn.Module):
    """
    Space-Time-Precision block.

    输入:
        x: [B, T, C, H, W]
        frame_mask: [B, T]，True 表示有效帧

    输出:
        [B, T, C, H, W]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Precision path
        self.precision_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

        # Time path
        self.time_down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.time_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.time_norm = nn.LayerNorm(channels)
        self.time_up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

        # Space path
        self.space_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        )
        self.space_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.space_norm = nn.LayerNorm(channels)
        self.space_up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.residual_norm = nn.GroupNorm(8, channels)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, H, W]
            frame_mask: [B, T], True 表示有效帧
        """
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)

        # Precision path
        x_prec = self.precision_conv(x_flat)

        # Time path
        x_t = self.time_down(x_flat)  # [B*T, C, H/2, W/2]
        _, c_t, h_t, w_t = x_t.shape
        x_t = x_t.reshape(b, t, c_t, h_t * w_t).permute(0, 3, 1, 2)  # [B, HW, T, C]
        x_t = x_t.reshape(b * h_t * w_t, t, c_t)

        key_padding_mask = None
        if frame_mask is not None:
            # MultiheadAttention 里 True 表示忽略
            key_padding_mask = (~frame_mask).repeat_interleave(h_t * w_t, dim=0)

        x_t_attn, _ = self.time_attn(
            x_t, x_t, x_t,
            key_padding_mask=key_padding_mask,
        )
        x_t_attn = self.time_norm(x_t_attn)
        x_t_attn = x_t_attn.reshape(b, h_t * w_t, t, c_t).permute(0, 2, 3, 1)
        x_t_up = x_t_attn.reshape(b * t, c_t, h_t, w_t)
        x_t_up = self.time_up(x_t_up)

        # Space path
        x_s = self.space_down(x_flat)  # [B*T, C, H/4, W/4]
        _, c_s, h_s, w_s = x_s.shape
        x_s_flat = x_s.reshape(b, t, c_s, h_s * w_s).permute(0, 1, 3, 2)
        x_s_flat = x_s_flat.reshape(b * t, h_s * w_s, c_s)
        x_s_attn, _ = self.space_attn(x_s_flat, x_s_flat, x_s_flat)
        x_s_attn = self.space_norm(x_s_attn)
        x_s_attn = x_s_attn.reshape(b, t, h_s, w_s, c_s).permute(0, 1, 4, 2, 3)
        x_s_up = x_s_attn.reshape(b * t, c_s, h_s, w_s)
        x_s_up = self.space_up(x_s_up)

        fused = self.fusion(torch.cat([x_prec, x_t_up, x_s_up], dim=1))
        residual = self.residual_norm(x_flat)
        out = fused + residual
        return out.reshape(b, t, c, h, w)