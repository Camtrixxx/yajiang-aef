from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention equivalent to nn.MultiheadAttention(batch_first=True).

    Parameter names match nn.MultiheadAttention (`in_proj_weight`,
    `in_proj_bias`, `out_proj`) so state_dict keys are unchanged and existing
    checkpoints load without translation.

    Why not just use nn.MultiheadAttention: it returns attention weights and
    supports cross-attention, and the generality costs real time at the shapes
    this model uses -- many short sequences rather than few long ones. Measured
    on one A800 under bf16 autocast, forward only:

        time_attn  (4096 seqs x len  48 x dim 256): 3.78 ms -> 1.47 ms  (2.56x)
        space_attn ( 192 seqs x len 256 x dim 256): 1.83 ms -> 0.36 ms  (5.13x)

    Verified equivalent against nn.MultiheadAttention to max|diff| 3.9e-7
    (time) and 1.6e-7 (space) in fp32.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Same initialization as nn.MultiheadAttention.
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,                          # [N, L, C]
        key_padding_mask: torch.Tensor | None = None,  # [N, L], True = ignore
    ) -> torch.Tensor:
        n, length, c = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(z: torch.Tensor) -> torch.Tensor:
            return z.reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA takes True = keep, the opposite of key_padding_mask.
            keep = ~key_padding_mask
            # A row with every key masked makes softmax produce NaN. The caller
            # (STPBlock) already guarantees at least one valid frame per sample,
            # but a fully-masked row must not poison the batch if that changes.
            all_masked = ~keep.any(dim=-1, keepdim=True)
            keep = keep | all_masked
            attn_mask = keep[:, None, None, :]

        out = F.scaled_dot_product_attention(
            split(q), split(k), split(v), attn_mask=attn_mask
        )
        out = out.transpose(1, 2).reshape(n, length, c)
        return self.out_proj(out)


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
        fast_attention: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.fast_attention = fast_attention

        def make_attn():
            if fast_attention:
                return SelfAttention(embed_dim=channels, num_heads=num_heads)
            return nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                batch_first=True,
            )

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
        self.time_attn = make_attn()
        self.time_norm = nn.LayerNorm(channels)
        self.time_up = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

        # Space path
        self.space_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
        )
        self.space_attn = make_attn()
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

        if self.fast_attention:
            x_t_attn = self.time_attn(x_t, key_padding_mask=key_padding_mask)
        else:
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
        if self.fast_attention:
            x_s_attn = self.space_attn(x_s_flat)
        else:
            x_s_attn, _ = self.space_attn(x_s_flat, x_s_flat, x_s_flat)
        x_s_attn = self.space_norm(x_s_attn)
        x_s_attn = x_s_attn.reshape(b, t, h_s, w_s, c_s).permute(0, 1, 4, 2, 3)
        x_s_up = x_s_attn.reshape(b * t, c_s, h_s, w_s)
        x_s_up = self.space_up(x_s_up)

        fused = self.fusion(torch.cat([x_prec, x_t_up, x_s_up], dim=1))
        residual = self.residual_norm(x_flat)
        out = fused + residual
        return out.reshape(b, t, c, h, w)