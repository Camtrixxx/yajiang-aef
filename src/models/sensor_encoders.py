from __future__ import annotations

import torch
from torch import nn


class SensorEncoder(nn.Module):
    """
    单个传感器编码器:
    adapter(可选) -> stem conv -> projection
    """

    def __init__(
        self,
        in_channels: int,
        stem_channels: int | None = None,
        stem_dim: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        stem_channels = stem_channels if stem_channels is not None else in_channels

        if in_channels != stem_channels:
            self.adapter = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.GELU(),
            )
        else:
            self.adapter = nn.Identity()

        self.stem = nn.Sequential(
            nn.Conv2d(stem_channels, stem_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(stem_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        x = self.stem(x)
        x = self.projection(x)
        return x


class SensorEncoderBank(nn.Module):
    """
    多源编码器组

    第一版固定支持:
        s2 -> 0
        s1 -> 1
        hls -> 2
        landsat -> 3
    """

    SOURCE_NAME_TO_ID = {
        "s2": 0,
        "s1": 1,
        "hls": 2,
        "landsat": 3,
    }
    SOURCE_ID_TO_NAME = {v: k for k, v in SOURCE_NAME_TO_ID.items()}

    def __init__(
        self,
        input_sources: list[str],
        input_dim: int,
        stem_dim: int,
        out_dim: int,
        source_channels: dict[str, int] | None = None,
        stem_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.input_sources = input_sources
        self.input_dim = input_dim
        self.out_dim = out_dim

        if source_channels is None:
            self.source_channels = {}
        elif isinstance(source_channels, dict):
            self.source_channels = source_channels
        else:
            self.source_channels = vars(source_channels)

        self.encoders = nn.ModuleDict()
        for src_name in input_sources:
            if src_name not in self.SOURCE_NAME_TO_ID:
                raise ValueError(f"Unsupported source name: {src_name}")
            in_ch = self.source_channels.get(src_name, input_dim)
            self.encoders[src_name] = SensorEncoder(
                in_channels=in_ch,
                stem_channels=stem_channels,
                stem_dim=stem_dim,
                out_dim=out_dim,
            )

    def forward(
        self,
        source_frames: torch.Tensor,     # [B, S, T, C, H, W]
        source_type_ids: torch.Tensor,   # [B, S]
    ) -> torch.Tensor:
        b, s, t, c, h, w = source_frames.shape
        encoded_list = []

        for s_idx in range(s):
            slot_encoded = None
            for src_id_tensor in torch.unique(source_type_ids[:, s_idx]):
                src_id = int(src_id_tensor.item())
                if src_id not in self.SOURCE_ID_TO_NAME:
                    raise ValueError(f"Unknown source_type_id: {src_id}")
                src_name = self.SOURCE_ID_TO_NAME[src_id]
                if src_name not in self.encoders:
                    raise ValueError(f"Source '{src_name}' is not configured in this encoder bank")

                batch_mask = source_type_ids[:, s_idx] == src_id
                group_b = int(batch_mask.sum().item())
                encoder = self.encoders[src_name]
                in_ch = self.source_channels.get(src_name, self.input_dim)

                frames = source_frames[batch_mask, s_idx]       # [B', T, C, H, W]
                frames = frames[:, :, :in_ch, :, :]             # 只取有效前 in_ch 通道
                frames = frames.reshape(group_b * t, in_ch, h, w)

                enc = encoder(frames)                           # [B'*T, D, H/2, W/2]
                _, d, hh, ww = enc.shape
                enc = enc.reshape(group_b, t, d, hh, ww)

                if slot_encoded is None:
                    slot_encoded = source_frames.new_zeros(b, t, d, hh, ww)
                slot_encoded[batch_mask] = enc

            if slot_encoded is None:
                raise ValueError(f"No source ids found for source slot {s_idx}")
            encoded_list.append(slot_encoded)

        return torch.stack(encoded_list, dim=1)  # [B, S, T, D, H/2, W/2]
