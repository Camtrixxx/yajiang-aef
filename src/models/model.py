from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from src.models.bottleneck import VMFBottleneck
from src.models.blocks import STPBlock
from src.models.decoders import ContinuousDecoder, CategoricalDecoder
from src.models.sensor_encoders import SensorEncoderBank
from src.models.time_encoding import (
    TimeCodeEncoder,
    WindowCodeEncoder,
    RelativeTimeCodeEncoder,
)


@dataclass(slots=True)
class AEFOutput:
    embedding_map: torch.Tensor
    embedding: torch.Tensor
    pre_norm_embedding: torch.Tensor
    pre_norm_map: torch.Tensor | None
    reconstructions: dict[str, torch.Tensor]


class AEFModel(nn.Module):
    """
    雅江初版 AEF 模型:
    - 多源输入编码
    - STP 主干
    - VMF bottleneck
    - 条件解码重建
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        d = cfg.data

        input_sources = d.input_sources
        target_sources = d.target_sources

        self.input_sources = input_sources
        self.target_sources = target_sources

        self.sensor_encoder_bank = SensorEncoderBank(
            input_sources=input_sources,
            input_dim=m.input_dim,
            stem_dim=m.stem_dim,
            out_dim=m.precision_dim,
            source_channels=getattr(m, "source_channels", None),
            stem_channels=getattr(m, "stem_channels", None),
        )

        self.time_encoder = TimeCodeEncoder(m.time_code_dim)
        self.window_encoder = WindowCodeEncoder(m.window_code_dim)
        self.relative_time_encoder = RelativeTimeCodeEncoder(m.relative_time_code_dim)

        self.stp_blocks = nn.ModuleList(
            [
                STPBlock(
                    channels=m.precision_dim,
                    num_heads=m.num_heads,
                )
                for _ in range(m.num_blocks)
            ]
        )
        self.use_checkpoint = getattr(m, "gradient_checkpointing", False)

        self.summary_query = nn.Linear(m.window_code_dim, m.precision_dim)

        self.bottleneck = VMFBottleneck(
            channels=m.precision_dim,
            embedding_dim=m.embedding_dim,
            kappa=m.vmf_kappa,
            skip_l2_training=getattr(
                m,
                "skip_l2_training",
                getattr(m, "skip_l2_norm_training", False),
            ),
        )

        # 为每个目标源构建一个 decoder
        self.decoders = nn.ModuleDict()
        for tgt in target_sources:
            name = tgt.name if hasattr(tgt, "name") else tgt["name"]
            loss_type = tgt.loss_type if hasattr(tgt, "loss_type") else tgt["loss_type"]
            out_channels = tgt.out_channels if hasattr(tgt, "out_channels") else tgt["out_channels"]

            if loss_type == "categorical":
                self.decoders[name] = CategoricalDecoder(
                    embedding_dim=m.embedding_dim,
                    window_code_dim=m.window_code_dim,
                    relative_time_code_dim=m.relative_time_code_dim,
                    metadata_dim=m.metadata_dim,
                    out_channels=out_channels,
                    hidden_mult=getattr(m, "decoder_hidden_mult", 1),
                )
            else:
                self.decoders[name] = ContinuousDecoder(
                    embedding_dim=m.embedding_dim,
                    window_code_dim=m.window_code_dim,
                    relative_time_code_dim=m.relative_time_code_dim,
                    metadata_dim=m.metadata_dim,
                    out_channels=out_channels,
                    hidden_mult=getattr(m, "decoder_hidden_mult", 1),
                )

    def encode_frames(
        self,
        source_frames: torch.Tensor,       # [B, S, T, C, H, W]
        source_timestamps_ms: torch.Tensor,# [B, S, T]
        source_frame_mask: torch.Tensor,   # [B, S, T]
        source_input_mask: torch.Tensor,   # [B, S]
        source_type_ids: torch.Tensor,     # [B, S]
        valid_start_ms: torch.Tensor,      # [B]
        valid_end_ms: torch.Tensor,        # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            summary_map: [B, C, H/2, W/2]
            window_code: [B, W]
        """
        encoded = self.sensor_encoder_bank(source_frames, source_type_ids)
        b, s, t, c, h, w = encoded.shape

        frames = encoded.reshape(b, s * t, c, h, w)
        timestamps = source_timestamps_ms.reshape(b, s * t).float()

        effective_mask = (source_frame_mask & source_input_mask[:, :, None]).reshape(b, s * t)

        valid_start = valid_start_ms.float().view(-1)
        valid_end = valid_end_ms.float().view(-1)

        window_mask = (
            (source_timestamps_ms.float() >= valid_start[:, None, None])
            & (source_timestamps_ms.float() <= valid_end[:, None, None])
        ).reshape(b, s * t)

        frame_mask = effective_mask & window_mask

        # 防止某个 batch 在窗口内没有有效帧
        empty_mask = ~frame_mask.any(dim=1)
        if empty_mask.any():
            frame_mask = torch.where(empty_mask[:, None], effective_mask, frame_mask)

        _time_codes = self.time_encoder(timestamps)   # 先保留，后续 loss 或 block 增强可用
        window_code = self.window_encoder(valid_start_ms, valid_end_ms)

        x = frames
        for block in self.stp_blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(block, x, frame_mask, use_reentrant=False)
            else:
                x = block(x, frame_mask)

        pooled = x.mean(dim=(-2, -1))  # [B, T_all, C]
        query = self.summary_query(window_code)[:, None, :]  # [B, 1, C]
        attn_scores = (query * pooled).sum(dim=-1)          # [B, T_all]
        attn_scores = attn_scores.masked_fill(~frame_mask, -1e9)
        attn = torch.softmax(attn_scores, dim=-1)

        summary_map = torch.einsum("bt,btchw->bchw", attn, x)
        return summary_map, window_code

    def forward(
        self,
        source_frames: torch.Tensor,
        source_timestamps_ms: torch.Tensor,
        source_frame_mask: torch.Tensor,
        source_input_mask: torch.Tensor,
        source_type_ids: torch.Tensor,
        valid_start_ms: torch.Tensor,
        valid_end_ms: torch.Tensor,
        target_relative_time: torch.Tensor,   # [B, N_tgt]
        target_metadata: torch.Tensor,        # [B, N_tgt, M]
    ) -> AEFOutput:
        """
        约定:
            target_relative_time 的第 2 维顺序与 cfg.data.target_sources 一致
            target_metadata 的第 2 维顺序与 cfg.data.target_sources 一致
        """
        summary_map, window_code = self.encode_frames(
            source_frames=source_frames,
            source_timestamps_ms=source_timestamps_ms,
            source_frame_mask=source_frame_mask,
            source_input_mask=source_input_mask,
            source_type_ids=source_type_ids,
            valid_start_ms=valid_start_ms,
            valid_end_ms=valid_end_ms,
        )

        embedding_map, embedding, pre_norm_embedding, pre_norm_map = self.bottleneck(summary_map)

        num_tgt = len(self.target_sources)

        if target_relative_time.dim() == 1:
            target_relative_time = target_relative_time[:, None]
        if target_metadata.dim() == 2:
            target_metadata = target_metadata[:, None, :]

        if target_relative_time.shape[1] != num_tgt:
            raise ValueError(
                f"target_relative_time second dim must equal num_tgt={num_tgt}, "
                f"but got {target_relative_time.shape[1]}"
            )
        if target_metadata.shape[1] != num_tgt:
            raise ValueError(
                f"target_metadata second dim must equal num_tgt={num_tgt}, "
                f"but got {target_metadata.shape[1]}"
            )

        reconstructions: dict[str, torch.Tensor] = {}

        for i, tgt in enumerate(self.target_sources):
            tgt_name = tgt.name if hasattr(tgt, "name") else tgt["name"]
            decoder = self.decoders[tgt_name]

            rel_t = target_relative_time[:, i:i+1]                  # [B, 1]
            rel_code = self.relative_time_encoder(rel_t)            # [B, R]
            meta = target_metadata[:, i, :]                         # [B, M]

            out = decoder(
                embedding_map=embedding_map,
                window_code=window_code,
                relative_time=rel_code,
                metadata=meta,
            )
            reconstructions[tgt_name] = out

        return AEFOutput(
            embedding_map=embedding_map,
            embedding=embedding,
            pre_norm_embedding=pre_norm_embedding,
            pre_norm_map=pre_norm_map,
            reconstructions=reconstructions,
        )