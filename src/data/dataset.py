from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import ManifestRecord, load_manifest


class YajiangAEFDataset(Dataset):
    """
    与当前 AEFModel.forward() 严格对齐的数据集输出协议。

    期望 manifest 中每条样本类似：
    {
      "sample_id": "patch_0001",
      "valid_start_ms": 1711929600000,
      "valid_end_ms": 1714521600000,
      "split": "train",
      "inputs": {
        "s2": {
          "frames": [
            {"path": ".../s2_1712016000000.npy", "timestamp_ms": 1712016000000},
            {"path": ".../s2_1712880000000.npy", "timestamp_ms": 1712880000000}
          ]
        },
        "s1": {
          "frames": [
            {"path": ".../s1_1712102400000.npy", "timestamp_ms": 1712102400000}
          ]
        }
      },
      "targets": {
        "dem": {
          "path": ".../dem.npy",
          "relative_time": 0.5,
          "metadata": [0.0, 0.0, 0.0, 0.0]
        },
        "worldcover": {
          "path": ".../worldcover.npy",
          "relative_time": 0.5,
          "metadata": [0.0, 0.0, 0.0, 0.0]
        }
      }
    }

    当前 loader 默认支持 .npy / .npz / .pt
    后续你可以很容易扩展到 GeoTIFF。
    """

    SOURCE_NAME_TO_ID = {
        "s2": 0,
        "s1": 1,
        "hls": 2,
    }

    def __init__(self, cfg, manifest_path: str, split: str = "train") -> None:
        self.cfg = cfg
        self.records: list[ManifestRecord] = load_manifest(manifest_path, split=split)

        self.input_sources = list(cfg.data.input_sources)
        self.target_sources = list(cfg.data.target_sources)
        self.max_frames = int(cfg.data.max_frames)
        self.image_size = int(cfg.data.image_size)
        self.metadata_dim = int(cfg.data.metadata_dim)

        source_channels = getattr(cfg.model, "source_channels", {})
        if isinstance(source_channels, dict):
            self.source_channels = source_channels
        else:
            self.source_channels = vars(source_channels)

        self.max_input_channels = max(self.source_channels.values())

        # target 配置表
        self.target_cfg_map: dict[str, Any] = {}
        for tgt in self.target_sources:
            name = tgt.name if hasattr(tgt, "name") else tgt["name"]
            self.target_cfg_map[name] = tgt

    def __len__(self) -> int:
        return len(self.records)

    def _load_array(self, path: str | Path) -> torch.Tensor:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            arr = np.load(path)
            return torch.from_numpy(arr)
        if suffix == ".npz":
            data = np.load(path)
            # 默认取第一个 key
            first_key = list(data.keys())[0]
            return torch.from_numpy(data[first_key])
        if suffix == ".pt":
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, dict):
                # 默认取第一个 tensor
                for v in obj.values():
                    if isinstance(v, torch.Tensor):
                        return v
            raise TypeError(f"Unsupported .pt content in {path}")

        raise ValueError(f"Unsupported file suffix for data loading: {path}")

    def _ensure_chw(self, x: torch.Tensor, expect_channels: int | None = None) -> torch.Tensor:
        """
        输入可能是:
        - [H, W]
        - [C, H, W]
        - [H, W, C]
        统一转成 [C, H, W]
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            # 若最后一维更像通道，则转置
            if x.shape[-1] <= 32 and x.shape[0] > 32:
                x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Expect 2D/3D tensor, got shape={tuple(x.shape)}")

        x = x.float()

        if expect_channels is not None:
            c, h, w = x.shape
            if c < expect_channels:
                pad = torch.zeros(expect_channels - c, h, w, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            elif c > expect_channels:
                x = x[:expect_channels]

        return x

    def _center_crop_or_pad(self, x: torch.Tensor, size: int, mode: str = "continuous") -> torch.Tensor:
        """
        x: [C, H, W]
        mode:
          - continuous: pad 0
          - categorical: pad 0（后续若作为 label，再转 long）
        """
        c, h, w = x.shape

        # pad
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)
            _, h, w = x.shape

        # crop
        if h > size:
            start_h = (h - size) // 2
            x = x[:, start_h:start_h + size, :]
        if w > size:
            start_w = (w - size) // 2
            x = x[:, :, start_w:start_w + size]

        return x

    def _build_source_tensor(self, rec: ManifestRecord):
        num_sources = len(self.input_sources)
        h = w = self.image_size

        source_frames = torch.zeros(num_sources, self.max_frames, self.max_input_channels, h, w)
        source_timestamps_ms = torch.zeros(num_sources, self.max_frames, dtype=torch.long)
        source_frame_mask = torch.zeros(num_sources, self.max_frames, dtype=torch.bool)
        source_input_mask = torch.zeros(num_sources, dtype=torch.bool)
        source_type_ids = torch.zeros(num_sources, dtype=torch.long)

        for s_idx, src_name in enumerate(self.input_sources):
            if src_name not in self.SOURCE_NAME_TO_ID:
                raise ValueError(f"Unsupported source name: {src_name}")
            source_type_ids[s_idx] = self.SOURCE_NAME_TO_ID[src_name]

            payload = rec.inputs.get(src_name)
            if payload is None:
                continue

            frames = payload.get("frames", [])
            if len(frames) == 0:
                continue

            source_input_mask[s_idx] = True
            expected_c = int(self.source_channels[src_name])

            # 按时间排序，再截断到 max_frames
            frames = sorted(frames, key=lambda x: int(x["timestamp_ms"]))[: self.max_frames]

            for t_idx, frame_info in enumerate(frames):
                arr = self._load_array(frame_info["path"])
                arr = self._ensure_chw(arr, expect_channels=expected_c)
                arr = self._center_crop_or_pad(arr, size=self.image_size, mode="continuous")

                # 写到前 expected_c 通道，剩余通道保持 0
                source_frames[s_idx, t_idx, :expected_c] = arr
                source_timestamps_ms[s_idx, t_idx] = int(frame_info["timestamp_ms"])
                source_frame_mask[s_idx, t_idx] = True

        return {
            "source_frames": source_frames,
            "source_timestamps_ms": source_timestamps_ms,
            "source_frame_mask": source_frame_mask,
            "source_input_mask": source_input_mask,
            "source_type_ids": source_type_ids,
        }

    def _build_target_tensor(self, rec: ManifestRecord):
        targets: dict[str, torch.Tensor] = {}
        target_masks: dict[str, torch.Tensor] = {}

        rel_times = []
        metadata_list = []

        # 模型里 decoder 输出分辨率 = encoder downsample 后的 H/2
        target_h = target_w = self.image_size // 2

        for tgt in self.target_sources:
            name = tgt.name if hasattr(tgt, "name") else tgt["name"]
            out_channels = tgt.out_channels if hasattr(tgt, "out_channels") else tgt["out_channels"]
            loss_type = tgt.loss_type if hasattr(tgt, "loss_type") else tgt["loss_type"]

            payload = rec.targets.get(name)
            if payload is None:
                raise ValueError(f"Sample '{rec.sample_id}' missing target '{name}' in manifest")

            arr = self._load_array(payload["path"])
            arr = self._ensure_chw(arr, expect_channels=None)
            arr = self._center_crop_or_pad(arr, size=target_h, mode=loss_type)

            if loss_type == "categorical":
                # 期望 [1, H, W] 或 one-hot；这里统一成 [1, H, W] long
                if arr.shape[0] > 1:
                    arr = arr[:1]
                arr = arr.long()
                mask = torch.ones(1, target_h, target_w, dtype=torch.bool)
            else:
                if arr.shape[0] < out_channels:
                    pad = torch.zeros(out_channels - arr.shape[0], target_h, target_w, dtype=arr.dtype)
                    arr = torch.cat([arr, pad], dim=0)
                elif arr.shape[0] > out_channels:
                    arr = arr[:out_channels]
                arr = arr.float()
                mask = torch.ones(out_channels, target_h, target_w, dtype=torch.bool)

            targets[name] = arr
            target_masks[name] = mask

            rel_times.append(float(payload.get("relative_time", 0.5)))

            meta = payload.get("metadata", [])
            meta = list(meta)[: self.metadata_dim]
            if len(meta) < self.metadata_dim:
                meta = meta + [0.0] * (self.metadata_dim - len(meta))
            metadata_list.append(meta)

        target_relative_time = torch.tensor(rel_times, dtype=torch.float32)
        target_metadata = torch.tensor(metadata_list, dtype=torch.float32)

        return {
            "targets": targets,
            "target_masks": target_masks,
            "target_relative_time": target_relative_time,
            "target_metadata": target_metadata,
        }

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        source_part = self._build_source_tensor(rec)
        target_part = self._build_target_tensor(rec)

        out = {
            **source_part,
            **target_part,
            "valid_start_ms": torch.tensor(rec.valid_start_ms, dtype=torch.long),
            "valid_end_ms": torch.tensor(rec.valid_end_ms, dtype=torch.long),
            "sample_id": rec.sample_id,
        }
        return out


def aef_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()

    for k in keys:
        if k == "sample_id":
            out[k] = [x[k] for x in batch]
        elif isinstance(batch[0][k], dict):
            out[k] = {
                kk: torch.stack([x[k][kk] for x in batch], dim=0)
                for kk in batch[0][k].keys()
            }
        else:
            out[k] = torch.stack([x[k] for x in batch], dim=0)

    return out
