from __future__ import annotations

import os
import random

import numpy as np
import torch


def import_npu_if_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return True


def get_local_rank(default: int = 0) -> int:
    return int(os.environ.get("LOCAL_RANK", default))


def resolve_device(device: str | None = None, local_rank: int | None = None) -> torch.device:
    if local_rank is None:
        local_rank = get_local_rank()

    if device is None or device == "auto":
        if import_npu_if_available() and torch.npu.is_available():
            return torch.device(f"npu:{local_rank}")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    if device.startswith("npu"):
        if not import_npu_if_available():
            raise RuntimeError("Requested NPU device, but torch_npu is not installed.")
        if device == "npu":
            return torch.device(f"npu:{local_rank}")

    if device == "cuda":
        return torch.device(f"cuda:{local_rank}")

    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if import_npu_if_available() and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def should_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"


def build_grad_scaler(device: torch.device, amp_dtype: str):
    if device.type == "cuda" and amp_dtype == "fp16":
        return torch.cuda.amp.GradScaler()
    return None
