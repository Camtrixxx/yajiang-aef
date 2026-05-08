from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def get_distributed_state() -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedState(
        enabled=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def init_distributed(device: torch.device) -> DistributedState:
    state = get_distributed_state()
    if not state.enabled:
        return state

    if device.type == "npu":
        torch.npu.set_device(device)
        backend = "hccl"
    elif device.type == "cuda":
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return state


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_metric(value: float, device: torch.device, average: bool = True) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value

    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor = tensor / dist.get_world_size()
    return float(tensor.detach().cpu().item())
