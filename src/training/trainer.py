from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .losses import compute_total_loss


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_loader: DataLoader,
        cfg,
        device: str = "cuda",
        scaler=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = torch.device(device)
        self.scaler = scaler
        self.state = TrainState()
        self.use_amp = getattr(cfg.training, "amp", True) and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if getattr(cfg.training, "amp_dtype", "bf16") == "bf16" else torch.float16

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        return obj

    def train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        log_interval = getattr(self.cfg.training, "log_interval", 20)
        grad_clip_norm = getattr(self.cfg.training, "grad_clip_norm", None)

        meter: dict[str, float] = {}
        count = 0
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            batch = self._move_to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(
                    source_frames=batch["source_frames"],
                    source_timestamps_ms=batch["source_timestamps_ms"],
                    source_frame_mask=batch["source_frame_mask"],
                    source_input_mask=batch["source_input_mask"],
                    source_type_ids=batch["source_type_ids"],
                    valid_start_ms=batch["valid_start_ms"],
                    valid_end_ms=batch["valid_end_ms"],
                    target_relative_time=batch["target_relative_time"],
                    target_metadata=batch["target_metadata"],
                )
                loss_out = compute_total_loss(output, batch, self.cfg)
                loss = loss_out.total

            if self.scaler is not None and self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

            self.state.global_step += 1
            count += 1

            for k, v in loss_out.components.items():
                meter[k] = meter.get(k, 0.0) + float(v.detach().item())

            if (step + 1) % log_interval == 0:
                avg_loss = meter["loss"] / count
                elapsed = time.time() - start_time
                print(
                    f"[epoch {self.state.epoch:03d} step {step+1:04d}] "
                    f"loss={avg_loss:.4f} "
                    f"time={elapsed:.1f}s"
                )

        for k in list(meter.keys()):
            meter[k] /= max(count, 1)

        return meter

    def fit(self):
        epochs = int(self.cfg.training.epochs)
        for epoch in range(epochs):
            self.state.epoch = epoch + 1
            train_metrics = self.train_one_epoch()
            print(f"Epoch {self.state.epoch} done: {train_metrics}")
