from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .losses import compute_total_loss
from src.utils.distributed import DistributedState, reduce_metric


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
        device: str | torch.device = "auto",
        scaler=None,
        distributed: DistributedState | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.scaler = scaler
        self.distributed = distributed or DistributedState(enabled=False)
        self.state = TrainState()
        self.use_amp = getattr(cfg.training, "amp", True) and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if getattr(cfg.training, "amp_dtype", "bf16") == "bf16"
            else torch.float16
        )

        exp_output_dir = getattr(cfg.experiment, "output_dir", "./outputs/default")
        self.output_dir = Path(exp_output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.export_dir = self.output_dir / "exports"
        if self.distributed.is_main_process:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.export_dir.mkdir(parents=True, exist_ok=True)

    def _state_dict(self):
        if hasattr(self.model, "module"):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _log(self, message: str):
        if self.distributed.is_main_process:
            print(message)

    def _config_to_dict(self, obj):
        if isinstance(obj, SimpleNamespace):
            return {k: self._config_to_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, dict):
            return {k: self._config_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._config_to_dict(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        return obj

    def save_checkpoint(self, name: str):
        if not self.distributed.is_main_process:
            return

        ckpt_path = self.ckpt_dir / name
        payload = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model": self._state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()

        torch.save(payload, ckpt_path)
        self._log(f"Saved checkpoint to {ckpt_path}")

    def export_deploy_model(self, name: str = "deploy.pt"):
        if not self.distributed.is_main_process:
            return

        export_path = self.export_dir / name
        payload = {
            "model": self._state_dict(),
            "config": self._config_to_dict(self.cfg),
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "format": "aef_deploy_v1",
        }
        torch.save(payload, export_path)
        self._log(f"Exported deploy model to {export_path}")

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

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
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
                avg_loss = reduce_metric(meter["loss"] / count, self.device)
                elapsed = time.time() - start_time
                self._log(
                    f"[epoch {self.state.epoch:03d} step {step+1:04d}] "
                    f"loss={avg_loss:.4f} "
                    f"time={elapsed:.1f}s"
                )

        for k in list(meter.keys()):
            meter[k] /= max(count, 1)
            meter[k] = reduce_metric(meter[k], self.device)

        return meter

    def fit(self):
        epochs = int(self.cfg.training.epochs)
        save_every = int(getattr(self.cfg.training, "save_every", 0))
        save_epoch_checkpoints = bool(getattr(self.cfg.training, "save_epoch_checkpoints", False))

        for epoch in range(epochs):
            self.state.epoch = epoch + 1
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            train_metrics = self.train_one_epoch()
            self._log(f"Epoch {self.state.epoch} done: {train_metrics}")

            loss = float(train_metrics.get("loss", float("inf")))
            if loss < self.state.best_loss:
                self.state.best_loss = loss
                self.save_checkpoint("best.pt")

            if save_epoch_checkpoints and save_every > 0 and self.state.epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{self.state.epoch:03d}.pt")

            self.save_checkpoint("latest.pt")

        self.save_checkpoint("final.pt")
        self.export_deploy_model("aef_hyh_yajiang_v0_3_c_deploy.pt")
