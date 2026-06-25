from __future__ import annotations

from datetime import datetime
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
        self.log_dir = self.output_dir / "logs"
        self.train_log_path = self.log_dir / "train.log"
        if self.distributed.is_main_process:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.export_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.train_log_path.write_text(
                f"Training started at {datetime.now().isoformat(timespec='seconds')}\n",
                encoding="utf-8",
            )

    def _state_dict(self):
        if hasattr(self.model, "module"):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _log(self, message: str):
        if self.distributed.is_main_process:
            print(message)
            timestamp = datetime.now().isoformat(timespec="seconds")
            with self.train_log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")

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

    def _namespace_to_dict(self, obj):
        if isinstance(obj, SimpleNamespace):
            return {k: self._namespace_to_dict(v) for k, v in vars(obj).items()}
        if isinstance(obj, dict):
            return {k: self._namespace_to_dict(v) for k, v in obj.items()}
        return obj

    def _move_to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]
        return obj

    def _forward_model(self, batch):
        return self.model(
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

    def _clone_batch_for_student(self, batch: dict) -> dict:
        cloned = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            elif isinstance(value, dict):
                cloned[key] = {
                    sub_key: sub_value.clone() if torch.is_tensor(sub_value) else sub_value
                    for sub_key, sub_value in value.items()
                }
            else:
                cloned[key] = value
        return cloned

    def _concat_view_batches(self, teacher_batch: dict, student_batch: dict) -> dict:
        merged = {}
        for key, value in teacher_batch.items():
            other = student_batch[key]
            if torch.is_tensor(value):
                merged[key] = torch.cat([value, other], dim=0)
            elif isinstance(value, dict):
                merged[key] = {
                    sub_key: torch.cat([sub_value, other[sub_key]], dim=0)
                    if torch.is_tensor(sub_value)
                    else sub_value
                    for sub_key, sub_value in value.items()
                }
            else:
                merged[key] = value
        return merged

    def _split_model_output(self, output, batch_size: int):
        reconstructions = {
            key: value[:batch_size]
            for key, value in output.reconstructions.items()
        }
        student_reconstructions = {
            key: value[batch_size:]
            for key, value in output.reconstructions.items()
        }
        output_cls = type(output)
        teacher_output = output_cls(
            embedding_map=output.embedding_map[:batch_size],
            embedding=output.embedding[:batch_size],
            pre_norm_embedding=output.pre_norm_embedding[:batch_size],
            pre_norm_map=output.pre_norm_map[:batch_size] if output.pre_norm_map is not None else None,
            reconstructions=reconstructions,
        )
        student_output = output_cls(
            embedding_map=output.embedding_map[batch_size:],
            embedding=output.embedding[batch_size:],
            pre_norm_embedding=output.pre_norm_embedding[batch_size:],
            pre_norm_map=output.pre_norm_map[batch_size:] if output.pre_norm_map is not None else None,
            reconstructions=student_reconstructions,
        )
        return teacher_output, student_output

    def _drop_frames_for_source(
        self,
        frame_mask: torch.Tensor,
        batch_idx: int,
        source_idx: int,
        drop_prob: float,
    ) -> None:
        valid = frame_mask[batch_idx, source_idx]
        if not valid.any() or drop_prob <= 0:
            return
        keep = torch.rand(valid.shape, device=valid.device) >= drop_prob
        updated = valid & keep
        if updated.any():
            frame_mask[batch_idx, source_idx] = updated

    def _drop_temporal_half(
        self,
        frame_mask: torch.Tensor,
        batch_idx: int,
        drop_latter_half: bool,
    ) -> None:
        for source_idx in range(frame_mask.shape[1]):
            valid_indices = torch.nonzero(frame_mask[batch_idx, source_idx], as_tuple=False).flatten()
            if valid_indices.numel() <= 1:
                continue
            midpoint = max(1, valid_indices.numel() // 2)
            drop_indices = valid_indices[midpoint:] if drop_latter_half else valid_indices[:midpoint]
            candidate = frame_mask[batch_idx, source_idx].clone()
            candidate[drop_indices] = False
            if candidate.any():
                frame_mask[batch_idx, source_idx] = candidate

    def make_student_batch(self, batch: dict) -> dict | None:
        training_cfg = self.cfg.training
        consistency_weight = float(getattr(training_cfg, "consistency_weight", 0.0))
        if consistency_weight <= 0:
            return None

        cfg = self._namespace_to_dict(getattr(training_cfg, "student_perturbation", {}))
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            return None

        student = self._clone_batch_for_student(batch)
        frame_mask = student["source_frame_mask"]
        input_mask = student["source_input_mask"]
        source_frames = student["source_frames"]

        source_names = list(getattr(self.cfg.data, "input_sources", []))
        source_drop_probs = cfg.get("source_drop_probs", {})
        frame_drop_probs = cfg.get("frame_drop_probs", {})
        default_frame_drop_prob = float(cfg.get("frame_drop_prob", 0.0))
        half_sequence_drop_prob = float(cfg.get("half_sequence_drop_prob", 0.0))

        batch_size, num_sources, _ = frame_mask.shape
        original_effective = frame_mask & input_mask[:, :, None]

        for batch_idx in range(batch_size):
            for source_idx, source_name in enumerate(source_names):
                source_drop_prob = float(source_drop_probs.get(source_name, 0.0))
                if input_mask[batch_idx, source_idx] and source_drop_prob > 0:
                    if torch.rand((), device=frame_mask.device) < source_drop_prob:
                        input_mask[batch_idx, source_idx] = False
                        frame_mask[batch_idx, source_idx] = False
                        source_frames[batch_idx, source_idx] = 0
                        continue

                frame_drop_prob = float(frame_drop_probs.get(source_name, default_frame_drop_prob))
                self._drop_frames_for_source(frame_mask, batch_idx, source_idx, frame_drop_prob)

            if half_sequence_drop_prob > 0:
                if torch.rand((), device=frame_mask.device) < half_sequence_drop_prob:
                    drop_latter_half = bool(torch.rand((), device=frame_mask.device) < 0.5)
                    self._drop_temporal_half(frame_mask, batch_idx, drop_latter_half)

            effective = frame_mask[batch_idx] & input_mask[batch_idx, :, None]
            if not effective.any():
                frame_mask[batch_idx] = original_effective[batch_idx]
                input_mask[batch_idx] = original_effective[batch_idx].any(dim=1)

        return student

    def save_checkpoint(self, name: str):
        if not self.distributed.is_main_process:
            return

        ckpt_path = self.ckpt_dir / name
        payload = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "best_loss": self.state.best_loss,
            "model": self._state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()

        torch.save(payload, ckpt_path)
        self._log(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        override_optimizer_lr: bool = True,
    ) -> None:
        ckpt_path = Path(path)
        payload = torch.load(ckpt_path, map_location="cpu")

        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(payload["model"], strict=True)

        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            if override_optimizer_lr:
                lr = float(getattr(self.cfg.training, "lr", self.optimizer.param_groups[0]["lr"]))
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

        if self.scaler is not None and "scaler" in payload:
            self.scaler.load_state_dict(payload["scaler"])

        self.state.epoch = int(payload.get("epoch", 0))
        self.state.global_step = int(payload.get("global_step", 0))
        self.state.best_loss = float(payload.get("best_loss", float("inf")))
        self._log(
            f"Loaded checkpoint from {ckpt_path} "
            f"(epoch={self.state.epoch}, global_step={self.state.global_step})"
        )

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
                student_batch = self.make_student_batch(batch)
                if student_batch is not None:
                    merged_batch = self._concat_view_batches(batch, student_batch)
                    merged_output = self._forward_model(merged_batch)
                    output, student_output = self._split_model_output(
                        merged_output,
                        batch["source_frames"].shape[0],
                    )
                else:
                    output = self._forward_model(batch)
                    student_output = None
                loss_out = compute_total_loss(
                    output,
                    batch,
                    self.cfg,
                    student_output=student_output,
                )
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

        for epoch in range(self.state.epoch, epochs):
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
        experiment_name = getattr(self.cfg.experiment, "name", "aef_model")
        self.export_deploy_model(f"{experiment_name}_deploy.pt")
