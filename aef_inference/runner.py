from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.eval.features import batch_to_device, load_deploy_model
from src.utils.device import resolve_device, set_seed


WORLD_COVER_CLASSES = {
    0: {"label": "tree_cover", "label_zh": "林地", "esa_code": 10, "color": (0, 100, 0)},
    1: {"label": "grassland", "label_zh": "草地", "esa_code": 30, "color": (255, 255, 76)},
    2: {"label": "cropland", "label_zh": "耕地", "esa_code": 40, "color": (240, 150, 255)},
    3: {"label": "built_up", "label_zh": "建设用地", "esa_code": 50, "color": (250, 0, 0)},
    4: {"label": "bare_sparse_vegetation", "label_zh": "裸地/稀疏植被", "esa_code": 60, "color": (180, 180, 180)},
    5: {"label": "snow_and_ice", "label_zh": "冰雪", "esa_code": 70, "color": (240, 240, 240)},
    6: {"label": "permanent_water_bodies", "label_zh": "永久水体", "esa_code": 80, "color": (0, 100, 200)},
    7: {"label": "moss_and_lichen", "label_zh": "苔藓/地衣", "esa_code": 100, "color": (250, 230, 160)},
    8: {"label": "shrubland", "label_zh": "灌木地", "esa_code": 20, "color": (255, 187, 34)},
}

SERVICE_CACHE_VERSION = "v7"
SUPPORTED_TASKS = {"all", "water", "landcover", "dem"}
TASK_ALIASES = {
    "jrc": "water",
    "jrc_water": "water",
    "water_classification": "water",
    "水体": "water",
    "水体分类": "water",
    "地物": "landcover",
    "地物分类": "landcover",
    "worldcover": "landcover",
    "land_cover": "landcover",
    "高程": "dem",
    "高程重建": "dem",
    "地形": "dem",
    "地形重建": "dem",
    "dem重建": "dem",
}


@dataclass(slots=True)
class AEFRunnerConfig:
    config_path: Path = Path("configs/yajiang_v1_2.yaml")
    manifest_path: Path = Path("data/full_npy/train.jsonl")
    deploy_model_path: Path = Path("outputs/aef_hyh_yajiang_v1_2/exports/aef_hyh_yajiang_v1_2_deploy.pt")
    cache_dir: Path = Path("outputs/aef_inference_service")
    device: str = "auto"
    seed: int = 42


def _finite_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _normalize_image(x: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _pca_rgb(feature_map: torch.Tensor) -> np.ndarray:
    x = feature_map.detach().cpu()[0]
    c, h, w = x.shape
    flat = x.permute(1, 2, 0).reshape(h * w, c).float().numpy()
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    comps = flat @ vt[:3].T
    return _normalize_image(comps.reshape(h, w, 3))


def _make_s2_rgb(source_frames: torch.Tensor) -> np.ndarray:
    s2 = source_frames[0, 0].float().numpy()
    return np.stack(
        [
            _normalize_image(s2[2]),
            _normalize_image(s2[1]),
            _normalize_image(s2[0]),
        ],
        axis=-1,
    )


def _make_source_rgb(arr: np.ndarray, source: str) -> np.ndarray:
    if arr.ndim == 2:
        gray = _normalize_image(arr)
        return np.stack([gray, gray, gray], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] <= 32 and arr.shape[0] > 32:
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim != 3:
        raise ValueError(f"expected 2D or 3D source array, got shape={arr.shape}")

    source = source.lower()
    if source in {"s2", "sentinel2", "sentinel-2", "landsat"}:
        if arr.shape[0] < 3:
            raise ValueError(f"{source} RGB rendering needs at least 3 channels, got {arr.shape[0]}")
        return np.stack(
            [
                _normalize_image(arr[2]),
                _normalize_image(arr[1]),
                _normalize_image(arr[0]),
            ],
            axis=-1,
        )

    if source in {"s1", "sentinel1", "sentinel-1"}:
        vv = _normalize_image(arr[0])
        vh = _normalize_image(arr[1] if arr.shape[0] > 1 else arr[0])
        ratio = _normalize_image(arr[0] - (arr[1] if arr.shape[0] > 1 else arr[0]))
        return np.stack([vv, vh, ratio], axis=-1)

    raise ValueError(f"unsupported RGB source: {source}")


def _resize_numpy_image(arr: np.ndarray, size: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    tensor = torch.from_numpy(arr)
    if tensor.ndim == 2:
        kwargs = {}
        interp_mode = "nearest" if mode == "nearest" else "bilinear"
        if interp_mode == "bilinear":
            kwargs["align_corners"] = False
        resized = F.interpolate(tensor[None, None].float(), size=size, mode=interp_mode, **kwargs)[0, 0]
        return resized.numpy()
    if tensor.ndim == 3:
        resized = F.interpolate(
            tensor.permute(2, 0, 1)[None].float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
        return resized.numpy()
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _categorical_prediction(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    pred = logits.detach().cpu().argmax(dim=1, keepdim=True).float()
    return F.interpolate(pred, size=size, mode="nearest").long()[:, 0]


def _landcover_prediction_from_logits(logits: torch.Tensor, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    logits = F.interpolate(logits.detach().cpu(), size=size, mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax(dim=0)
    confidence = probs.max(dim=0).values
    return pred.numpy(), confidence.numpy()


def _continuous_prediction(pred: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(pred.detach().cpu(), size=size, mode="bilinear", align_corners=False)[:, 0]


def _water_score_from_logits(logits: torch.Tensor, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = F.interpolate(logits.detach().cpu(), size=size, mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[0]
    num_classes = int(probs.shape[0])
    levels = torch.linspace(0.0, 1.0, num_classes, dtype=probs.dtype).view(num_classes, 1, 1)
    score = (probs * levels).sum(dim=0)
    hard_class = probs.argmax(dim=0)
    confidence = probs.max(dim=0).values
    return score.numpy(), hard_class.numpy(), confidence.numpy()


def _class_distribution(pred: np.ndarray) -> list[dict[str, Any]]:
    values, counts = np.unique(pred.astype(np.int64), return_counts=True)
    total = max(int(counts.sum()), 1)
    rows = []
    for value, count in zip(values.tolist(), counts.tolist(), strict=False):
        class_info = WORLD_COVER_CLASSES.get(int(value), {})
        rows.append(
            {
                "class_id": int(value),
                "label": class_info.get("label", f"class_{int(value)}"),
                "label_zh": class_info.get("label_zh", f"类别{int(value)}"),
                "esa_worldcover_code": class_info.get("esa_code"),
                "pixels": int(count),
                "ratio": float(count / total),
            }
        )
    return sorted(rows, key=lambda row: row["ratio"], reverse=True)


def _landcover_colorize(pred: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pred.shape, 3), dtype=np.float32)
    for class_id, info in WORLD_COVER_CLASSES.items():
        rgb[pred == class_id] = np.asarray(info["color"], dtype=np.float32) / 255.0
    return rgb


def _confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 9) -> list[list[int]]:
    valid = (target >= 0) & (target < num_classes)
    pred = pred[valid].astype(np.int64)
    target = target[valid].astype(np.int64)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(target.tolist(), pred.tolist(), strict=False):
        if 0 <= pred_id < num_classes:
            matrix[true_id, pred_id] += 1
    return matrix.tolist()


def _class_accuracy_rows(pred: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for class_id, info in WORLD_COVER_CLASSES.items():
        mask = target == class_id
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int(((pred == class_id) & mask).sum())
        rows.append(
            {
                "class_id": class_id,
                "label": info["label"],
                "label_zh": info["label_zh"],
                "esa_worldcover_code": info["esa_code"],
                "target_pixels": total,
                "correct_pixels": correct,
                "accuracy": float(correct / total) if total else None,
            }
        )
    return rows


def _binary_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    if int(valid.sum()) == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
            "true_positive_pixels": 0,
            "true_negative_pixels": 0,
            "false_positive_pixels": 0,
            "false_negative_pixels": 0,
            "valid_target_pixels": 0,
            "evaluation_status": "no_valid_target_pixels",
        }
    pred = pred.astype(bool) & valid
    target = target.astype(bool) & valid
    tp = int((pred & target).sum())
    tn = int((~pred & ~target & valid).sum())
    fp = int((pred & ~target).sum())
    fn = int((~pred & target).sum())
    total = max(int(valid.sum()), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "true_positive_pixels": tp,
        "true_negative_pixels": tn,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "valid_target_pixels": int(valid.sum()),
        "evaluation_status": "ok",
    }


def _regression_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    if int(valid.sum()) == 0:
        return {
            "mae": None,
            "rmse": None,
            "bias": None,
            "r2": None,
            "pearson_r": None,
            "valid_target_pixels": 0,
            "evaluation_status": "no_valid_target_pixels",
        }
    y_pred = pred[valid].astype(np.float64)
    y_true = target[valid].astype(np.float64)
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    pearson = np.corrcoef(y_pred, y_true)[0, 1] if y_pred.size > 1 and np.std(y_pred) > 0 and np.std(y_true) > 0 else None
    return {
        "mae": _finite_float(mae),
        "rmse": _finite_float(rmse),
        "bias": _finite_float(bias),
        "r2": _finite_float(r2),
        "pearson_r": _finite_float(pearson),
        "valid_target_pixels": int(valid.sum()),
        "evaluation_status": "ok",
    }


def _hillshade(dem: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float64)
    safe_dem = np.nan_to_num(dem, nan=float(np.nanmean(dem)) if np.isfinite(dem).any() else 0.0)
    dy, dx = np.gradient(safe_dem)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shaded + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def _slope_per_cell(dem: np.ndarray) -> np.ndarray:
    dem = np.asarray(dem, dtype=np.float64)
    safe_dem = np.nan_to_num(dem, nan=float(np.nanmean(dem)) if np.isfinite(dem).any() else 0.0)
    dy, dx = np.gradient(safe_dem)
    return np.hypot(dx, dy).astype(np.float32)


def _smooth_array(arr: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    for _ in range(max(iterations, 0)):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out


def _elevation_zones(dem: np.ndarray, num_zones: int = 5) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    valid = np.isfinite(dem)
    if not valid.any():
        zones = np.zeros_like(dem, dtype=np.int64)
        return zones, [], np.array([0.0, 1.0], dtype=np.float32)

    lo = float(np.nanmin(dem[valid]))
    hi = float(np.nanmax(dem[valid]))
    if hi <= lo:
        edges = np.linspace(lo, lo + 1.0, num_zones + 1)
    else:
        edges = np.linspace(lo, hi, num_zones + 1)

    zones = np.digitize(dem, edges[1:-1], right=False).astype(np.int64)
    zones[~valid] = -1
    total = max(int(valid.sum()), 1)
    rows = []
    for zone_id in range(num_zones):
        mask = zones == zone_id
        rows.append(
            {
                "zone_id": zone_id,
                "label": f"{edges[zone_id]:.0f}-{edges[zone_id + 1]:.0f}",
                "min_elevation": _finite_float(edges[zone_id]),
                "max_elevation": _finite_float(edges[zone_id + 1]),
                "pixels": int(mask.sum()),
                "ratio": float(mask.sum() / total),
            }
        )
    return zones, rows, edges.astype(np.float32)


def _profile_values(dem: np.ndarray, axis: str = "east_west") -> np.ndarray:
    if axis == "north_south":
        return dem[:, dem.shape[1] // 2]
    return dem[dem.shape[0] // 2, :]


def _normalize_task(task: str) -> str:
    key = str(task or "all").strip().lower()
    key = TASK_ALIASES.get(key, key)
    if key not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task: {task}; supported tasks: {sorted(SUPPORTED_TASKS)}")
    return key


def _threshold_slug(value: float) -> str:
    return f"t{int(round(value * 100)):03d}"


def _period_slug(period: str | None) -> str:
    value = str(period or "latest").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


class AEFInferenceRunner:
    def __init__(self, config: AEFRunnerConfig) -> None:
        set_seed(config.seed)
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.cache_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.cache_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = load_config(str(config.config_path))
        self.device = resolve_device(config.device)
        self.model, self.deploy_cfg = load_deploy_model(config.deploy_model_path, device=self.device)
        self.dataset = YajiangAEFDataset(cfg=self.cfg, manifest_path=str(config.manifest_path), split="train")
        self.dem_stats = self._load_dem_stats()
        self.lock = threading.Lock()

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    def meta(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "yajiang-aef-inference",
            "config_path": str(self.config.config_path),
            "manifest_path": str(self.config.manifest_path),
            "deploy_model_path": str(self.config.deploy_model_path),
            "device": str(self.device),
            "dataset_size": self.dataset_size,
            "input_sources": list(self.cfg.data.input_sources),
            "target_sources": [
                target.name if hasattr(target, "name") else target["name"]
                for target in self.cfg.data.target_sources
            ],
            "dem_stats": self.dem_stats,
        }

    def _load_dem_stats(self) -> dict[str, Any] | None:
        meta_path = Path(self.config.manifest_path).parent / "preprocess_meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        dem = meta.get("dem", {})
        if dem.get("normalization") != "zscore":
            return None
        mean = dem.get("mean")
        std = dem.get("std")
        if mean is None or std is None:
            return None
        return {"normalization": "zscore", "mean": float(mean), "std": float(std)}

    def infer(
        self,
        sample_indices: list[int],
        task: str = "all",
        use_cache: bool = True,
        water_threshold: float = 0.5,
        rgb_source: str = "s2",
        rgb_period: str | None = None,
    ) -> dict[str, Any]:
        task = _normalize_task(task)
        water_threshold = float(water_threshold)
        if not 0.0 <= water_threshold <= 1.0:
            raise ValueError(f"water_threshold must be in [0, 1], got {water_threshold}")

        clean_indices = []
        for idx in sample_indices:
            if idx < 0 or idx >= self.dataset_size:
                raise ValueError(f"sample index out of range: {idx}, dataset size={self.dataset_size}")
            if idx not in clean_indices:
                clean_indices.append(idx)
        if not clean_indices:
            raise ValueError("sample_indices must not be empty")

        items = [
            self._infer_one(
                idx,
                task=task,
                water_threshold=water_threshold,
                rgb_source=rgb_source,
                rgb_period=rgb_period,
                use_cache=use_cache,
            )
            for idx in clean_indices
        ]
        return {
            "status": "ok",
            "service": "yajiang-aef-inference",
            "task": task,
            "water_threshold": water_threshold,
            "rgb_source": rgb_source,
            "rgb_period": rgb_period or "latest",
            "sample_indices": clean_indices,
            "model": {
                "deploy_model_path": str(self.config.deploy_model_path),
                "config_path": str(self.config.config_path),
                "device": str(self.device),
            },
            "summary": self._summarize(items, task=task, water_threshold=water_threshold),
            "items": items,
        }

    def _infer_one(
        self,
        sample_index: int,
        *,
        task: str,
        water_threshold: float,
        rgb_source: str,
        rgb_period: str | None,
        use_cache: bool,
    ) -> dict[str, Any]:
        cache_parts = [
            SERVICE_CACHE_VERSION,
            f"sample_{sample_index:06d}",
            task,
            rgb_source.lower(),
            _period_slug(rgb_period),
        ]
        if task in {"all", "water"}:
            cache_parts.append(_threshold_slug(water_threshold))
        cache_path = self.metrics_dir / ("_".join(cache_parts) + ".json")
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        batch = aef_collate_fn([self.dataset[sample_index]])
        with self.lock:
            with torch.no_grad():
                device_batch = batch_to_device(batch, self.device)
                output = self.model(
                    source_frames=device_batch["source_frames"],
                    source_timestamps_ms=device_batch["source_timestamps_ms"],
                    source_frame_mask=device_batch["source_frame_mask"],
                    source_input_mask=device_batch["source_input_mask"],
                    source_type_ids=device_batch["source_type_ids"],
                    valid_start_ms=device_batch["valid_start_ms"],
                    valid_end_ms=device_batch["valid_end_ms"],
                    target_relative_time=device_batch["target_relative_time"],
                    target_metadata=device_batch["target_metadata"],
                )

        metrics, artifacts = self._build_task_outputs(
            sample_index=sample_index,
            task=task,
            batch=batch,
            output=output,
            water_threshold=water_threshold,
            rgb_source=rgb_source,
            rgb_period=rgb_period,
        )

        payload = {
            "sample_index": sample_index,
            "sample_id": batch["sample_id"][0],
            "valid_start_ms": int(batch["valid_start_ms"][0].item()),
            "valid_end_ms": int(batch["valid_end_ms"][0].item()),
            "task": task,
            "water_threshold": water_threshold,
            "rgb_source": rgb_source,
            "rgb_period": rgb_period or "latest",
            "metrics": metrics,
            "artifacts": artifacts,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def render_patch_rgb(
        self,
        sample_index: int,
        *,
        source: str = "s2",
        period: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if sample_index < 0 or sample_index >= self.dataset_size:
            raise ValueError(f"sample index out of range: {sample_index}, dataset size={self.dataset_size}")

        source = str(source or "s2").strip().lower()
        frame = self._select_input_frame(sample_index=sample_index, source=source, period=period)
        actual_period = Path(frame["path"]).stem
        out_path = self.artifact_dir / f"sample_{sample_index:06d}_{source}_{_period_slug(actual_period)}_rgb.png"
        if not use_cache or not out_path.exists():
            arr = np.load(frame["path"])
            rgb = _make_source_rgb(arr, source)
            self._save_rgb(out_path, rgb)

        rec = self.dataset.records[sample_index]
        return {
            "sample_index": sample_index,
            "sample_id": rec.sample_id,
            "source": source,
            "period": actual_period,
            "timestamp_ms": int(frame["timestamp_ms"]),
            "source_path": str(frame["path"]),
            "artifact_url": f"/artifacts/{out_path.name}",
            "artifact_path": str(out_path),
        }

    def _select_input_frame(self, *, sample_index: int, source: str, period: str | None) -> dict[str, Any]:
        rec = self.dataset.records[sample_index]
        payload = rec.inputs.get(source)
        if payload is None:
            raise ValueError(f"sample {sample_index} has no input source: {source}")
        frames = list(payload.get("frames", []))
        if not frames:
            raise ValueError(f"sample {sample_index} source {source} has no frames")

        if period is None or str(period).strip().lower() in {"", "latest", "last"}:
            return max(frames, key=lambda frame: int(frame["timestamp_ms"]))

        target = str(period).strip().lower()
        for frame in frames:
            if Path(frame["path"]).stem.lower() == target:
                return frame
        available = ", ".join(Path(frame["path"]).stem for frame in frames)
        raise ValueError(f"period '{period}' not found for sample {sample_index} source {source}; available: {available}")

    def _build_task_outputs(
        self,
        *,
        sample_index: int,
        task: str,
        batch: dict[str, Any],
        output,
        water_threshold: float,
        rgb_source: str,
        rgb_period: str | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        metrics: dict[str, Any] = {}
        artifacts: dict[str, str] = {}
        target_shape = tuple(batch["targets"]["worldcover"].shape[-2:])

        if task in {"all", "water"}:
            water_metrics, water_artifacts = self._water_outputs(
                sample_index=sample_index,
                batch=batch,
                output=output,
                target_shape=target_shape,
                threshold=water_threshold,
                rgb_source=rgb_source,
                rgb_period=rgb_period,
            )
            metrics["water"] = water_metrics
            artifacts.update(water_artifacts)

        if task in {"all", "landcover"}:
            landcover_metrics, landcover_artifacts = self._landcover_outputs(
                sample_index=sample_index,
                batch=batch,
                output=output,
                target_shape=target_shape,
                rgb_source=rgb_source,
                rgb_period=rgb_period,
            )
            metrics["landcover"] = landcover_metrics
            artifacts.update(landcover_artifacts)

        if task in {"all", "dem"}:
            dem_metrics, dem_artifacts = self._dem_outputs(
                sample_index=sample_index,
                batch=batch,
                output=output,
                target_shape=target_shape,
                rgb_source=rgb_source,
                rgb_period=rgb_period,
            )
            metrics["dem"] = dem_metrics
            artifacts.update(dem_artifacts)

        metrics["diagnostics"] = self._diagnostics(output)
        return metrics, artifacts

    def _diagnostics(self, output) -> dict[str, Any]:
        embedding = output.embedding.detach().cpu()
        embedding_map = output.embedding_map.detach().cpu()
        return {
            "embedding_dim": int(embedding.shape[-1]),
            "embedding_norm": _finite_float(embedding.norm(dim=1)[0].item()),
            "embedding_map_shape": list(embedding_map.shape[-2:]),
            "embedding_map_mean": _finite_float(embedding_map.mean().item()),
            "embedding_map_std": _finite_float(embedding_map.std().item()),
        }

    def _dem_outputs(
        self,
        *,
        sample_index: int,
        batch: dict[str, Any],
        output,
        target_shape: tuple[int, int],
        rgb_source: str,
        rgb_period: str | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        dem_true_t = batch["targets"]["dem"][0, 0].float()
        dem_pred_t = _continuous_prediction(output.reconstructions["dem"], target_shape)[0]
        dem_true = dem_true_t.numpy()
        dem_pred = dem_pred_t.numpy()
        dem_valid = np.isfinite(dem_true)
        abs_error = np.abs(dem_pred - dem_true)

        metric_unit = "normalized_zscore"
        metric_pred = dem_pred
        metric_true = dem_true
        normalized_metrics = _regression_metrics(dem_pred, dem_true, dem_valid)
        meter_metrics = None
        if self.dem_stats:
            metric_unit = "meters"
            std = float(self.dem_stats["std"])
            mean = float(self.dem_stats["mean"])
            metric_pred = dem_pred * std + mean
            metric_true = dem_true * std + mean
            meter_metrics = _regression_metrics(metric_pred, metric_true, dem_valid)

        metrics = meter_metrics or normalized_metrics

        rgb_info = self.render_patch_rgb(
            sample_index,
            source=rgb_source,
            period=rgb_period,
            use_cache=True,
        )
        rgb = _make_source_rgb(np.load(rgb_info["source_path"]), rgb_info["source"])
        if rgb.shape[:2] != target_shape:
            rgb = _resize_numpy_image(rgb, target_shape, mode="bilinear")

        rgb_key = f"{rgb_info['source']}_{_period_slug(rgb_info['period'])}"
        target_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_target.png"
        dem_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_reconstruction.png"
        error_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_abs_error.png"
        compare_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_compare_{rgb_key}.png"
        hillshade_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_hillshade.png"
        contour_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_contours.png"
        zones_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_elevation_zones.png"
        slope_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_slope.png"
        profile_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_profile.png"
        terrain_overview_path = self.artifact_dir / f"sample_{sample_index:06d}_dem_terrain_overview_{rgb_key}.png"

        display_dem = _smooth_array(metric_pred, iterations=2)
        hillshade = _hillshade(display_dem)
        slope = _slope_per_cell(display_dem)
        elevation_zones, elevation_zone_distribution, elevation_zone_edges = _elevation_zones(display_dem)
        profile = _profile_values(display_dem, axis="east_west")

        vmin = _finite_float(np.nanpercentile(metric_true, 2))
        vmax = _finite_float(np.nanpercentile(metric_true, 98))
        self._save_raster(target_path, metric_true, cmap="terrain", vmin=vmin, vmax=vmax, colorbar=True)
        self._save_raster(dem_path, metric_pred, cmap="terrain", vmin=vmin, vmax=vmax, colorbar=True)
        self._save_raster(error_path, np.abs(metric_pred - metric_true), cmap="magma", vmin=0.0, colorbar=True)
        self._save_dem_compare(compare_path, rgb, metric_true, metric_pred, np.abs(metric_pred - metric_true), metric_unit)
        self._save_raster(hillshade_path, hillshade, cmap="gray", vmin=0.0, vmax=1.0, colorbar=False)
        self._save_dem_contours(contour_path, metric_pred, hillshade, metric_unit)
        self._save_elevation_zones(zones_path, elevation_zones, elevation_zone_edges, metric_unit)
        self._save_raster(slope_path, slope, cmap="magma", vmin=0.0, colorbar=True)
        self._save_dem_profile(profile_path, profile, metric_unit)
        self._save_dem_terrain_overview(
            terrain_overview_path,
            rgb,
            display_dem,
            hillshade,
            elevation_zones,
            elevation_zone_edges,
            slope,
            profile,
            metric_unit,
        )

        return (
            {
                "regression_method": "continuous_dem_reconstruction",
                "grid_shape": list(dem_pred.shape),
                "grid_pixels": int(dem_pred.size),
                "metric_unit": metric_unit,
                "display_mode": "terrain_analysis",
                "pred_mean": _finite_float(np.nanmean(metric_pred)),
                "pred_std": _finite_float(np.nanstd(metric_pred)),
                "pred_min": _finite_float(np.nanmin(metric_pred)),
                "pred_max": _finite_float(np.nanmax(metric_pred)),
                "target_mean": _finite_float(np.nanmean(metric_true)),
                "target_std": _finite_float(np.nanstd(metric_true)),
                "target_min": _finite_float(np.nanmin(metric_true)),
                "target_max": _finite_float(np.nanmax(metric_true)),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "r2": metrics["r2"],
                "pearson_r": metrics["pearson_r"],
                "valid_target_pixels": metrics["valid_target_pixels"],
                "evaluation_status": metrics["evaluation_status"],
                "terrain_relief": _finite_float(np.nanmax(metric_pred) - np.nanmin(metric_pred)),
                "display_dem_smoothing": "3x3_mean_filter_2_iterations",
                "slope_method": "smoothed_elevation_change_per_output_grid_cell",
                "slope_mean": _finite_float(np.nanmean(slope)),
                "slope_p95": _finite_float(np.nanpercentile(slope, 95)),
                "slope_max": _finite_float(np.nanmax(slope)),
                "elevation_zone_distribution": elevation_zone_distribution,
                "profile_axis": "east_west_centerline",
                "profile_min": _finite_float(np.nanmin(profile)),
                "profile_max": _finite_float(np.nanmax(profile)),
                "profile_relief": _finite_float(np.nanmax(profile) - np.nanmin(profile)),
                "normalized_metrics": normalized_metrics,
                "dem_stats": self.dem_stats,
            },
            {
                "original_patch_rgb_png": rgb_info["artifact_url"],
                "dem_target_png": f"/artifacts/{target_path.name}",
                "dem_reconstruction_png": f"/artifacts/{dem_path.name}",
                "dem_abs_error_png": f"/artifacts/{error_path.name}",
                "dem_compare_png": f"/artifacts/{compare_path.name}",
                "dem_hillshade_png": f"/artifacts/{hillshade_path.name}",
                "dem_contours_png": f"/artifacts/{contour_path.name}",
                "dem_elevation_zones_png": f"/artifacts/{zones_path.name}",
                "dem_slope_png": f"/artifacts/{slope_path.name}",
                "dem_profile_png": f"/artifacts/{profile_path.name}",
                "dem_terrain_overview_png": f"/artifacts/{terrain_overview_path.name}",
            },
        )

    def _landcover_outputs(
        self,
        *,
        sample_index: int,
        batch: dict[str, Any],
        output,
        target_shape: tuple[int, int],
        rgb_source: str,
        rgb_period: str | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        wc_pred, confidence = _landcover_prediction_from_logits(output.reconstructions["worldcover"], target_shape)
        wc_target = batch["targets"]["worldcover"][0, 0].detach().cpu().numpy().astype(np.int64)
        valid_target = wc_target != 255
        correct_mask = (wc_pred == wc_target) & valid_target
        overall_accuracy = float(correct_mask.sum() / max(int(valid_target.sum()), 1))

        landcover_path = self.artifact_dir / f"sample_{sample_index:06d}_landcover_classification.png"
        target_path = self.artifact_dir / f"sample_{sample_index:06d}_landcover_target.png"
        confidence_path = self.artifact_dir / f"sample_{sample_index:06d}_landcover_confidence.png"
        correctness_path = self.artifact_dir / f"sample_{sample_index:06d}_landcover_correctness.png"
        compare_path = self.artifact_dir / f"sample_{sample_index:06d}_landcover_compare.png"

        rgb_info = self.render_patch_rgb(
            sample_index,
            source=rgb_source,
            period=rgb_period,
            use_cache=True,
        )
        rgb_key = f"{rgb_info['source']}_{_period_slug(rgb_info['period'])}"
        overlay_path = self.artifact_dir / f"sample_{sample_index:06d}_landcover_overlay_{rgb_key}.png"

        self._save_rgb(landcover_path, _landcover_colorize(wc_pred))
        self._save_rgb(target_path, _landcover_colorize(wc_target))
        self._save_raster(confidence_path, confidence, cmap="viridis", vmin=0.0, vmax=1.0, colorbar=True)
        self._save_correctness(correctness_path, correct_mask, valid_target)
        self._save_landcover_compare(compare_path, wc_pred, wc_target, correct_mask, confidence)

        rgb = _make_source_rgb(np.load(rgb_info["source_path"]), rgb_info["source"])
        if rgb.shape[:2] != wc_pred.shape:
            rgb = _resize_numpy_image(rgb, tuple(wc_pred.shape), mode="bilinear")
        self._save_landcover_overlay(overlay_path, rgb, wc_pred, confidence)

        wc_distribution = _class_distribution(wc_pred)
        low_confidence_mask = confidence < 0.5
        return (
            {
                "classification_method": "softmax_argmax_worldcover_9class",
                "grid_shape": list(wc_pred.shape),
                "grid_pixels": int(wc_pred.size),
                "class_schema": "ESA WorldCover remapped by data/full_npy/preprocess_meta.json",
                "distribution": wc_distribution,
                "target_distribution": _class_distribution(wc_target[valid_target]),
                "dominant_class": wc_distribution[0] if wc_distribution else None,
                "num_predicted_classes": int(len(wc_distribution)),
                "mean_confidence": _finite_float(confidence.mean()),
                "median_confidence": _finite_float(np.median(confidence)),
                "p10_confidence": _finite_float(np.percentile(confidence, 10)),
                "low_confidence_ratio": _finite_float(low_confidence_mask.mean()),
                "overall_accuracy": _finite_float(overall_accuracy),
                "correct_pixels": int(correct_mask.sum()),
                "valid_target_pixels": int(valid_target.sum()),
                "class_accuracy": _class_accuracy_rows(wc_pred, wc_target),
                "confusion_matrix": _confusion_matrix(wc_pred, wc_target),
            },
            {
                "original_patch_rgb_png": rgb_info["artifact_url"],
                "landcover_classification_png": f"/artifacts/{landcover_path.name}",
                "landcover_target_png": f"/artifacts/{target_path.name}",
                "landcover_confidence_png": f"/artifacts/{confidence_path.name}",
                "landcover_correctness_png": f"/artifacts/{correctness_path.name}",
                "landcover_compare_png": f"/artifacts/{compare_path.name}",
                "landcover_overlay_png": f"/artifacts/{overlay_path.name}",
            },
        )

    def _water_outputs(
        self,
        *,
        sample_index: int,
        batch: dict[str, Any],
        output,
        target_shape: tuple[int, int],
        threshold: float,
        rgb_source: str,
        rgb_period: str | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        water_score, water_class, confidence = _water_score_from_logits(
            output.reconstructions["jrc_water"],
            target_shape,
        )
        water_mask = water_score >= threshold
        water_target_class = batch["targets"]["jrc_water"][0, 0].detach().cpu().numpy().astype(np.int64)
        valid_target = water_target_class != 255
        water_target_score = np.clip(water_target_class.astype(np.float32), 0.0, 100.0) / 100.0
        water_target_mask = (water_target_class > 0) & valid_target
        correct_mask = (water_mask == water_target_mask) & valid_target
        binary_metrics = _binary_metrics(water_mask, water_target_mask, valid_target)
        high_confidence_water = water_score >= 0.8
        potential_water = (water_score >= threshold) & (water_score < 0.8)

        threshold_key = _threshold_slug(threshold)
        rgb_info = self.render_patch_rgb(
            sample_index,
            source=rgb_source,
            period=rgb_period,
            use_cache=True,
        )
        rgb_key = f"{rgb_info['source']}_{_period_slug(rgb_info['period'])}"
        probability_path = self.artifact_dir / f"sample_{sample_index:06d}_water_probability.png"
        mask_path = self.artifact_dir / f"sample_{sample_index:06d}_water_mask_{threshold_key}.png"
        target_path = self.artifact_dir / f"sample_{sample_index:06d}_water_target.png"
        target_level_path = self.artifact_dir / f"sample_{sample_index:06d}_water_target_level.png"
        correctness_path = self.artifact_dir / f"sample_{sample_index:06d}_water_correctness_{threshold_key}.png"
        compare_path = self.artifact_dir / f"sample_{sample_index:06d}_water_compare_{threshold_key}.png"
        overlay_path = self.artifact_dir / f"sample_{sample_index:06d}_water_overlay_{threshold_key}_{rgb_key}.png"
        class_path = self.artifact_dir / f"sample_{sample_index:06d}_water_level_class.png"

        self._save_raster(probability_path, water_score, cmap="Blues", vmin=0.0, vmax=1.0, colorbar=True)
        self._save_water_mask(mask_path, water_mask)
        self._save_water_mask(target_path, water_target_mask)
        self._save_raster(target_level_path, water_target_score, cmap="Blues", vmin=0.0, vmax=1.0, colorbar=True)
        self._save_correctness(correctness_path, correct_mask, valid_target)
        self._save_water_compare(
            compare_path,
            water_target_mask,
            water_mask,
            correct_mask,
            valid_target,
            water_score,
            threshold,
        )
        self._save_raster(class_path, water_class, cmap="Blues", vmin=0, vmax=100, colorbar=True)

        rgb = _make_source_rgb(np.load(rgb_info["source_path"]), rgb_info["source"])
        if rgb.shape[:2] != target_shape:
            rgb = _resize_numpy_image(rgb, target_shape, mode="bilinear")
        self._save_water_overlay(overlay_path, rgb, water_score, water_mask)

        total_pixels = int(water_mask.size)
        water_pixels = int(water_mask.sum())
        non_water_pixels = int(total_pixels - water_pixels)
        return (
            {
                "threshold": float(threshold),
                "score_method": "softmax_expected_jrc_water_level",
                "score_range": [0.0, 1.0],
                "pred_water_ratio": _finite_float(water_mask.mean()),
                "pred_water_pixels": water_pixels,
                "pred_non_water_pixels": non_water_pixels,
                "target_water_ratio": _finite_float(water_target_mask[valid_target].mean()) if valid_target.any() else None,
                "target_water_pixels": int(water_target_mask.sum()),
                "target_non_water_pixels": int((valid_target & ~water_target_mask).sum()),
                "mean_water_score": _finite_float(water_score.mean()),
                "max_water_score": _finite_float(water_score.max()),
                "p95_water_score": _finite_float(np.percentile(water_score, 95)),
                "mean_confidence": _finite_float(confidence.mean()),
                "high_confidence_water_ratio": _finite_float(high_confidence_water.mean()),
                "potential_water_ratio": _finite_float(potential_water.mean()),
                **binary_metrics,
            },
            {
                "original_patch_rgb_png": rgb_info["artifact_url"],
                "water_probability_png": f"/artifacts/{probability_path.name}",
                "water_mask_png": f"/artifacts/{mask_path.name}",
                "water_target_png": f"/artifacts/{target_path.name}",
                "water_target_level_png": f"/artifacts/{target_level_path.name}",
                "water_correctness_png": f"/artifacts/{correctness_path.name}",
                "water_compare_png": f"/artifacts/{compare_path.name}",
                "water_overlay_png": f"/artifacts/{overlay_path.name}",
                "water_level_class_png": f"/artifacts/{class_path.name}",
            },
        )

    def _save_raster(
        self,
        path: Path,
        arr: np.ndarray,
        *,
        cmap: str,
        vmin: float | None = None,
        vmax: float | None = None,
        colorbar: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_dem_compare(
        self,
        path: Path,
        rgb: np.ndarray,
        target: np.ndarray,
        pred: np.ndarray,
        abs_error: np.ndarray,
        unit: str,
    ) -> None:
        valid = np.isfinite(target)
        if valid.any():
            vmin, vmax = np.nanpercentile(target[valid], [2, 98])
        else:
            vmin, vmax = None, None

        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5), constrained_layout=True)
        axes[0, 0].imshow(np.clip(rgb, 0.0, 1.0))
        axes[0, 0].set_title("Source RGB")
        axes[0, 0].set_axis_off()

        im_target = axes[0, 1].imshow(target, cmap="terrain", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Target DEM ({unit})")
        axes[0, 1].set_axis_off()
        fig.colorbar(im_target, ax=axes[0, 1], fraction=0.046, pad=0.02)

        im_pred = axes[1, 0].imshow(pred, cmap="terrain", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f"Prediction DEM ({unit})")
        axes[1, 0].set_axis_off()
        fig.colorbar(im_pred, ax=axes[1, 0], fraction=0.046, pad=0.02)

        im_error = axes[1, 1].imshow(abs_error, cmap="magma", vmin=0.0)
        axes[1, 1].set_title(f"Absolute error ({unit})")
        axes[1, 1].set_axis_off()
        fig.colorbar(im_error, ax=axes[1, 1], fraction=0.046, pad=0.02)

        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_dem_contours(self, path: Path, dem: np.ndarray, hillshade: np.ndarray, unit: str) -> None:
        valid = np.isfinite(dem)
        fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        ax.imshow(hillshade, cmap="gray", vmin=0.0, vmax=1.0)
        if valid.any() and float(np.nanmax(dem)) > float(np.nanmin(dem)):
            levels = np.linspace(float(np.nanmin(dem[valid])), float(np.nanmax(dem[valid])), 9)
            contours = ax.contour(dem, levels=levels, colors="#1f2937", linewidths=0.65, alpha=0.78)
            ax.clabel(contours, inline=True, fontsize=6, fmt="%.0f")
        ax.set_title(f"Terrain contours ({unit})")
        ax.set_axis_off()
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_elevation_zones(
        self,
        path: Path,
        zones: np.ndarray,
        edges: np.ndarray,
        unit: str,
    ) -> None:
        num_zones = max(len(edges) - 1, 1)
        cmap = plt.get_cmap("terrain", num_zones)
        display = np.ma.masked_where(zones < 0, zones)
        fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        im = ax.imshow(display, cmap=cmap, vmin=-0.5, vmax=num_zones - 0.5)
        ax.set_title(f"Elevation zones ({unit})")
        ax.set_axis_off()
        ticks = np.arange(num_zones)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, ticks=ticks)
        cbar.ax.set_yticklabels([f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(num_zones)])
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_dem_profile(self, path: Path, profile: np.ndarray, unit: str) -> None:
        x = np.arange(profile.size)
        fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
        ax.plot(x, profile, color="#2563eb", linewidth=2.0)
        ax.fill_between(x, profile, np.nanmin(profile), color="#93c5fd", alpha=0.35)
        ax.set_title(f"East-west centerline elevation profile ({unit})")
        ax.set_xlabel("Grid cell")
        ax.set_ylabel(f"Elevation ({unit})")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    def _save_dem_terrain_overview(
        self,
        path: Path,
        rgb: np.ndarray,
        dem: np.ndarray,
        hillshade: np.ndarray,
        zones: np.ndarray,
        edges: np.ndarray,
        slope: np.ndarray,
        profile: np.ndarray,
        unit: str,
    ) -> None:
        num_zones = max(len(edges) - 1, 1)
        zone_cmap = plt.get_cmap("terrain", num_zones)
        zone_display = np.ma.masked_where(zones < 0, zones)

        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), constrained_layout=True)
        axes[0, 0].imshow(np.clip(rgb, 0.0, 1.0))
        axes[0, 0].set_title("Source RGB")
        axes[0, 0].set_axis_off()

        axes[0, 1].imshow(hillshade, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title("Hillshade")
        axes[0, 1].set_axis_off()

        im_zones = axes[0, 2].imshow(zone_display, cmap=zone_cmap, vmin=-0.5, vmax=num_zones - 0.5)
        axes[0, 2].set_title("Elevation zones")
        axes[0, 2].set_axis_off()
        cbar_zones = fig.colorbar(im_zones, ax=axes[0, 2], fraction=0.046, pad=0.02, ticks=np.arange(num_zones))
        cbar_zones.ax.set_yticklabels([f"{edges[i]:.0f}-{edges[i + 1]:.0f}" for i in range(num_zones)])

        im_slope = axes[1, 0].imshow(slope, cmap="magma", vmin=0.0)
        axes[1, 0].set_title("Slope intensity")
        axes[1, 0].set_axis_off()
        fig.colorbar(im_slope, ax=axes[1, 0], fraction=0.046, pad=0.02)

        axes[1, 1].imshow(hillshade, cmap="gray", vmin=0.0, vmax=1.0)
        valid = np.isfinite(dem)
        if valid.any() and float(np.nanmax(dem[valid])) > float(np.nanmin(dem[valid])):
            axes[1, 1].contour(
                dem,
                levels=np.linspace(float(np.nanmin(dem[valid])), float(np.nanmax(dem[valid])), 8),
                colors="#111827",
                linewidths=0.7,
                alpha=0.55,
            )
        axes[1, 1].set_title("Terrain structure")
        axes[1, 1].set_axis_off()

        x = np.arange(profile.size)
        axes[1, 2].plot(x, profile, color="#2563eb", linewidth=2.0)
        axes[1, 2].fill_between(x, profile, np.nanmin(profile), color="#93c5fd", alpha=0.35)
        axes[1, 2].set_title("Elevation profile")
        axes[1, 2].set_xlabel("Grid cell")
        axes[1, 2].set_ylabel(unit)
        axes[1, 2].grid(True, color="#e5e7eb", linewidth=0.8)
        for spine in axes[1, 2].spines.values():
            spine.set_color("#d1d5db")

        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    def _save_water_mask(self, path: Path, mask: np.ndarray) -> None:
        self._save_rgb(path, self._water_mask_rgb(mask))

    def _water_mask_rgb(self, mask: np.ndarray) -> np.ndarray:
        rgb = np.ones((*mask.shape, 3), dtype=np.float32)
        rgb[mask] = np.array([0.05, 0.36, 0.75], dtype=np.float32)
        return rgb

    def _save_water_overlay(self, path: Path, rgb: np.ndarray, score: np.ndarray, mask: np.ndarray) -> None:
        base = np.clip(rgb.astype(np.float32), 0.0, 1.0)
        water_color = np.array([0.0, 0.45, 0.9], dtype=np.float32)
        alpha = np.clip(score, 0.0, 1.0) * 0.65
        alpha = np.where(mask, alpha, 0.0)[..., None]
        overlay = base * (1.0 - alpha) + water_color * alpha
        self._save_rgb(path, overlay)

    def _save_water_compare(
        self,
        path: Path,
        target_mask: np.ndarray,
        pred_mask: np.ndarray,
        correct_mask: np.ndarray,
        valid_mask: np.ndarray,
        water_score: np.ndarray,
        threshold: float,
    ) -> None:
        correctness = np.ones((*correct_mask.shape, 3), dtype=np.float32)
        correctness[valid_mask & correct_mask] = np.array([0.12, 0.62, 0.32], dtype=np.float32)
        correctness[valid_mask & ~correct_mask] = np.array([0.86, 0.18, 0.18], dtype=np.float32)
        correctness[~valid_mask] = np.array([0.72, 0.72, 0.72], dtype=np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5), constrained_layout=True)
        panels = [
            (axes[0, 0], self._water_mask_rgb(target_mask), "Target water"),
            (axes[0, 1], self._water_mask_rgb(pred_mask), f"Prediction threshold={threshold:.2f}"),
            (axes[1, 0], correctness, "Correct / Error"),
        ]
        for ax, image, title in panels:
            ax.imshow(np.clip(image, 0.0, 1.0))
            ax.set_title(title)
            ax.set_axis_off()
        im = axes[1, 1].imshow(water_score, cmap="Blues", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Water score")
        axes[1, 1].set_axis_off()
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02)
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_landcover_overlay(
        self,
        path: Path,
        rgb: np.ndarray,
        pred: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        base = np.clip(rgb.astype(np.float32), 0.0, 1.0)
        classes = _landcover_colorize(pred)
        alpha = np.clip(confidence, 0.0, 1.0)
        alpha = (0.25 + 0.45 * alpha)[..., None]
        overlay = base * (1.0 - alpha) + classes * alpha
        self._save_rgb(path, overlay)

    def _save_correctness(self, path: Path, correct_mask: np.ndarray, valid_mask: np.ndarray) -> None:
        rgb = np.ones((*correct_mask.shape, 3), dtype=np.float32)
        rgb[valid_mask & correct_mask] = np.array([0.12, 0.62, 0.32], dtype=np.float32)
        rgb[valid_mask & ~correct_mask] = np.array([0.86, 0.18, 0.18], dtype=np.float32)
        rgb[~valid_mask] = np.array([0.72, 0.72, 0.72], dtype=np.float32)
        self._save_rgb(path, rgb)

    def _save_landcover_compare(
        self,
        path: Path,
        pred: np.ndarray,
        target: np.ndarray,
        correct_mask: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        correctness = np.ones((*correct_mask.shape, 3), dtype=np.float32)
        correctness[correct_mask] = np.array([0.12, 0.62, 0.32], dtype=np.float32)
        correctness[~correct_mask] = np.array([0.86, 0.18, 0.18], dtype=np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5), constrained_layout=True)
        panels = [
            (axes[0, 0], _landcover_colorize(target), "Target"),
            (axes[0, 1], _landcover_colorize(pred), "Prediction"),
            (axes[1, 0], correctness, "Correct / Error"),
        ]
        for ax, image, title in panels:
            ax.imshow(np.clip(image, 0.0, 1.0))
            ax.set_title(title)
            ax.set_axis_off()
        im = axes[1, 1].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Top-1 confidence")
        axes[1, 1].set_axis_off()
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02)
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_rgb(self, path: Path, arr: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
        ax.imshow(np.clip(arr, 0.0, 1.0))
        ax.set_axis_off()
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def _save_image(self, path: Path, arr: np.ndarray, title: str, **imshow_kwargs) -> None:
        fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
        im = ax.imshow(arr, **imshow_kwargs)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _summarize(self, items: list[dict[str, Any]], task: str, water_threshold: float) -> dict[str, Any]:
        water_ratios = [
            item["metrics"]["water"]["pred_water_ratio"]
            for item in items
            if "water" in item["metrics"]
            if item["metrics"]["water"]["pred_water_ratio"] is not None
        ]
        water_accuracy = [
            item["metrics"]["water"]["accuracy"]
            for item in items
            if "water" in item["metrics"]
            if item["metrics"]["water"].get("accuracy") is not None
        ]
        water_f1 = [
            item["metrics"]["water"]["f1"]
            for item in items
            if "water" in item["metrics"]
            if item["metrics"]["water"].get("f1") is not None
        ]
        water_iou = [
            item["metrics"]["water"]["iou"]
            for item in items
            if "water" in item["metrics"]
            if item["metrics"]["water"].get("iou") is not None
        ]
        dem_means = [
            item["metrics"]["dem"]["pred_mean"]
            for item in items
            if "dem" in item["metrics"]
            if item["metrics"]["dem"]["pred_mean"] is not None
        ]
        dem_mae = [
            item["metrics"]["dem"]["mae"]
            for item in items
            if "dem" in item["metrics"]
            if item["metrics"]["dem"].get("mae") is not None
        ]
        dem_rmse = [
            item["metrics"]["dem"]["rmse"]
            for item in items
            if "dem" in item["metrics"]
            if item["metrics"]["dem"].get("rmse") is not None
        ]
        dem_r2 = [
            item["metrics"]["dem"]["r2"]
            for item in items
            if "dem" in item["metrics"]
            if item["metrics"]["dem"].get("r2") is not None
        ]
        dem_relief = [
            item["metrics"]["dem"]["terrain_relief"]
            for item in items
            if "dem" in item["metrics"]
            if item["metrics"]["dem"].get("terrain_relief") is not None
        ]
        dem_slope_mean = [
            item["metrics"]["dem"]["slope_mean"]
            for item in items
            if "dem" in item["metrics"]
            if item["metrics"]["dem"].get("slope_mean") is not None
        ]
        landcover_confidences = [
            item["metrics"]["landcover"]["mean_confidence"]
            for item in items
            if "landcover" in item["metrics"]
            if item["metrics"]["landcover"]["mean_confidence"] is not None
        ]
        landcover_low_confidence_ratios = [
            item["metrics"]["landcover"]["low_confidence_ratio"]
            for item in items
            if "landcover" in item["metrics"]
            if item["metrics"]["landcover"]["low_confidence_ratio"] is not None
        ]
        landcover_accuracies = [
            item["metrics"]["landcover"]["overall_accuracy"]
            for item in items
            if "landcover" in item["metrics"]
            if item["metrics"]["landcover"].get("overall_accuracy") is not None
        ]
        class_pixels: dict[int, dict[str, Any]] = {}
        for item in items:
            landcover = item["metrics"].get("landcover")
            if not landcover:
                continue
            for row in landcover["distribution"]:
                cid = int(row["class_id"])
                if cid not in class_pixels:
                    class_pixels[cid] = {
                        "class_id": cid,
                        "label": row["label"],
                        "label_zh": row.get("label_zh"),
                        "esa_worldcover_code": row.get("esa_worldcover_code"),
                        "pixels": 0,
                    }
                class_pixels[cid]["pixels"] += int(row["pixels"])
        total_pixels = sum(row["pixels"] for row in class_pixels.values()) or 1
        distribution = sorted(
            [
                {**row, "ratio": float(row["pixels"] / total_pixels)}
                for row in class_pixels.values()
            ],
            key=lambda row: row["ratio"],
            reverse=True,
        )
        summary: dict[str, Any] = {
            "task": task,
            "num_samples": len(items),
            "sample_ids": [item["sample_id"] for item in items],
        }
        if task in {"all", "water"}:
            summary["water_threshold"] = float(water_threshold)
            summary["water_ratio_mean"] = _finite_float(np.mean(water_ratios)) if water_ratios else None
            summary["water_accuracy_mean"] = _finite_float(np.mean(water_accuracy)) if water_accuracy else None
            summary["water_f1_mean"] = _finite_float(np.mean(water_f1)) if water_f1 else None
            summary["water_iou_mean"] = _finite_float(np.mean(water_iou)) if water_iou else None
        if task in {"all", "dem"}:
            summary["dem_pred_mean"] = _finite_float(np.mean(dem_means)) if dem_means else None
            summary["dem_mae_mean"] = _finite_float(np.mean(dem_mae)) if dem_mae else None
            summary["dem_rmse_mean"] = _finite_float(np.mean(dem_rmse)) if dem_rmse else None
            summary["dem_r2_mean"] = _finite_float(np.mean(dem_r2)) if dem_r2 else None
            summary["dem_terrain_relief_mean"] = _finite_float(np.mean(dem_relief)) if dem_relief else None
            summary["dem_slope_mean"] = _finite_float(np.mean(dem_slope_mean)) if dem_slope_mean else None
        if task in {"all", "landcover"}:
            summary["landcover_distribution"] = distribution
            summary["dominant_landcover"] = distribution[0] if distribution else None
            summary["landcover_mean_confidence"] = (
                _finite_float(np.mean(landcover_confidences)) if landcover_confidences else None
            )
            summary["landcover_low_confidence_ratio_mean"] = (
                _finite_float(np.mean(landcover_low_confidence_ratios))
                if landcover_low_confidence_ratios
                else None
            )
            summary["landcover_overall_accuracy_mean"] = (
                _finite_float(np.mean(landcover_accuracies)) if landcover_accuracies else None
            )
        return summary
