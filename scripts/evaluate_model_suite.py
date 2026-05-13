from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "32")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from demo_visualize import plot_demo
from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.eval.baselines import (
    evaluate_feature_set,
    evaluate_predictions,
    fit_predict_dummy,
)
from src.eval.features import (
    batch_to_device,
    extract_aef_embedding_map,
    extract_composite_map,
    load_deploy_model,
)
from src.eval.metrics import (
    binary_boundary_f1,
    classification_metrics,
    confusion_matrix_dict,
    regression_metrics,
)
from src.eval.sampling import sample_paired_feature_pixels, target_to_eval_label
from src.utils.device import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the core evaluation suite for a trained Yajiang AEF model. "
            "This does not create a pretraining validation/test split; it evaluates "
            "the supplied model on an evaluation manifest and reports reconstruction "
            "quality plus downstream probe quality."
        )
    )
    parser.add_argument("--config", type=str, default="configs/yajiang_v0_3_c.yaml")
    parser.add_argument("--manifest", type=str, default="data/full_npy/train.jsonl")
    parser.add_argument(
        "--deploy-model",
        type=str,
        default="outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/model_eval/v0_3_c")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-patches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pixels-per-patch", type=int, default=128)
    parser.add_argument("--few-shot", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--knn-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--demo-indices", type=int, nargs="+", default=[4, 1425])
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def resize_like_target(pred: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(pred, size=size, mode=mode)
    return F.interpolate(pred, size=size, mode=mode, align_corners=False)


def batch_reconstruction_metrics(output, batch: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    dem_true = batch["targets"]["dem"][:, 0].cpu()
    dem_pred = resize_like_target(output.reconstructions["dem"].detach().cpu(), dem_true.shape[-2:], "bilinear")[:, 0]
    valid = torch.isfinite(dem_true)
    if valid.any():
        metrics["dem"] = {
            "y_true": dem_true[valid].numpy(),
            "y_pred": dem_pred[valid].numpy(),
        }

    wc_true = batch["targets"]["worldcover"][:, 0].long().cpu()
    wc_logits = resize_like_target(output.reconstructions["worldcover"].detach().cpu(), wc_true.shape[-2:], "nearest")
    wc_pred = wc_logits.argmax(dim=1).long()
    wc_valid = wc_true != 255
    if wc_valid.any():
        metrics["worldcover"] = {
            "y_true": wc_true[wc_valid].numpy(),
            "y_pred": wc_pred[wc_valid].numpy(),
        }

    jrc_true = batch["targets"]["jrc_water"][:, 0].long().cpu()
    jrc_logits = resize_like_target(output.reconstructions["jrc_water"].detach().cpu(), jrc_true.shape[-2:], "nearest")
    jrc_pred = jrc_logits.argmax(dim=1).long()
    jrc_valid = jrc_true != 255
    if jrc_valid.any():
        metrics["jrc_water"] = {
            "y_true": jrc_true[jrc_valid].numpy(),
            "y_pred": jrc_pred[jrc_valid].numpy(),
        }
        metrics["jrc_binary"] = {
            "y_true": (jrc_true[jrc_valid] > 0).long().numpy(),
            "y_pred": (jrc_pred[jrc_valid] > 0).long().numpy(),
        }

    boundary_scores = []
    for i in range(jrc_true.shape[0]):
        mask = jrc_true[i] != 255
        if mask.any():
            true_img = ((jrc_true[i] > 0) & mask).numpy()
            pred_img = ((jrc_pred[i] > 0) & mask).numpy()
            boundary_scores.append(binary_boundary_f1(true_img, pred_img))
    if boundary_scores:
        metrics["jrc_boundary_f1"] = np.asarray(boundary_scores, dtype=np.float32)

    return metrics


def merge_prediction_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ["dem", "worldcover", "jrc_water", "jrc_binary"]:
        y_true = [p[key]["y_true"] for p in parts if key in p]
        y_pred = [p[key]["y_pred"] for p in parts if key in p]
        if y_true:
            merged[key] = {
                "y_true": np.concatenate(y_true, axis=0),
                "y_pred": np.concatenate(y_pred, axis=0),
            }
    boundary = [p["jrc_boundary_f1"] for p in parts if "jrc_boundary_f1" in p]
    if boundary:
        merged["jrc_boundary_f1"] = np.concatenate(boundary, axis=0)
    return merged


def summarize_reconstruction(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged = merge_prediction_parts(parts)
    summary: dict[str, Any] = {}
    if "dem" in merged:
        summary["dem"] = regression_metrics(merged["dem"]["y_true"], merged["dem"]["y_pred"])
    for key in ["worldcover", "jrc_water", "jrc_binary"]:
        if key in merged:
            summary[key] = classification_metrics(merged[key]["y_true"], merged[key]["y_pred"])
            summary[key]["confusion"] = confusion_matrix_dict(merged[key]["y_true"], merged[key]["y_pred"])
    if "jrc_boundary_f1" in merged:
        vals = merged["jrc_boundary_f1"]
        summary.setdefault("jrc_binary", {})["boundary_f1"] = float(np.nanmean(vals))
    return summary


def embedding_diagnostics(embeddings: np.ndarray) -> dict[str, float]:
    if embeddings.size == 0:
        return {}
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0)
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape[0] > 1:
        corr = np.corrcoef(centered, rowvar=False)
        off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_abs_corr = float(np.nanmean(np.abs(off_diag)))
    else:
        mean_abs_corr = float("nan")
    return {
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "mean_channel_std": float(std.mean()),
        "min_channel_std": float(std.min()),
        "mean_abs_channel_corr": mean_abs_corr,
    }


def collect_eval_data(
    dataset: YajiangAEFDataset,
    cfg: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_pixels_per_patch: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=aef_collate_fn,
    )
    rng = np.random.default_rng(seed)
    recon_parts: list[dict[str, Any]] = []
    fewshot: dict[str, dict[str, list[np.ndarray]]] = {
        "worldcover": {"aef": [], "composite": [], "y": []},
        "jrc_binary": {"aef": [], "composite": [], "y": []},
        "dem": {"aef": [], "composite": [], "y": []},
    }
    sample_ids: list[str] = []

    for batch in loader:
        sample_ids.extend(batch["sample_id"])
        with torch.no_grad():
            device_batch = batch_to_device(batch, device)
            output = model(
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

        recon_parts.append(batch_reconstruction_metrics(output, batch))
        aef_map = output.embedding_map.detach().cpu()
        composite_map = extract_composite_map(batch, cfg)
        for target in fewshot.keys():
            labels, valid_mask, _ = target_to_eval_label(batch["targets"], target)
            x_aef, x_comp, y = sample_paired_feature_pixels(
                aef_map,
                composite_map,
                labels,
                valid_mask,
                max_pixels_per_patch=max_pixels_per_patch,
                rng=rng,
            )
            if len(y) > 0:
                fewshot[target]["aef"].append(x_aef)
                fewshot[target]["composite"].append(x_comp)
                fewshot[target]["y"].append(y)

    reconstruction = summarize_reconstruction(recon_parts)
    embeddings = np.empty((0, 0))
    emb_summary = {}

    feature_sets: dict[str, dict[str, np.ndarray]] = {}
    for target, parts in fewshot.items():
        if parts["y"]:
            feature_sets[target] = {
                "aef": np.concatenate(parts["aef"], axis=0),
                "composite": np.concatenate(parts["composite"], axis=0),
                "y": np.concatenate(parts["y"], axis=0),
            }

    return {"reconstruction": reconstruction, "embedding": emb_summary}, feature_sets, embeddings, sample_ids


def fewshot_split(
    y: np.ndarray,
    n: int,
    task_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if task_type == "classification":
        train_parts = []
        for cls in np.unique(y):
            cls_idx = indices[y == cls]
            if len(cls_idx) == 0:
                continue
            take = min(n, len(cls_idx))
            train_parts.append(rng.choice(cls_idx, size=take, replace=False))
        train_idx = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=int)
    else:
        take = min(max(n, 8), max(1, len(indices) // 2))
        train_idx = rng.choice(indices, size=take, replace=False)
    train_mask = np.zeros(len(indices), dtype=bool)
    train_mask[train_idx] = True
    return indices[train_mask], indices[~train_mask]


def run_fewshot(
    feature_sets: dict[str, dict[str, np.ndarray]],
    shots: list[int],
    knn_k: list[int],
    seed: int,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for target, data in feature_sets.items():
        x_aef = data["aef"]
        x_comp = data["composite"]
        y = data["y"]
        task_type = "regression" if target == "dem" else "classification"
        reports[target] = {"task_type": task_type, "shots": {}}
        for n in shots:
            rng = np.random.default_rng(seed + n)
            train_idx, test_idx = fewshot_split(y, n=n, task_type=task_type, rng=rng)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            cur: dict[str, Any] = {}
            cur["dummy"] = evaluate_predictions(
                y[test_idx],
                fit_predict_dummy(x_comp[train_idx], y[train_idx], x_comp[test_idx], task_type, seed=seed),
                task_type,
            )
            cur.update(
                evaluate_feature_set(
                    "composite",
                    x_comp[train_idx],
                    y[train_idx],
                    x_comp[test_idx],
                    y[test_idx],
                    task_type=task_type,
                    seed=seed,
                    knn_k=knn_k,
                )
            )
            cur.update(
                evaluate_feature_set(
                    "aef",
                    x_aef[train_idx],
                    y[train_idx],
                    x_aef[test_idx],
                    y[test_idx],
                    task_type=task_type,
                    seed=seed,
                    knn_k=knn_k,
                )
            )
            reports[target]["shots"][str(n)] = {
                "num_train": int(len(train_idx)),
                "num_eval": int(len(test_idx)),
                "results": cur,
            }
    return reports


def plot_confusion(cm: dict[str, Any], title: str, out_path: Path) -> None:
    labels = cm["labels"]
    matrix = np.asarray(cm["matrix"], dtype=np.float32)
    row_sum = matrix.sum(axis=1, keepdims=True).clip(min=1.0)
    norm = matrix / row_sum
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_fewshot(reports: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, max(1, len(reports)), figsize=(5 * max(1, len(reports)), 4), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, (target, report) in zip(axes, reports.items()):
        task_type = report["task_type"]
        metric = "r2" if task_type == "regression" else "macro_f1"
        xs = []
        aef = []
        comp = []
        for shot, payload in sorted(report["shots"].items(), key=lambda x: int(x[0])):
            results = payload["results"]
            if "aef_linear" not in results or "composite_linear" not in results:
                continue
            xs.append(int(shot))
            aef.append(results["aef_linear"].get(metric, np.nan))
            comp.append(results["composite_linear"].get(metric, np.nan))
        ax.plot(xs, comp, marker="o", label="Composite linear")
        ax.plot(xs, aef, marker="o", label="AEF linear")
        ax.set_xscale("log")
        ax.set_title(f"{target} few-shot")
        ax.set_xlabel("shots")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_embedding_pca(embeddings: np.ndarray, sample_ids: list[str], out_path: Path) -> None:
    if embeddings.shape[0] < 2:
        return
    n = min(len(embeddings), 1000)
    x = embeddings[:n]
    coords = PCA(n_components=2, random_state=42).fit_transform(x)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.8)
    ax.set_title("Patch embedding PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    for idx in range(min(6, n)):
        ax.annotate(sample_ids[idx], (coords[idx, 0], coords[idx, 1]), fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def generate_demo_panels(
    cfg: Any,
    manifest: str,
    model: torch.nn.Module,
    device: torch.device,
    demo_indices: list[int],
    out_dir: Path,
) -> list[str]:
    paths: list[str] = []
    for idx in demo_indices:
        ds = YajiangAEFDataset(cfg=cfg, manifest_path=manifest, split="train")
        if idx < 0 or idx >= len(ds):
            continue
        ds.records = [ds.records[idx]]
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=aef_collate_fn)
        batch = next(iter(loader))
        with torch.no_grad():
            device_batch = batch_to_device(batch, device)
            output = model(
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
        out_path = out_dir / f"patch_{idx:06d}_panel.png"
        plot_demo(batch, output, out_path)
        paths.append(str(out_path))
    return paths


def pick_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def build_core_report(raw: dict[str, Any]) -> dict[str, Any]:
    recon = raw["diagnostics"]["reconstruction"]

    reconstruction = {
        "dem": {
            "mae": pick_metric(recon.get("dem", {}), "mae"),
            "r2": pick_metric(recon.get("dem", {}), "r2"),
            "meaning": "DEM continuous reconstruction; lower MAE and higher R2 are better.",
        },
        "worldcover": {
            "macro_f1": pick_metric(recon.get("worldcover", {}), "macro_f1"),
            "macro_iou": pick_metric(recon.get("worldcover", {}), "macro_iou"),
            "meaning": "WorldCover categorical reconstruction; macro metrics reduce majority-class bias.",
        },
        "water": {
            "binary_f1": pick_metric(recon.get("jrc_binary", {}), "macro_f1"),
            "binary_iou": pick_metric(recon.get("jrc_binary", {}), "macro_iou"),
            "boundary_f1": pick_metric(recon.get("jrc_binary", {}), "boundary_f1"),
            "meaning": "JRC Water converted to water/non-water; boundary F1 reflects contour quality.",
        },
    }

    downstream: dict[str, Any] = {}
    for target, payload in raw["fewshot"].items():
        task_type = payload["task_type"]
        metric = "r2" if task_type == "regression" else "macro_f1"
        target_rows = []
        for shot, shot_payload in sorted(payload["shots"].items(), key=lambda x: int(x[0])):
            results = shot_payload["results"]
            composite = results.get("composite_linear", {}).get(metric)
            aef = results.get("aef_linear", {}).get(metric)
            if composite is None or aef is None:
                continue
            target_rows.append(
                {
                    "shots": int(shot),
                    "metric": metric,
                    "composite_linear": float(composite),
                    "aef_linear": float(aef),
                    "delta": float(aef - composite),
                }
            )
        downstream[target] = {
            "task_type": task_type,
            "primary_metric": metric,
            "rows": target_rows,
            "meaning": "AEF is useful when aef_linear is higher than composite_linear under the same shot setting.",
        }

    return {
        "protocol": raw["protocol"],
        "scope": "core_report_reconstruction_and_downstream_probe",
        "config": raw["config"],
        "manifest": raw["manifest"],
        "deploy_model": raw["deploy_model"],
        "num_patches": raw["num_patches"],
        "max_pixels_per_patch": raw["max_pixels_per_patch"],
        "reconstruction": reconstruction,
        "downstream": downstream,
        "artifacts": raw["artifacts"],
    }


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Yajiang AEF 模型测评报告",
        "",
        "## 评测体系",
        "",
        "当前报告只保留两类核心指标：模型重建能力、下游任务能力。预训练阶段不强行划分验证集/测试集；这里是在给定 manifest 上对已导出的模型做统一诊断和版本对比。",
        "",
        f"- deploy model: `{report['deploy_model']}`",
        f"- manifest: `{report['manifest']}`",
        f"- patches evaluated: `{report['num_patches']}`",
        f"- max pixels per patch for probes: `{report['max_pixels_per_patch']}`",
        "",
        "## 1. 重建能力",
        "",
        "| 目标 | 核心指标 | 结果 | 说明 |",
        "| --- | --- | ---: | --- |",
    ]
    dem = report["reconstruction"]["dem"]
    wc = report["reconstruction"]["worldcover"]
    water = report["reconstruction"]["water"]
    lines.append(f"| DEM | MAE / R2 | `{dem['mae']:.4f}` / `{dem['r2']:.4f}` | 高程连续值重建 |")
    lines.append(f"| WorldCover | macro F1 / macro IoU | `{wc['macro_f1']:.4f}` / `{wc['macro_iou']:.4f}` | 地表覆盖分类重建 |")
    lines.append(f"| JRC Water | F1 / IoU / boundary F1 | `{water['binary_f1']:.4f}` / `{water['binary_iou']:.4f}` / `{water['boundary_f1']:.4f}` | 水体二分类与轮廓 |")

    lines += [
        "",
        "## 2. 下游任务能力",
        "",
        "下游能力比较的是传统 S2/S1 composite 特征和 AEF embedding。两者使用同样的少量标签、同样的 linear probe。AEF 高于 composite，说明预训练 embedding 对下游任务有增益。",
        "",
    ]
    for target, payload in report["downstream"].items():
        metric = payload["primary_metric"]
        lines.append(f"### {target} ({metric})")
        lines.append("| shots | composite linear | AEF linear | AEF - composite |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for row in payload["rows"]:
            lines.append(
                f"| {row['shots']} | {row['composite_linear']:.4f} | "
                f"{row['aef_linear']:.4f} | {row['delta']:+.4f} |"
            )
        lines.append("")

    lines += [
        "## 3. 展示文件",
        "",
    ]
    for name, path in report["artifacts"].items():
        if isinstance(path, list):
            for item in path:
                rel = Path(item).relative_to(out_dir)
                lines.append(f"- {name}: [{rel}]({rel})")
        else:
            rel = Path(path).relative_to(out_dir)
            lines.append(f"- {name}: [{rel}]({rel})")
    lines.append("")

    md = "\n".join(lines)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    html = "<html><head><meta charset='utf-8'><title>Yajiang AEF Eval</title></head><body>"
    html += "<pre style='white-space: pre-wrap; font-family: sans-serif'>" + md.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
    for path in report["artifacts"].values():
        paths = path if isinstance(path, list) else [path]
        for item in paths:
            rel = Path(item).relative_to(out_dir)
            html += f"<h3>{rel}</h3><img src='{rel}' style='max-width: 1100px; width: 100%; border: 1px solid #ddd'>"
    html += "</body></html>"
    (out_dir / "report.html").write_text(html, encoding="utf-8")


def compact_console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": report["protocol"],
        "num_patches": report["num_patches"],
        "reconstruction": report["reconstruction"],
        "downstream": report["downstream"],
        "artifacts": report["artifacts"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    if args.clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    cfg = load_config(args.config)
    model, deploy_cfg = load_deploy_model(args.deploy_model, device=device)
    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    if args.max_patches > 0:
        dataset.records = dataset.records[: args.max_patches]

    diagnostics, feature_sets, embeddings, sample_ids = collect_eval_data(
        dataset=dataset,
        cfg=cfg,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_pixels_per_patch=args.max_pixels_per_patch,
        seed=args.seed,
    )
    fewshot = run_fewshot(feature_sets, shots=args.few_shot, knn_k=args.knn_k, seed=args.seed)

    artifacts: dict[str, Any] = {}

    fewshot_path = out_dir / "fewshot_curves.png"
    plot_fewshot(fewshot, fewshot_path)
    artifacts["downstream_curves"] = str(fewshot_path)

    demo_paths = generate_demo_panels(
        cfg=cfg,
        manifest=args.manifest,
        model=model,
        device=device,
        demo_indices=args.demo_indices,
        out_dir=out_dir / "demo_panels",
    )
    artifacts["demo_panels"] = demo_paths

    raw_report = {
        "protocol": "diagnostic_on_supplied_manifest_no_pretraining_split",
        "config": str(Path(args.config).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "deploy_model": str(Path(args.deploy_model).resolve()),
        "deploy_format": getattr(deploy_cfg, "format", None),
        "num_patches": len(dataset),
        "batch_size": args.batch_size,
        "max_pixels_per_patch": args.max_pixels_per_patch,
        "diagnostics": diagnostics,
        "fewshot": fewshot,
        "artifacts": artifacts,
    }
    report = build_core_report(raw_report)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(to_jsonable(report), out_dir)

    print(json.dumps(to_jsonable(compact_console_summary(report)), ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved report to: {out_dir / 'report.md'}")
    print(f"Open HTML report: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
