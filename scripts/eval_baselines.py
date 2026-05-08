from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import YajiangAEFDataset, aef_collate_fn
from src.eval.baselines import (
    evaluate_feature_set,
    evaluate_predictions,
    evaluate_random_projection,
    fit_predict_dummy,
)
from src.eval.features import extract_aef_embedding_map, extract_composite_map, load_deploy_model
from src.eval.sampling import (
    sample_paired_feature_pixels,
    split_indices,
    target_to_eval_label,
)
from src.utils.device import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AEF embeddings against simple baselines.")
    parser.add_argument("--config", type=str, default="configs/yajiang_v0_3_c.yaml")
    parser.add_argument("--manifest", type=str, default="data/full_npy/train.jsonl")
    parser.add_argument(
        "--deploy-model",
        type=str,
        default="outputs/aef_hyh_yajiang_v0_3_c/exports/aef_hyh_yajiang_v0_3_c_deploy.pt",
    )
    parser.add_argument("--target", type=str, default="worldcover")
    parser.add_argument("--output", type=str, default="outputs/eval_v0_3_c/report.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-patches", type=int, default=256)
    parser.add_argument("--max-pixels-per-patch", type=int, default=128)
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--random-dim", type=int, default=128)
    return parser.parse_args()


def collect_features(
    dataset: YajiangAEFDataset,
    cfg,
    model,
    device: torch.device,
    target: str,
    batch_size: int,
    max_pixels_per_patch: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=aef_collate_fn,
    )

    rng = np.random.default_rng(seed)
    feature_parts: dict[str, list[np.ndarray]] = {"aef": [], "composite": []}
    label_parts: list[np.ndarray] = []
    task_type: str | None = None

    for batch in loader:
        labels, valid_mask, cur_task_type = target_to_eval_label(batch["targets"], target)
        if task_type is None:
            task_type = cur_task_type
        elif task_type != cur_task_type:
            raise RuntimeError("Mixed task types are not supported")

        aef_map = extract_aef_embedding_map(model, batch, device)
        composite_map = extract_composite_map(batch, cfg)

        x_aef, x_comp, y = sample_paired_feature_pixels(
            aef_map,
            composite_map,
            labels,
            valid_mask,
            max_pixels_per_patch=max_pixels_per_patch,
            rng=rng,
        )

        if len(y) == 0:
            continue

        feature_parts["aef"].append(x_aef)
        feature_parts["composite"].append(x_comp)
        label_parts.append(y)

    if not label_parts or task_type is None:
        raise RuntimeError(f"No valid evaluation pixels found for target: {target}")

    features = {k: np.concatenate(v, axis=0) for k, v in feature_parts.items()}
    labels = np.concatenate(label_parts, axis=0)
    return features, labels, task_type


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    cfg = load_config(args.config)
    model, deploy_cfg = load_deploy_model(args.deploy_model, device=device)

    dataset = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    if args.max_patches > 0:
        dataset.records = dataset.records[: args.max_patches]

    train_idx, test_idx = split_indices(len(dataset), train_fraction=args.train_fraction, seed=args.seed)

    train_ds = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    test_ds = YajiangAEFDataset(cfg=cfg, manifest_path=args.manifest, split="train")
    records = dataset.records
    train_ds.records = [records[i] for i in train_idx]
    test_ds.records = [records[i] for i in test_idx]

    x_train, y_train, task_type = collect_features(
        train_ds,
        cfg,
        model,
        device,
        target=args.target,
        batch_size=args.batch_size,
        max_pixels_per_patch=args.max_pixels_per_patch,
        seed=args.seed,
    )
    x_test, y_test, test_task_type = collect_features(
        test_ds,
        cfg,
        model,
        device,
        target=args.target,
        batch_size=args.batch_size,
        max_pixels_per_patch=args.max_pixels_per_patch,
        seed=args.seed + 1,
    )
    if task_type != test_task_type:
        raise RuntimeError("Train/test task type mismatch")

    results = {
        "dummy": evaluate_predictions(
            y_test,
            fit_predict_dummy(x_train["composite"], y_train, x_test["composite"], task_type, seed=args.seed),
            task_type,
        )
    }
    results.update(
        evaluate_feature_set(
            "composite",
            x_train["composite"],
            y_train,
            x_test["composite"],
            y_test,
            task_type=task_type,
            seed=args.seed,
            knn_k=args.knn_k,
        )
    )
    results.update(
        evaluate_feature_set(
            "aef",
            x_train["aef"],
            y_train,
            x_test["aef"],
            y_test,
            task_type=task_type,
            seed=args.seed,
            knn_k=args.knn_k,
        )
    )
    results["random_projection_linear"] = evaluate_random_projection(
        x_train["composite"],
        y_train,
        x_test["composite"],
        y_test,
        task_type=task_type,
        seed=args.seed,
        out_dim=args.random_dim,
    )

    report = {
        "target": args.target,
        "task_type": task_type,
        "deploy_model": str(Path(args.deploy_model).resolve()),
        "deploy_format": getattr(deploy_cfg, "format", None),
        "num_patches": len(dataset),
        "num_train_patches": len(train_idx),
        "num_test_patches": len(test_idx),
        "num_train_pixels": int(len(y_train)),
        "num_test_pixels": int(len(y_test)),
        "max_pixels_per_patch": args.max_pixels_per_patch,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report to: {output}")


if __name__ == "__main__":
    main()
