from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_iou": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels = sorted(set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist()))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": [int(x) if float(x).is_integer() else float(x) for x in labels],
        "matrix": matrix.astype(int).tolist(),
    }


def binary_boundary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Lightweight boundary F1 for binary masks. It uses 4-neighbor edge pixels and
    reports F1 on boundary membership without distance tolerance.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        return float("nan")

    def edge(mask: np.ndarray) -> np.ndarray:
        e = np.zeros_like(mask, dtype=bool)
        e[1:, :] |= mask[1:, :] != mask[:-1, :]
        e[:-1, :] |= mask[1:, :] != mask[:-1, :]
        e[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        e[:, :-1] |= mask[:, 1:] != mask[:, :-1]
        return e

    true_edge = edge(y_true)
    pred_edge = edge(y_pred)
    tp = np.logical_and(true_edge, pred_edge).sum()
    fp = np.logical_and(~true_edge, pred_edge).sum()
    fn = np.logical_and(true_edge, ~pred_edge).sum()
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)
