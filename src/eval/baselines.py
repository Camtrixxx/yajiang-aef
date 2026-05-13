from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.features import random_project_features
from src.eval.metrics import classification_metrics, regression_metrics


def fit_predict_linear(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task_type: str,
    seed: int,
) -> np.ndarray:
    if task_type == "classification" and np.unique(y_train).size < 2:
        return fit_predict_dummy(x_train, y_train, x_test, task_type=task_type, seed=seed)

    if task_type == "classification":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        )
    else:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))

    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_predict_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task_type: str,
    k: int,
) -> np.ndarray:
    if task_type == "classification":
        model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    else:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")

    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_predict_dummy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task_type: str,
    seed: int,
) -> np.ndarray:
    if task_type == "classification":
        model = DummyClassifier(strategy="most_frequent", random_state=seed)
    else:
        model = DummyRegressor(strategy="mean")
    model.fit(x_train, y_train)
    return model.predict(x_test)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true, y_pred)
    return regression_metrics(y_true, y_pred)


def evaluate_feature_set(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    seed: int,
    knn_k: list[int],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}

    pred = fit_predict_linear(x_train, y_train, x_test, task_type=task_type, seed=seed)
    results[f"{name}_linear"] = evaluate_predictions(y_test, pred, task_type)

    for k in knn_k:
        if len(y_train) < k:
            continue
        pred = fit_predict_knn(x_train, y_train, x_test, task_type=task_type, k=k)
        results[f"{name}_knn_k{k}"] = evaluate_predictions(y_test, pred, task_type)

    return results


def evaluate_random_projection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    seed: int,
    out_dim: int,
) -> dict[str, float]:
    train_proj = random_project_features(x_train, out_dim=out_dim, seed=seed)
    test_proj = random_project_features(x_test, out_dim=out_dim, seed=seed)
    pred = fit_predict_linear(train_proj, y_train, test_proj, task_type=task_type, seed=seed)
    return evaluate_predictions(y_test, pred, task_type)
