"""Treino de modelos com tracking MLflow padronizado."""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
from collections.abc import Callable
from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RISK_LEVELS = {"low", "medium", "high", "critical"}
MODEL_TYPES = {"classification", "regression", "bandit", "generation"}


def _get_git_sha() -> str:
    """SHA curto do HEAD ou 'unknown' se git não disponível."""
    try:
        return (
            subprocess.check_output(  # nosec B603 B607
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_standard_tags(
    model_name: str,
    model_type: str,
    framework: str,
    owner: str,
    risk_level: str,
    fairness_checked: bool,
    training_data_version: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Monta o conjunto mínimo de tags MLflow exigido para qualquer modelo do grupo."""
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"model_type deve estar em {MODEL_TYPES}, recebido: {model_type!r}"
        )
    if risk_level not in RISK_LEVELS:
        raise ValueError(
            f"risk_level deve estar em {RISK_LEVELS}, recebido: {risk_level!r}"
        )

    tags: dict[str, str] = {
        "model_name": model_name,
        "model_type": model_type,
        "framework": framework,
        "owner": owner,
        "risk_level": risk_level,
        "fairness_checked": str(fairness_checked).lower(),
        "training_data_version": training_data_version,
        "git_sha": _get_git_sha(),
        "phase": "datathon-fase05",
        "group": "grupo-75",
    }
    if extra:
        tags.update(extra)
    return tags


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series | None = None,
) -> dict[str, float]:
    """Métricas de classificação binária: auc, precision, recall, f1."""
    score_for_auc = y_proba if y_proba is not None else y_pred
    return {
        "auc": float(roc_auc_score(y_true, score_for_auc)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_and_log(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    model_factory: Callable[[], Any],
    model_params: dict[str, Any],
    *,
    owner: str,
    framework: str = "sklearn",
    model_type: str = "classification",
    risk_level: str = "medium",
    fairness_checked: bool = False,
    training_data_version: str = "kaggle:telemarketing-jyb-dataset@v1",
    test_size: float = 0.2,
    random_state: int = 42,
    experiment_name: str = "datathon-fase05",
) -> str:
    """Treina, avalia e registra um modelo no MLflow. Retorna o run_id."""
    if target_col not in df.columns:
        raise KeyError(f"target_col {target_col!r} não está no DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])

        tags = _build_standard_tags(
            model_name=model_name,
            model_type=model_type,
            framework=framework,
            owner=owner,
            risk_level=risk_level,
            fairness_checked=fairness_checked,
            training_data_version=training_data_version,
        )
        mlflow.set_tags(tags)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        if framework == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(
            "Run %s registrado: model=%s framework=%s AUC=%.4f F1=%.4f",
            run.info.run_id,
            model_name,
            framework,
            metrics["auc"],
            metrics["f1"],
        )
        return run.info.run_id


def setup_mlflow_from_env() -> None:
    """Aplica MLFLOW_TRACKING_URI do ambiente, se presente."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
