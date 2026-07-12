"""Integration test: train_and_log com modelo sklearn real."""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.train import train_and_log


@pytest.fixture
def isolated_mlflow_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isola tracking MLflow em diretório temporário do pytest."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file:{tmp_path}")
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.integration
def test_train_and_log_persists_run_with_metrics(
    sample_classification_df, isolated_mlflow_dir
):
    run_id = train_and_log(
        df=sample_classification_df,
        target_col="target",
        model_name="logreg-smoke",
        model_factory=lambda: LogisticRegression(max_iter=200),
        model_params={"max_iter": 200},
        owner="sheylasilvana18@gmail.com",
        framework="sklearn",
        risk_level="low",
    )
    assert run_id

    run = mlflow.get_run(run_id)
    assert set(run.data.metrics) >= {"auc", "precision", "recall", "f1"}
    assert run.data.tags["model_name"] == "logreg-smoke"
    assert run.data.tags["phase"] == "datathon-fase05"
    assert run.data.tags["framework"] == "sklearn"
