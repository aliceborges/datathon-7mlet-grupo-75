from __future__ import annotations


def test_import_train_module():
    from src.models import train  # noqa: F401


def test_import_drift_module():
    from src.monitoring import drift  # noqa: F401


def test_train_module_exposes_public_api():
    from src.models import train

    assert callable(train.train_and_log)
    assert callable(train.compute_classification_metrics)
    assert callable(train.setup_mlflow_from_env)


def test_drift_module_exposes_public_api():
    from src.monitoring import drift

    assert callable(drift.compute_psi)
    assert hasattr(drift, "DriftReport")
    assert hasattr(drift, "DriftAssessment")
    assert drift.PSI_WARNING < drift.PSI_RETRAIN_TRIGGER
