from __future__ import annotations

import pandas as pd
import pytest

from src.models.train import (
    MODEL_TYPES,
    RISK_LEVELS,
    _build_standard_tags,
    compute_classification_metrics,
)


class TestComputeClassificationMetrics:
    def test_returns_all_four_metrics(self):
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = pd.Series([0, 1, 1, 0, 0, 0, 1, 1])
        result = compute_classification_metrics(y_true, y_pred)
        assert set(result.keys()) == {"auc", "precision", "recall", "f1"}

    def test_values_are_floats_in_unit_range(self):
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
        result = compute_classification_metrics(y_true, y_pred)
        assert all(isinstance(v, float) for v in result.values())
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_perfect_prediction_yields_unit_metrics(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 1, 0])
        result = compute_classification_metrics(y_true, y_pred)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_uses_proba_for_auc_when_provided(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])
        y_proba = pd.Series([0.1, 0.9, 0.6, 0.2])
        result = compute_classification_metrics(y_true, y_pred, y_proba)
        assert result["auc"] == 1.0


class TestBuildStandardTags:
    def _valid_kwargs(self, **overrides):
        defaults = dict(
            model_name="baseline",
            model_type="classification",
            framework="sklearn",
            owner="sheylasilvana18@gmail.com",
            risk_level="low",
            fairness_checked=False,
            training_data_version="kaggle:telemarketing-jyb-dataset@v1",
        )
        defaults.update(overrides)
        return defaults

    def test_returns_required_keys(self):
        tags = _build_standard_tags(**self._valid_kwargs())
        required = {
            "model_name",
            "model_type",
            "framework",
            "owner",
            "risk_level",
            "fairness_checked",
            "training_data_version",
            "git_sha",
            "phase",
            "group",
        }
        assert required.issubset(tags.keys())

    def test_phase_and_group_are_pinned(self):
        tags = _build_standard_tags(**self._valid_kwargs())
        assert tags["phase"] == "datathon-fase05"
        assert tags["group"] == "grupo-75"

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            _build_standard_tags(**self._valid_kwargs(model_type="invalid"))

    def test_invalid_risk_level_raises(self):
        with pytest.raises(ValueError, match="risk_level"):
            _build_standard_tags(**self._valid_kwargs(risk_level="extreme"))

    def test_fairness_checked_is_serialized_as_string(self):
        tags = _build_standard_tags(**self._valid_kwargs(fairness_checked=True))
        assert tags["fairness_checked"] == "true"

    def test_extra_tags_are_merged(self):
        tags = _build_standard_tags(
            **self._valid_kwargs(),
            extra={"experiment_branch": "feature/x"},
        )
        assert tags["experiment_branch"] == "feature/x"


def test_model_type_and_risk_constants_are_non_empty():
    assert len(MODEL_TYPES) >= 3
    assert len(RISK_LEVELS) == 4
