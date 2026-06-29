from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    PSI_RETRAIN_TRIGGER,
    PSI_WARNING,
    DriftAssessment,
    DriftReport,
    _classify,
    compute_psi,
)


class TestClassify:
    def test_stable_below_warning(self):
        assert _classify(0.05) == "stable"

    def test_warning_between_thresholds(self):
        assert _classify(PSI_WARNING) == "warning"
        assert _classify(0.15) == "warning"

    def test_drift_above_trigger(self):
        assert _classify(PSI_RETRAIN_TRIGGER) == "drift"
        assert _classify(0.5) == "drift"

    def test_nan_is_warning(self):
        assert _classify(float("nan")) == "warning"


class TestComputePsi:
    def test_returns_drift_report(self, reference_and_current_no_drift):
        ref, cur = reference_and_current_no_drift
        report = compute_psi(ref, cur)
        assert isinstance(report, DriftReport)
        assert len(report.assessments) == 2

    def test_same_distribution_is_stable(self, reference_and_current_no_drift):
        ref, cur = reference_and_current_no_drift
        report = compute_psi(ref, cur)
        assert report.share_of_drifted_columns == 0.0
        for assessment in report.assessments:
            assert assessment.status == "stable"

    def test_shifted_distribution_is_drift(self, reference_and_current_with_drift):
        ref, cur = reference_and_current_with_drift
        report = compute_psi(ref, cur)
        assert "x1" in report.drifted_columns
        assert "x2" not in report.drifted_columns

    def test_filters_to_requested_columns(self, reference_and_current_with_drift):
        ref, cur = reference_and_current_with_drift
        report = compute_psi(ref, cur, columns=["x2"])
        assert [a.column for a in report.assessments] == ["x2"]


class TestDriftReport:
    def test_metrics_dict_format_matches_mlflow_contract(self):
        assessments = [
            DriftAssessment(column="x1", psi=0.05, status="stable"),
            DriftAssessment(column="x2", psi=0.30, status="drift"),
        ]
        report = DriftReport(assessments=assessments)
        metrics = report.to_metrics_dict()

        assert metrics["drift_share"] == 0.5
        assert metrics["drift_n_columns"] == 1.0
        assert metrics["warning_n_columns"] == 0.0
        assert metrics["psi__x1"] == 0.05
        assert metrics["psi__x2"] == 0.30
        assert all(isinstance(v, float) for v in metrics.values())

    def test_empty_report_does_not_divide_by_zero(self):
        report = DriftReport()
        assert report.share_of_drifted_columns == 0.0

    def test_needs_retrain_flag_only_on_drift(self):
        stable = DriftAssessment(column="a", psi=0.01, status="stable")
        warning = DriftAssessment(column="b", psi=0.12, status="warning")
        drift = DriftAssessment(column="c", psi=0.50, status="drift")
        assert stable.needs_retrain is False
        assert warning.needs_retrain is False
        assert drift.needs_retrain is True


def test_psi_handles_columns_with_nan():
    rng = np.random.default_rng(7)
    ref = pd.DataFrame({"x": rng.normal(0, 1, size=200)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, size=200)})
    cur.loc[:20, "x"] = np.nan

    report = compute_psi(ref, cur)
    assert len(report.assessments) == 1
    assert not np.isnan(report.assessments[0].psi)
