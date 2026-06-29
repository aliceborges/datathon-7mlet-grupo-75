"""Detecção de drift via PSI e Evidently."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PSI_WARNING = 0.10
PSI_RETRAIN_TRIGGER = 0.20


@dataclass
class DriftAssessment:
    """PSI e status (stable/warning/drift) de uma coluna."""

    column: str
    psi: float
    status: str

    @property
    def needs_retrain(self) -> bool:
        return self.status == "drift"


@dataclass
class DriftReport:
    """Agrega DriftAssessments e resume em métricas para MLflow."""

    assessments: list[DriftAssessment] = field(default_factory=list)

    @property
    def drifted_columns(self) -> list[str]:
        return [a.column for a in self.assessments if a.needs_retrain]

    @property
    def warning_columns(self) -> list[str]:
        return [a.column for a in self.assessments if a.status == "warning"]

    @property
    def share_of_drifted_columns(self) -> float:
        if not self.assessments:
            return 0.0
        return len(self.drifted_columns) / len(self.assessments)

    def to_metrics_dict(self) -> dict[str, float]:
        return {
            "drift_share": self.share_of_drifted_columns,
            "drift_n_columns": float(len(self.drifted_columns)),
            "warning_n_columns": float(len(self.warning_columns)),
            **{f"psi__{a.column}": a.psi for a in self.assessments},
        }


def _psi_single(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI clássico por binning em quantis. Clipa proporções para evitar log(0)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if reference.size == 0 or current.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / reference.size, 1e-6, None)
    cur_pct = np.clip(cur_counts / current.size, 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _classify(psi: float) -> str:
    if np.isnan(psi):
        return "warning"
    if psi >= PSI_RETRAIN_TRIGGER:
        return "drift"
    if psi >= PSI_WARNING:
        return "warning"
    return "stable"


def compute_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> DriftReport:
    """PSI coluna a coluna entre referência e dados atuais."""
    if columns is None:
        numeric_ref = reference.select_dtypes(include=[np.number]).columns
        numeric_cur = current.select_dtypes(include=[np.number]).columns
        columns = sorted(set(numeric_ref) & set(numeric_cur))

    assessments: list[DriftAssessment] = []
    for col in columns:
        psi = _psi_single(reference[col].to_numpy(), current[col].to_numpy(), bins=bins)
        status = _classify(psi)
        assessments.append(DriftAssessment(column=col, psi=psi, status=status))
        if status != "stable":
            logger.warning("Drift em %s: PSI=%.4f (%s)", col, psi, status)

    return DriftReport(assessments=assessments)


def run_evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Relatório DataDriftPreset do Evidently serializado em dict."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report.as_dict()
