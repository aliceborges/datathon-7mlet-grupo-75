"""Carrega o modelo champion do MLflow ou cai num stub Thompson Sampling."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.api.schemas import CustomerContext, OfferDecision

logger = logging.getLogger(__name__)

DEFAULT_OFFERS = [
    "loan_personal",
    "loan_payroll",
    "credit_card_gold",
    "savings_premium",
    "insurance_basic",
]


class RecommendationPolicy(Protocol):
    """Interface mínima de qualquer política de recomendação."""

    policy_version: str

    def recommend(
        self,
        context: CustomerContext,
        candidates: list[str] | None = None,
    ) -> tuple[OfferDecision, list[OfferDecision]]: ...


@dataclass
class ThompsonSamplingStub:
    """Baseline contextual-bandit-livre usado enquanto o MAB treinado não está no registry."""

    arms: list[str] = field(default_factory=lambda: list(DEFAULT_OFFERS))
    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    policy_version: str = "stub-thompson-v0"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        for arm in self.arms:
            self.alpha.setdefault(arm, 1.0)
            self.beta.setdefault(arm, 1.0)

    def recommend(
        self,
        context: CustomerContext,
        candidates: list[str] | None = None,
    ) -> tuple[OfferDecision, list[OfferDecision]]:
        pool = candidates or self.arms
        unknown = [c for c in pool if c not in self.alpha]
        if unknown:
            raise ValueError(f"Ofertas desconhecidas: {unknown}")

        samples = {c: float(self._rng.beta(self.alpha[c], self.beta[c])) for c in pool}
        ranked = sorted(samples.items(), key=lambda kv: kv[1], reverse=True)

        chosen = OfferDecision(
            offer_id=ranked[0][0],
            score=ranked[0][1],
            reason_codes=self._reason_codes_for(context, ranked[0][0]),
        )
        alternatives = [OfferDecision(offer_id=a, score=s) for a, s in ranked[1:]]
        return chosen, alternatives

    def _reason_codes_for(self, context: CustomerContext, offer_id: str) -> list[str]:
        codes = [f"thompson_sampled(seed={self.seed})"]
        if context.age >= 60:
            codes.append("senior_segment")
        if context.balance is not None and context.balance > 5000:
            codes.append("high_balance")
        if offer_id.startswith("loan") and context.housing:
            codes.append("housing_loan_synergy")
        return codes


def load_policy() -> RecommendationPolicy:
    """Tenta carregar champion do MLflow Registry; cai no stub se não disponível."""
    model_name = os.environ.get("CHAMPION_MODEL_NAME")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri or not model_name:
        logger.info("MLflow não configurado, usando ThompsonSamplingStub")
        return ThompsonSamplingStub()

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.pyfunc.load_model(f"models:/{model_name}/champion")
        logger.info("Champion %s carregado do MLflow Registry", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falha ao carregar champion (%s); usando stub", exc)

    return ThompsonSamplingStub()
