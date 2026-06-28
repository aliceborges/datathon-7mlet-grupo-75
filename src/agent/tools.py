"""Tools customizadas que o agente ReAct usa pra responder sobre decisões MAB."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain.tools import Tool

logger = logging.getLogger(__name__)


MOCK_CUSTOMER_DB: dict[str, dict[str, Any]] = {
    "C001": {
        "customer_id": "C001",
        "age": 34,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "balance": 1500.0,
        "housing": True,
        "loan": False,
    },
    "C002": {
        "customer_id": "C002",
        "age": 62,
        "job": "retired",
        "marital": "married",
        "education": "tertiary",
        "balance": 12000.0,
        "housing": False,
        "loan": False,
    },
    "C003": {
        "customer_id": "C003",
        "age": 27,
        "job": "student",
        "marital": "single",
        "education": "tertiary",
        "balance": 200.0,
        "housing": False,
        "loan": False,
    },
}


class RAGRetriever(Protocol):
    def retrieve(self, query: str, k: int = 3) -> list[str]: ...


class MetricsClient(Protocol):
    def offer_performance(self, offer_id: str) -> dict[str, float]: ...


@dataclass
class StaticMetricsClient:
    """Métricas históricas mockadas; substituível por wrapper MLflow real."""

    performance: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "loan_personal": {"ctr": 0.082, "conversion": 0.014, "n": 8423.0},
            "loan_payroll": {"ctr": 0.117, "conversion": 0.029, "n": 5210.0},
            "credit_card_gold": {"ctr": 0.063, "conversion": 0.011, "n": 12110.0},
            "savings_premium": {"ctr": 0.091, "conversion": 0.022, "n": 4302.0},
            "insurance_basic": {"ctr": 0.045, "conversion": 0.007, "n": 9871.0},
        }
    )

    def offer_performance(self, offer_id: str) -> dict[str, float]:
        if offer_id not in self.performance:
            raise KeyError(f"Sem histórico para offer_id={offer_id!r}")
        return dict(self.performance[offer_id])


@dataclass
class StubRAGRetriever:
    """Retriever placeholder com lookup por substring; usado quando não há Chroma."""

    corpus: dict[str, str] = field(
        default_factory=lambda: {
            "loan_personal": "Empréstimo pessoal padrão, taxa CDI + 2.5% a.m., prazo até 36 meses.",
            "loan_payroll": "Empréstimo consignado, taxa CDI + 0.8% a.m., desconto direto em folha.",
            "credit_card_gold": "Cartão Gold sem anuidade no primeiro ano, milhas 1.5x.",
            "savings_premium": "Conta poupança com bonificação de 110% do CDI.",
            "insurance_basic": "Seguro residencial básico, cobertura para incêndio e roubo.",
        }
    )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        q = query.lower()
        matches = [
            f"[{offer}] {desc}"
            for offer, desc in self.corpus.items()
            if offer in q or any(w in desc.lower() for w in q.split())
        ]
        return matches[:k] if matches else [next(iter(self.corpus.values()))]


def _build_get_customer_context(
    customer_db: dict[str, dict[str, Any]],
) -> Tool:
    def fn(customer_id: str) -> str:
        cid = customer_id.strip().strip('"').strip("'")
        if cid not in customer_db:
            return json.dumps({"error": f"customer_id {cid!r} não encontrado"})
        return json.dumps(customer_db[cid], ensure_ascii=False)

    return Tool(
        name="get_customer_context",
        description=(
            "Recupera as features de um cliente pelo customer_id "
            "(ex.: C001). Retorna JSON com age, job, marital, balance, "
            "housing, loan e demais atributos disponíveis."
        ),
        func=fn,
    )


def _build_lookup_offer_details(retriever: RAGRetriever) -> Tool:
    def fn(query: str) -> str:
        hits = retriever.retrieve(query.strip(), k=3)
        return (
            "\n".join(hits) if hits else "Nenhum detalhe encontrado para essa oferta."
        )

    return Tool(
        name="lookup_offer_details",
        description=(
            "Busca no catálogo de ofertas detalhes sobre condições, taxas, "
            "prazos e benefícios. Entrada: nome da oferta ou termo relacionado."
        ),
        func=fn,
    )


def _build_historical_performance(metrics: MetricsClient) -> Tool:
    def fn(offer_id: str) -> str:
        oid = offer_id.strip().strip('"').strip("'")
        try:
            data = metrics.offer_performance(oid)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(data)

    return Tool(
        name="historical_performance",
        description=(
            "Retorna métricas históricas agregadas de uma oferta "
            "(CTR, taxa de conversão, volume de impressões). "
            "Entrada: offer_id (ex.: loan_payroll)."
        ),
        func=fn,
    )


def build_default_tools(
    customer_db: dict[str, dict[str, Any]] | None = None,
    retriever: RAGRetriever | None = None,
    metrics: MetricsClient | None = None,
) -> list[Tool]:
    """Monta as 3 tools default usadas pelo agente."""
    return [
        _build_get_customer_context(customer_db or MOCK_CUSTOMER_DB),
        _build_lookup_offer_details(retriever or StubRAGRetriever()),
        _build_historical_performance(metrics or StaticMetricsClient()),
    ]
