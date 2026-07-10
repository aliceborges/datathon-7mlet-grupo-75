"""Contratos de entrada e saida da API de decisao."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerContext(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "C001",
                "age": 34,
                "job": "admin.",
                "marital": "married",
                "education": "secondary",
                "default": False,
                "housing": True,
                "loan": False,
                "balance": 1500.0,
                "contact": "cellular",
                "extras": {"campaign": 3},
            }
        }
    )

    customer_id: str = Field(
        ..., min_length=1, max_length=64, description="Identificador do cliente."
    )
    age: int = Field(..., ge=18, le=120, description="Idade do cliente.")
    job: str | None = Field(default=None, description="Profissão declarada.")
    marital: str | None = Field(default=None, description="Estado civil.")
    education: str | None = Field(default=None, description="Nível de escolaridade.")
    default: bool | None = Field(default=None, description="Histórico de inadimplência.")
    housing: bool | None = Field(
        default=None, description="Possui financiamento imobiliário."
    )
    loan: bool | None = Field(default=None, description="Possui empréstimo pessoal.")
    balance: float | None = Field(default=None, description="Saldo médio da conta.")
    contact: str | None = Field(default=None, description="Canal de contato preferencial.")
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Atributos adicionais não modelados explicitamente.",
    )


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context": {
                    "customer_id": "C001",
                    "age": 34,
                    "balance": 1500.0,
                    "housing": True,
                },
                "candidate_offers": ["loan_payroll", "savings_premium"],
            }
        }
    )

    context: CustomerContext
    candidate_offers: list[str] | None = Field(
        default=None,
        description="Lista opcional de ofertas candidatas para restringir a recomendação.",
        examples=[["loan_payroll", "savings_premium"]],
    )

    @field_validator("candidate_offers")
    @classmethod
    def _non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("candidate_offers nao pode ser lista vazia")
        return v


class OfferDecision(BaseModel):
    offer_id: str = Field(..., description="Identificador da oferta recomendada.")
    score: float = Field(..., ge=0.0, le=1.0, description="Score normalizado da decisão.")
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Códigos curtos que explicam a decisão.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision_id": "c1b2a1d4-2f5d-4b5a-a0f3-9a6ef2a6db1f",
                "policy_version": "stub-thompson-v0",
                "chosen": {
                    "offer_id": "loan_payroll",
                    "score": 0.72,
                    "reason_codes": ["thompson_sampled(seed=42)", "high_balance"],
                },
                "alternatives": [
                    {
                        "offer_id": "savings_premium",
                        "score": 0.61,
                        "reason_codes": ["low_risk", "savings_profile"],
                    }
                ],
                "served_at": "2026-06-28T12:34:56Z",
            }
        }
    )

    decision_id: str = Field(..., description="Identificador único da decisão.")
    policy_version: str = Field(..., description="Versão da política usada na recomendação.")
    chosen: OfferDecision
    alternatives: list[OfferDecision] = Field(
        default_factory=list,
        description="Outras ofertas ordenadas por relevância.",
    )
    served_at: datetime = Field(..., description="Timestamp UTC de geração da resposta.")


class AgentRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Qual a melhor oferta pro cliente C002 e por quê?",
                "customer_id": "C002",
                "session_id": "sess-123",
            }
        }
    )

    question: str = Field(
        ..., min_length=1, max_length=2000, description="Pergunta para o agente."
    )
    customer_id: str | None = Field(
        default=None, description="Identificador opcional do cliente."
    )
    session_id: str | None = Field(
        default=None, description="Identificador opcional da sessão."
    )


class AgentResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision_id": "3d65d0a0-5a0f-4e49-8d72-2f5c2a0a7f88",
                "answer": "A melhor oferta para este cliente é loan_payroll porque ...",
                "tools_used": ["lookup_offer_details", "historical_performance"],
                "served_at": "2026-06-28T12:34:56Z",
            }
        }
    )

    decision_id: str = Field(..., description="Identificador único da interação.")
    answer: str = Field(..., description="Resposta textual produzida pelo agente.")
    tools_used: list[str] = Field(
        default_factory=list, description="Tools efetivamente usadas."
    )
    served_at: datetime = Field(..., description="Timestamp UTC de geração da resposta.")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "bad_request",
                "detail": "candidate_offers nao pode ser lista vazia",
            }
        }
    )

    error: str = Field(..., description="Código curto do erro.")
    detail: str | None = Field(
        default=None, description="Mensagem legível para diagnóstico."
    )


class AuditLog(BaseModel):
    decision_id: str
    endpoint: str
    policy_version: str
    customer_id: str | None = None
    chosen_offer: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    created_at: datetime
