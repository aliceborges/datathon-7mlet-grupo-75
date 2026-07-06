"""Contratos de entrada e saida da API de decisao."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CustomerContext(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=64)
    age: int = Field(..., ge=18, le=120)
    job: str | None = None
    marital: str | None = None
    education: str | None = None
    default: bool | None = None
    housing: bool | None = None
    loan: bool | None = None
    balance: float | None = None
    contact: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    context: CustomerContext
    candidate_offers: list[str] | None = None

    @field_validator("candidate_offers")
    @classmethod
    def _non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("candidate_offers nao pode ser lista vazia")
        return v


class OfferDecision(BaseModel):
    offer_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    decision_id: str
    policy_version: str
    chosen: OfferDecision
    alternatives: list[OfferDecision] = Field(default_factory=list)
    served_at: datetime


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    customer_id: str | None = None
    session_id: str | None = None


class AgentResponse(BaseModel):
    decision_id: str
    answer: str
    tools_used: list[str] = Field(default_factory=list)
    served_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


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
