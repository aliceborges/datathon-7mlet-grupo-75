"""API de decisão do Datathon: serve recomendações e o agente LLM."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

from src.agent.rag_pipeline import build_default_pipeline
from src.agent.react_agent import build_react_agent
from src.agent.tools import build_default_tools
from src.api.audit import build_audit_logger
from src.api.metrics import (
    agent_latency_seconds,
    agent_tools_used,
    decisions_total,
)
from src.api.model_loader import load_policy
from src.api.schemas import (
    AgentRequest,
    AgentResponse,
    AuditLog,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
)
from src.security.guardrails import InputGuardrail

logger = logging.getLogger(__name__)


def _load_environment() -> None:
    if os.environ.get("DISABLE_DOTENV") == "1":
        return
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


def _missing_azure_openai_credentials() -> list[str]:
    missing: list[str] = []
    if not (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_AD_TOKEN")):
        missing.append("AZURE_OPENAI_API_KEY ou AZURE_OPENAI_AD_TOKEN")
    if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not os.environ.get("AZURE_OPENAI_DEPLOYMENT"):
        missing.append("AZURE_OPENAI_DEPLOYMENT")
    return missing


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Inicializa policy, RAG, tools, agente e audit logger no startup."""
    _load_environment()
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    resources: dict[str, Any] = {
        "policy": load_policy(),
        "audit": build_audit_logger(),
        "agent": None,
        "guardrail": InputGuardrail(),
    }

    rag = build_default_pipeline()
    tools = build_default_tools(retriever=rag)
    missing_credentials = _missing_azure_openai_credentials()
    if missing_credentials:
        logger.info(
            "Agente ReAct desativado: credenciais Azure OpenAI ausentes (%s)",
            ", ".join(missing_credentials),
        )
    else:
        try:
            resources["agent"] = build_react_agent(tools)
            logger.info("Agente ReAct inicializado")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Agente indisponível durante a inicialização; mantendo a API em modo degradado: %s",
                exc,
            )

    app.state.resources = resources
    yield
    logger.info("API shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Datathon Grupo 75 - API de Decisão",
        version="0.1.0",
        summary="API de recomendação de ofertas e agente ReAct para explicações.",
        description=(
            "Serviço FastAPI que expõe uma política de recomendação para ofertas "
            "e um endpoint conversacional para justificar decisões. A documentação "
            "interativa fica disponível em /docs e o schema OpenAPI em /openapi.json."
        ),
        lifespan=_lifespan,
    )

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.get(
        "/health/live",
        tags=["meta"],
        summary="Liveness check",
        description="Verifica se o processo da API está vivo.",
    )
    async def live() -> dict[str, Any]:
        return {"status": "alive", "version": app.version}

    @app.get(
        "/health/ready",
        tags=["meta"],
        summary="Readiness check",
        description="Verifica se policy, audit logger e agente foram inicializados.",
    )
    async def ready() -> JSONResponse:
        resources = getattr(app.state, "resources", {})
        checks = {
            "policy": resources.get("policy") is not None,
            "audit": resources.get("audit") is not None,
            "agent": resources.get("agent") is not None,
        }
        is_ready = checks["policy"] and checks["audit"]
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"ready": is_ready, "checks": checks, "version": app.version},
        )

    @app.get("/health", tags=["meta"], include_in_schema=False)
    async def health_alias() -> dict[str, Any]:
        return {"status": "ok", "version": app.version}

    @app.post(
        "/predict",
        response_model=PredictResponse,
        responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        tags=["decision"],
        summary="Gerar recomendação",
        description=(
            "Executa a política de recomendação e retorna a oferta escolhida, "
            "alternativas e metadados de auditoria."
        ),
    )
    async def predict(req: PredictRequest) -> PredictResponse:
        policy = app.state.resources["policy"]
        audit = app.state.resources["audit"]
        try:
            chosen, alternatives = policy.recommend(req.context, req.candidate_offers)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        now = datetime.now(timezone.utc)
        decision_id = str(uuid.uuid4())
        response = PredictResponse(
            decision_id=decision_id,
            policy_version=policy.policy_version,
            chosen=chosen,
            alternatives=alternatives,
            served_at=now,
        )
        audit.write(
            AuditLog(
                decision_id=decision_id,
                endpoint="/predict",
                policy_version=policy.policy_version,
                customer_id=req.context.customer_id,
                chosen_offer=chosen.offer_id,
                reason_codes=chosen.reason_codes,
                created_at=now,
            )
        )
        decisions_total.labels(
            endpoint="/predict",
            policy_version=policy.policy_version,
            chosen_offer=chosen.offer_id,
        ).inc()
        return response

    @app.post(
        "/agent",
        response_model=AgentResponse,
        responses={
            400: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
        tags=["decision"],
        summary="Perguntar ao agente",
        description=(
            "Encaminha uma pergunta ao agente ReAct com guardrails de entrada e "
            "retorna a resposta textual com as tools usadas."
        ),
    )
    async def agent_qa(req: AgentRequest) -> AgentResponse:
        guardrail = app.state.resources["guardrail"]
        verdict = guardrail.validate(req.question)
        if not verdict.is_valid:
            raise HTTPException(status_code=400, detail=verdict.reason)

        agent = app.state.resources.get("agent")
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Agente indisponível — verifique AZURE_OPENAI_API_KEY/ENDPOINT",
            )
        audit = app.state.resources["audit"]

        start = datetime.now(timezone.utc)
        result = await asyncio.to_thread(agent.invoke, {"input": req.question})
        end = datetime.now(timezone.utc)
        elapsed_s = (end - start).total_seconds()

        tools_used = [step[0].tool for step in result.get("intermediate_steps", [])]
        decision_id = str(uuid.uuid4())
        policy_version = "agent-react-v0"
        response = AgentResponse(
            decision_id=decision_id,
            answer=result.get("output", ""),
            tools_used=tools_used,
            served_at=end,
        )
        audit.write(
            AuditLog(
                decision_id=decision_id,
                endpoint="/agent",
                policy_version=policy_version,
                customer_id=req.customer_id,
                tools_used=tools_used,
                latency_ms=elapsed_s * 1000.0,
                created_at=end,
            )
        )
        agent_latency_seconds.labels(policy_version=policy_version).observe(elapsed_s)
        agent_tools_used.labels(policy_version=policy_version).observe(len(tools_used))
        decisions_total.labels(
            endpoint="/agent",
            policy_version=policy_version,
            chosen_offer="<agent-answer>",
        ).inc()
        return response

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Erro não tratado em %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail=str(exc),
            ).model_dump(),
        )

    return app


app = create_app()
