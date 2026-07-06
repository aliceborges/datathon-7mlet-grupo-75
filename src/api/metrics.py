"""Métricas Prometheus customizadas do domínio (além das HTTP genéricas)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

decisions_total = Counter(
    "datathon_decisions_total",
    "Decisões servidas pela API agrupadas por endpoint, policy e oferta escolhida.",
    labelnames=("endpoint", "policy_version", "chosen_offer"),
)

agent_latency_seconds = Histogram(
    "datathon_agent_latency_seconds",
    "Latência da invocação do agente ReAct em segundos.",
    labelnames=("policy_version",),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

agent_tools_used = Histogram(
    "datathon_agent_tools_used",
    "Quantidade de tools chamadas pelo agente por requisição.",
    labelnames=("policy_version",),
    buckets=(0, 1, 2, 3, 4, 5, 6, 7, 8),
)
