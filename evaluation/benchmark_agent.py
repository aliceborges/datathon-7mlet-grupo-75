"""Benchmark do agente em múltiplas configurações."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS: list[str] = [
    "Quais ofertas são mais indicadas para clientes acima de 60 anos?",
    "Compare loan_personal e loan_payroll em termos de taxa e prazo.",
    "Qual oferta o cliente C001 deveria receber e por quê?",
    "Qual oferta tem maior CTR histórico no portfólio?",
    "Em que situação faz sentido recomendar credit_card_gold em vez de loan_personal?",
]


@dataclass
class AgentConfig:
    name: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    use_rag: bool = True
    max_iterations: int = 8


@dataclass
class RunResult:
    question: str
    answer: str
    tools_used: list[str]
    latency_ms: float


@dataclass
class ConfigSummary:
    config: AgentConfig
    runs: list[RunResult] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(r.latency_ms for r in self.runs) if self.runs else 0.0

    @property
    def avg_tools_used(self) -> float:
        return (
            statistics.mean(len(r.tools_used) for r in self.runs) if self.runs else 0.0
        )

    @property
    def avg_answer_length(self) -> float:
        return statistics.mean(len(r.answer) for r in self.runs) if self.runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "n_runs": len(self.runs),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_tools_used": round(self.avg_tools_used, 2),
            "avg_answer_length": round(self.avg_answer_length, 2),
            "runs": [asdict(r) for r in self.runs],
        }


def run_benchmark(
    configs: list[AgentConfig],
    questions: list[str],
    agent_factory: Callable[[AgentConfig], Any],
) -> list[ConfigSummary]:
    """Roda cada (config, question) e devolve um resumo por config."""
    summaries: list[ConfigSummary] = []
    for cfg in configs:
        logger.info("Rodando config %s", cfg.name)
        agent = agent_factory(cfg)
        summary = ConfigSummary(config=cfg)
        for q in questions:
            start = time.perf_counter()
            result = agent.invoke({"input": q})
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            tools_used = [step[0].tool for step in result.get("intermediate_steps", [])]
            summary.runs.append(
                RunResult(
                    question=q,
                    answer=result.get("output", ""),
                    tools_used=tools_used,
                    latency_ms=elapsed_ms,
                )
            )
        summaries.append(summary)
    return summaries


def render_markdown_table(summaries: list[ConfigSummary]) -> str:
    """Tabela resumo em markdown com uma linha por config."""
    header = "| Config | Modelo | Temp | RAG | Max iter | Latência (ms) | Tools/req | Tam. resp. |\n"
    separator = "|---|---|---|---|---|---|---|---|\n"
    rows = []
    for s in summaries:
        c = s.config
        rows.append(
            f"| `{c.name}` | {c.model} | {c.temperature} | {'sim' if c.use_rag else 'não'} | "
            f"{c.max_iterations} | {s.avg_latency_ms:.0f} | {s.avg_tools_used:.1f} | "
            f"{s.avg_answer_length:.0f} |"
        )
    return header + separator + "\n".join(rows) + "\n"


# ----------------- mock helpers ---------------------------------------------


class _MockAgent:
    """Agente fake para gerar benchmark sem chamar LLM real."""

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg

    def invoke(self, payload: dict) -> dict:
        time.sleep(0.01 * (1 + self.cfg.temperature))  # latência sintética
        n_tools = 2 if self.cfg.use_rag else 1
        return {
            "output": (
                f"[mock][{self.cfg.name}] resposta sintética para: "
                f"{payload['input'][:60]}..."
            ),
            "intermediate_steps": [
                (_MockToolCall(f"tool_{i}"), "ok") for i in range(n_tools)
            ],
        }


@dataclass
class _MockToolCall:
    tool: str


def mock_agent_factory(cfg: AgentConfig) -> _MockAgent:
    return _MockAgent(cfg)


# ----------------- CLI -------------------------------------------------------


DEFAULT_CONFIGS: list[AgentConfig] = [
    AgentConfig(name="baseline-cold", temperature=0.0, use_rag=True),
    AgentConfig(name="baseline-warm", temperature=0.7, use_rag=True),
    AgentConfig(name="no-rag-cold", temperature=0.0, use_rag=False),
]


def _real_agent_factory(cfg: AgentConfig) -> Any:
    from langchain_openai import AzureChatOpenAI  # noqa: PLC0415

    from src.agent.rag_pipeline import build_default_pipeline  # noqa: PLC0415
    from src.agent.react_agent import build_react_agent  # noqa: PLC0415
    from src.agent.tools import (  # noqa: PLC0415
        StubRAGRetriever,
        build_default_tools,
    )

    llm = AzureChatOpenAI(
        azure_deployment=cfg.model,
        temperature=cfg.temperature,
        api_version="2024-08-01-preview",
    )
    retriever = build_default_pipeline() if cfg.use_rag else StubRAGRetriever()
    tools = build_default_tools(retriever=retriever)
    return build_react_agent(tools, llm=llm, max_iterations=cfg.max_iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark do agente ReAct")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Usa agente fake (sem custo Azure); útil para gerar template do relatório.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark_results.json"),
        help="Caminho do JSON com resultados detalhados.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/benchmark_results.md"),
        help="Caminho do markdown com tabela resumo.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    factory = mock_agent_factory if args.mock else _real_agent_factory
    summaries = run_benchmark(DEFAULT_CONFIGS, DEFAULT_QUESTIONS, factory)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([s.to_dict() for s in summaries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown_table(summaries), encoding="utf-8")
    logger.info("Resultados em %s e %s", args.output, args.markdown)


if __name__ == "__main__":
    main()
