"""CLI que roda RAGAS + LLM-as-judge contra o golden set."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluation.golden_set import (
    DEFAULT_PATH,
    GoldenSample,
    count_by_category,
    load_golden_set,
)
from evaluation.llm_judge import (
    DeterministicJudgeLLM,
    JudgeAggregate,
    judge_golden_set,
)
from evaluation.ragas_eval import RAGAnswer, RAGASScores, evaluate_with_ragas

logger = logging.getLogger(__name__)


def _mock_rag(sample_lookup: dict[str, GoldenSample]) -> Callable[[str], RAGAnswer]:
    """RAG fake que devolve a expected_answer + contexts do próprio golden set."""

    def fn(query: str) -> RAGAnswer:
        for s in sample_lookup.values():
            if s.query == query:
                return RAGAnswer(answer=s.expected_answer, contexts=list(s.contexts))
        return RAGAnswer(answer="", contexts=[])

    return fn


def _real_rag() -> Callable[[str], RAGAnswer]:
    from src.agent.rag_pipeline import build_default_pipeline
    from src.agent.react_agent import build_react_agent
    from src.agent.tools import build_default_tools

    rag = build_default_pipeline()
    tools = build_default_tools(retriever=rag)
    agent = build_react_agent(tools)

    def fn(query: str) -> RAGAnswer:
        result = agent.invoke({"input": query})
        contexts = rag.retrieve(query, k=4)
        return RAGAnswer(answer=result.get("output", ""), contexts=contexts)

    return fn


def render_markdown_report(
    ragas: RAGASScores,
    judge: JudgeAggregate,
    coverage: dict[str, int],
) -> str:
    judge_dict = judge.to_dict()
    ragas_dict = ragas.to_dict()

    lines = [
        "# Avaliação offline",
        "",
        f"Golden set: {ragas_dict['n_samples']} amostras",
        "",
        "## Cobertura por categoria",
        "",
        "| Categoria | N |",
        "| --- | --- |",
    ]
    lines.extend(f"| {k} | {v} |" for k, v in sorted(coverage.items()))
    lines.append("")
    lines.append("## RAGAS")
    lines.append("")
    lines.append("| Métrica | Score |")
    lines.append("| --- | --- |")
    lines.append(f"| faithfulness | {ragas_dict['faithfulness']} |")
    lines.append(f"| answer_relevancy | {ragas_dict['answer_relevancy']} |")
    lines.append(f"| context_precision | {ragas_dict['context_precision']} |")
    lines.append(f"| context_recall | {ragas_dict['context_recall']} |")
    lines.append("")
    lines.append("## LLM-as-judge")
    lines.append("")
    lines.append("| Critério | Score médio |")
    lines.append("| --- | --- |")
    for name, score in judge_dict["per_criterion_mean"].items():
        lines.append(f"| {name} | {score} |")
    lines.append("")
    lines.append(f"**Overall mean:** {judge_dict['overall_mean']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação offline do agente")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Usa RAG fake (espelha o golden set) + judge determinístico. Sem custo Azure.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/evaluation_results.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/evaluation_results.md"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    samples = load_golden_set(args.golden_set)
    coverage = count_by_category(samples)
    logger.info("Golden set: %d amostras (%s)", len(samples), coverage)

    if args.mock:
        rag_fn = _mock_rag({s.id: s for s in samples})
        judge_llm: Any = DeterministicJudgeLLM()
        ragas_llm: Any = None
    else:
        rag_fn = _real_rag()
        from langchain_openai import AzureChatOpenAI

        judge_llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            temperature=0.0,
        )
        ragas_llm = judge_llm

    ragas_scores = evaluate_with_ragas(samples, rag_fn, llm=ragas_llm)
    judge_aggregate = judge_golden_set(judge_llm, samples, rag_fn)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "ragas": ragas_scores.to_dict(),
                "judge": judge_aggregate.to_dict(),
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_markdown_report(ragas_scores, judge_aggregate, coverage),
        encoding="utf-8",
    )
    logger.info("Resultados em %s e %s", args.output_json, args.output_md)


if __name__ == "__main__":
    main()
