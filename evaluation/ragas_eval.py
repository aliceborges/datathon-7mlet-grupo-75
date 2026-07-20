"""Wrapper RAGAS com as 4 métricas obrigatórias."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evaluation.golden_set import GoldenSample

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    answer: str
    contexts: list[str]


RAGCallable = Callable[[str], RAGAnswer]


@dataclass
class RAGASScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "n_samples": self.n_samples,
        }


def build_dataset_records(
    samples: list[GoldenSample], rag_fn: RAGCallable
) -> list[dict[str, Any]]:
    """Roda o RAG sobre cada query e monta o formato esperado pelo RAGAS."""
    records: list[dict[str, Any]] = []
    for sample in samples:
        result = rag_fn(sample.query)
        records.append(
            {
                "question": sample.query,
                "answer": result.answer,
                "contexts": result.contexts,
                "ground_truth": sample.expected_answer,
            }
        )
    return records


def evaluate_with_ragas(
    samples: list[GoldenSample],
    rag_fn: RAGCallable,
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> RAGASScores:
    """Roda as 4 métricas RAGAS contra o golden set.

    Sem `llm`, devolve scores determinísticos baseados em overlap simples —
    útil pra testes e CI sem credenciais Azure.
    """
    records = build_dataset_records(samples, rag_fn)

    if llm is None:
        return _deterministic_scores(records)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return RAGASScores(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_precision=float(result["context_precision"]),
        context_recall=float(result["context_recall"]),
        n_samples=len(records),
    )


def _token_overlap(a: str, b: str) -> float:
    tokens_a = {t.lower() for t in a.split() if len(t) > 3}
    tokens_b = {t.lower() for t in b.split() if len(t) > 3}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _deterministic_scores(records: list[dict[str, Any]]) -> RAGASScores:
    """Proxies sem LLM, baseados em overlap lexical. Útil pra smoke tests."""
    if not records:
        return RAGASScores(0.0, 0.0, 0.0, 0.0, 0)

    faith_scores: list[float] = []
    relev_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []

    for r in records:
        contexts_joined = " ".join(r["contexts"])
        faith_scores.append(_token_overlap(r["answer"], contexts_joined))
        relev_scores.append(_token_overlap(r["answer"], r["question"]))
        precision_scores.append(_token_overlap(contexts_joined, r["ground_truth"]))
        recall_scores.append(_token_overlap(r["ground_truth"], contexts_joined))

    return RAGASScores(
        faithfulness=sum(faith_scores) / len(faith_scores),
        answer_relevancy=sum(relev_scores) / len(relev_scores),
        context_precision=sum(precision_scores) / len(precision_scores),
        context_recall=sum(recall_scores) / len(recall_scores),
        n_samples=len(records),
    )
