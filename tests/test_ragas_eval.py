from __future__ import annotations

import pytest

from evaluation.golden_set import GoldenSample
from evaluation.ragas_eval import (
    RAGAnswer,
    RAGASScores,
    build_dataset_records,
    evaluate_with_ragas,
)


@pytest.fixture
def samples() -> list[GoldenSample]:
    return [
        GoldenSample(
            id="s1",
            query="Qual oferta para clientes com housing?",
            expected_answer="insurance_basic é indicado para clientes com housing True.",
            contexts=["insurance_basic indicado para housing True"],
        ),
        GoldenSample(
            id="s2",
            query="Maior taxa de conversão?",
            expected_answer="loan_payroll tem maior conversão histórica.",
            contexts=["loan_payroll: conversion 0.029, maior do portfolio"],
        ),
    ]


def _fake_rag(query: str) -> RAGAnswer:
    if "housing" in query:
        return RAGAnswer(
            answer="insurance_basic é a oferta para clientes housing True",
            contexts=["insurance_basic indicado para housing True"],
        )
    return RAGAnswer(
        answer="loan_payroll tem a maior conversão histórica",
        contexts=["loan_payroll: conversion 0.029, maior do portfolio"],
    )


def test_build_dataset_records_has_expected_keys(samples):
    records = build_dataset_records(samples, _fake_rag)
    assert len(records) == 2
    for r in records:
        assert set(r.keys()) == {"question", "answer", "contexts", "ground_truth"}


def test_deterministic_scoring_returns_four_metrics(samples):
    scores = evaluate_with_ragas(samples, _fake_rag, llm=None)
    assert isinstance(scores, RAGASScores)
    assert scores.n_samples == 2
    assert 0.0 <= scores.faithfulness <= 1.0
    assert 0.0 <= scores.answer_relevancy <= 1.0
    assert 0.0 <= scores.context_precision <= 1.0
    assert 0.0 <= scores.context_recall <= 1.0


def test_scores_to_dict_has_expected_format(samples):
    scores = evaluate_with_ragas(samples, _fake_rag, llm=None)
    d = scores.to_dict()
    assert set(d.keys()) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "n_samples",
    }
    assert d["n_samples"] == 2


def test_empty_samples_returns_zero_scores():
    scores = evaluate_with_ragas([], _fake_rag, llm=None)
    assert scores.n_samples == 0
    assert scores.faithfulness == 0.0


def test_relevant_answer_scores_higher_than_irrelevant(samples):
    def good_rag(q: str) -> RAGAnswer:
        for s in samples:
            if s.query == q:
                return RAGAnswer(answer=s.expected_answer, contexts=list(s.contexts))
        return RAGAnswer("", [])

    def bad_rag(q: str) -> RAGAnswer:
        return RAGAnswer("resposta totalmente fora", ["contexto qualquer"])

    good = evaluate_with_ragas(samples, good_rag, llm=None)
    bad = evaluate_with_ragas(samples, bad_rag, llm=None)
    assert good.faithfulness >= bad.faithfulness
    assert good.context_recall >= bad.context_recall
