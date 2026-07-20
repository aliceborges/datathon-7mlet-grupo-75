from __future__ import annotations

import json

import pytest

from evaluation.golden_set import GoldenSample
from evaluation.llm_judge import (
    CRITERIA,
    CriterionScore,
    DeterministicJudgeLLM,
    JudgeAggregate,
    SampleJudgement,
    judge_golden_set,
    judge_sample,
)
from evaluation.ragas_eval import RAGAnswer


@pytest.fixture
def llm() -> DeterministicJudgeLLM:
    return DeterministicJudgeLLM()


@pytest.fixture
def sample() -> GoldenSample:
    return GoldenSample(
        id="s1",
        query="Qual oferta para clientes com housing?",
        expected_answer="insurance_basic é indicado para clientes com housing True.",
        contexts=["insurance_basic indicado para housing True"],
    )


def test_criteria_count_is_at_least_three():
    assert len(CRITERIA) >= 3


def test_criteria_include_business_alignment():
    names = {c.name for c in CRITERIA}
    assert "business_alignment" in names


class TestDeterministicJudgeLLM:
    def test_returns_valid_json(self, llm, sample):
        prompt = (
            f"Pergunta: {sample.query}\n"
            f"Resposta do agente: insurance_basic é a melhor opção\n"
            f"Resposta esperada: {sample.expected_answer}\n"
            f'Avalie o critério "factual_correctness":\n'
        )
        out = llm.invoke(prompt)
        data = json.loads(out)
        assert "score" in data
        assert "justification" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_business_alignment_credits_correct_offer_mention(self, llm):
        prompt = (
            "Pergunta: q\n"
            "Resposta do agente: insurance_basic é a oferta indicada\n"
            "Resposta esperada: insurance_basic é a melhor opção para housing\n"
            'Avalie o critério "business_alignment":\n'
        )
        data = json.loads(llm.invoke(prompt))
        assert data["score"] == 1.0

    def test_business_alignment_penalizes_wrong_offer(self, llm):
        prompt = (
            "Pergunta: q\n"
            "Resposta do agente: loan_personal é a oferta certa\n"
            "Resposta esperada: insurance_basic é a melhor opção\n"
            'Avalie o critério "business_alignment":\n'
        )
        data = json.loads(llm.invoke(prompt))
        assert data["score"] < 1.0


class TestJudgeSample:
    def test_returns_score_per_criterion(self, llm, sample):
        rag_answer = RAGAnswer(
            answer="insurance_basic é a oferta", contexts=sample.contexts
        )
        result = judge_sample(llm=llm, sample=sample, rag_answer=rag_answer)
        assert isinstance(result, SampleJudgement)
        assert result.sample_id == "s1"
        assert set(result.scores.keys()) == {c.name for c in CRITERIA}
        for s in result.scores.values():
            assert isinstance(s, CriterionScore)

    def test_average_is_in_unit_range(self, llm, sample):
        rag_answer = RAGAnswer(
            answer="insurance_basic é a oferta", contexts=sample.contexts
        )
        result = judge_sample(llm=llm, sample=sample, rag_answer=rag_answer)
        assert 0.0 <= result.average <= 1.0


class TestJudgeAggregate:
    def test_per_criterion_mean_covers_all_criteria(self, llm, sample):
        def fake_rag(q: str) -> RAGAnswer:
            return RAGAnswer(answer="insurance_basic", contexts=sample.contexts)

        agg = judge_golden_set(llm=llm, samples=[sample, sample], rag_fn=fake_rag)
        means = agg.per_criterion_mean()
        assert set(means.keys()) == {c.name for c in CRITERIA}

    def test_overall_mean_matches_average_of_criteria(self, llm, sample):
        def fake_rag(q: str) -> RAGAnswer:
            return RAGAnswer(answer="insurance_basic", contexts=sample.contexts)

        agg = judge_golden_set(llm=llm, samples=[sample], rag_fn=fake_rag)
        per = agg.per_criterion_mean()
        expected = sum(per.values()) / len(per)
        assert agg.overall_mean() == pytest.approx(expected)

    def test_to_dict_has_expected_keys(self):
        agg = JudgeAggregate(judgements=[])
        d = agg.to_dict()
        assert set(d.keys()) >= {
            "n_samples",
            "n_criteria",
            "criteria",
            "per_criterion_mean",
            "overall_mean",
        }


def test_malformed_json_response_falls_back_to_zero(monkeypatch, sample):
    class BadLLM:
        def invoke(self, prompt: str) -> str:
            return "isso não é JSON"

    rag_answer = RAGAnswer(answer="qualquer", contexts=[])
    result = judge_sample(llm=BadLLM(), sample=sample, rag_answer=rag_answer)
    for s in result.scores.values():
        assert s.score == 0.0
        assert s.justification.startswith("parse_error")
