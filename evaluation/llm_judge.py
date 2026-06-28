"""LLM-as-judge: 4 critérios de avaliação por amostra."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evaluation.golden_set import GoldenSample
from evaluation.ragas_eval import RAGAnswer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeCriterion:
    name: str
    description: str


CRITERIA: tuple[JudgeCriterion, ...] = (
    JudgeCriterion(
        name="factual_correctness",
        description=(
            "A resposta está factualmente alinhada com a resposta esperada? "
            "Sem inventar números, ofertas ou condições não suportadas."
        ),
    ),
    JudgeCriterion(
        name="relevance",
        description=(
            "A resposta responde diretamente à pergunta feita, sem desviar "
            "ou enrolar?"
        ),
    ),
    JudgeCriterion(
        name="completeness",
        description=(
            "A resposta cobre os pontos-chave da resposta esperada (ofertas, "
            "critérios, condições) sem omitir o essencial?"
        ),
    ),
    JudgeCriterion(
        name="business_alignment",
        description=(
            "A resposta respeita as regras de negócio do catálogo: nomes de "
            "ofertas corretos, restrições (loan=True bloqueia loan_personal, "
            "housing=True desbloqueia insurance_basic), reason codes válidos?"
        ),
    ),
)


@dataclass
class CriterionScore:
    score: float
    justification: str


@dataclass
class SampleJudgement:
    sample_id: str
    scores: dict[str, CriterionScore]

    @property
    def average(self) -> float:
        if not self.scores:
            return 0.0
        return sum(c.score for c in self.scores.values()) / len(self.scores)


@dataclass
class JudgeAggregate:
    judgements: list[SampleJudgement]

    def per_criterion_mean(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for j in self.judgements:
            for name, score in j.scores.items():
                out.setdefault(name, []).append(score.score)
        return {name: sum(vals) / len(vals) for name, vals in out.items() if vals}

    def overall_mean(self) -> float:
        per = self.per_criterion_mean()
        if not per:
            return 0.0
        return sum(per.values()) / len(per)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": len(self.judgements),
            "n_criteria": len(CRITERIA),
            "criteria": [c.name for c in CRITERIA],
            "per_criterion_mean": {
                k: round(v, 4) for k, v in self.per_criterion_mean().items()
            },
            "overall_mean": round(self.overall_mean(), 4),
        }


JUDGE_PROMPT = """Você é um avaliador imparcial de respostas de um agente bancário.

Pergunta: {query}
Resposta do agente: {answer}
Resposta esperada: {ground_truth}

Avalie o critério "{criterion_name}":
{criterion_description}

Responda APENAS com um JSON válido no formato:
{{"score": <float entre 0.0 e 1.0>, "justification": "<frase curta>"}}"""


def _call_llm_judge(
    llm: Any,
    query: str,
    answer: str,
    ground_truth: str,
    criterion: JudgeCriterion,
) -> CriterionScore:
    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        ground_truth=ground_truth,
        criterion_name=criterion.name,
        criterion_description=criterion.description,
    )
    raw = llm.invoke(prompt)
    text = getattr(raw, "content", str(raw))
    try:
        payload = json.loads(text)
        return CriterionScore(
            score=float(payload["score"]),
            justification=str(payload.get("justification", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Judge parse falhou para %s: %s", criterion.name, exc)
        return CriterionScore(score=0.0, justification=f"parse_error: {exc}")


def judge_sample(
    llm: Any,
    sample: GoldenSample,
    rag_answer: RAGAnswer,
    criteria: tuple[JudgeCriterion, ...] = CRITERIA,
) -> SampleJudgement:
    scores: dict[str, CriterionScore] = {}
    for c in criteria:
        scores[c.name] = _call_llm_judge(
            llm=llm,
            query=sample.query,
            answer=rag_answer.answer,
            ground_truth=sample.expected_answer,
            criterion=c,
        )
    return SampleJudgement(sample_id=sample.id, scores=scores)


RAGCallable = Callable[[str], RAGAnswer]


def judge_golden_set(
    llm: Any,
    samples: list[GoldenSample],
    rag_fn: RAGCallable,
    criteria: tuple[JudgeCriterion, ...] = CRITERIA,
) -> JudgeAggregate:
    judgements = [
        judge_sample(llm=llm, sample=s, rag_answer=rag_fn(s.query), criteria=criteria)
        for s in samples
    ]
    return JudgeAggregate(judgements=judgements)


# ----------- mocks --------------------------------------------------------


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in text.split() if len(t) > 3}


class DeterministicJudgeLLM:
    """LLM fake que devolve JSON válido com score por overlap lexical."""

    def __init__(self) -> None:
        self._criterion_to_method = {
            "factual_correctness": self._factual,
            "relevance": self._relevance,
            "completeness": self._completeness,
            "business_alignment": self._business,
        }

    def invoke(self, prompt: str) -> str:
        query = _extract(prompt, "Pergunta:", "Resposta do agente:")
        answer = _extract(prompt, "Resposta do agente:", "Resposta esperada:")
        ground = _extract(prompt, "Resposta esperada:", "Avalie o critério")
        criterion_name = _extract(prompt, 'Avalie o critério "', '":')

        scorer = self._criterion_to_method.get(criterion_name, self._factual)
        score = scorer(query=query, answer=answer, ground_truth=ground)
        return json.dumps(
            {
                "score": round(score, 4),
                "justification": f"deterministic_{criterion_name}",
            }
        )

    def _factual(self, query: str, answer: str, ground_truth: str) -> float:
        a = _token_set(answer)
        g = _token_set(ground_truth)
        if not a or not g:
            return 0.0
        return len(a & g) / len(a | g)

    def _relevance(self, query: str, answer: str, ground_truth: str) -> float:
        a = _token_set(answer)
        q = _token_set(query)
        if not a or not q:
            return 0.0
        return len(a & q) / len(q)

    def _completeness(self, query: str, answer: str, ground_truth: str) -> float:
        a = _token_set(answer)
        g = _token_set(ground_truth)
        if not g:
            return 0.0
        return len(a & g) / len(g)

    def _business(self, query: str, answer: str, ground_truth: str) -> float:
        offer_names = {
            "loan_personal",
            "loan_payroll",
            "credit_card_gold",
            "savings_premium",
            "insurance_basic",
        }
        a_lower = answer.lower()
        g_lower = ground_truth.lower()
        mentioned_in_truth = {o for o in offer_names if o in g_lower}
        if not mentioned_in_truth:
            return 1.0 if not any(o in a_lower for o in offer_names) else 0.5
        mentioned_in_answer = {o for o in offer_names if o in a_lower}
        return len(mentioned_in_truth & mentioned_in_answer) / len(mentioned_in_truth)


def _extract(text: str, start: str, end: str) -> str:
    try:
        i = text.index(start) + len(start)
        j = text.index(end, i)
        return text[i:j].strip()
    except ValueError:
        return ""
