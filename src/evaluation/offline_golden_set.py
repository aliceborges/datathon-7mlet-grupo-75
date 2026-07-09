"""Avaliacao offline reproduzivel para o golden set versionado."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from src.api.schemas import CustomerContext
from src.evaluation.bandit_benchmark import (
    DEFAULT_HISTORICAL_CONVERSION,
    BanditPolicy,
    BanditScenario,
    DeterministicBaselinePolicy,
    ThompsonSamplingPolicy,
    compare_bandit_policies,
)


GOLDEN_SET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "golden_set" / "evaluation_cases.jsonl"
)


@dataclass(frozen=True)
class GoldenSetCase:
    case_id: str
    segment: str
    context: CustomerContext
    candidate_offers: list[str]
    expected_action: str
    expected_reward: float
    justification: str
    pass_criteria: str
    policy_should_not_be_used: bool = False
    sensitivity_probe: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GoldenSetCase":
        return cls(
            case_id=str(payload["case_id"]),
            segment=str(payload["segment"]),
            context=CustomerContext.model_validate(payload["context"]),
            candidate_offers=list(payload["candidate_offers"]),
            expected_action=str(payload["expected_action"]),
            expected_reward=float(payload["expected_reward"]),
            justification=str(payload["justification"]),
            pass_criteria=str(payload["pass_criteria"]),
            policy_should_not_be_used=bool(payload.get("policy_should_not_be_used", False)),
            sensitivity_probe=dict(payload.get("sensitivity_probe", {})),
            risk_flags=list(payload.get("risk_flags", [])),
        )


@dataclass(frozen=True)
class PolicyCaseResult:
    case_id: str
    segment: str
    chosen_offer: str
    expected_action: str
    expected_reward: float
    chosen_score: float
    reward_gap: float
    passed: bool
    out_of_scope: bool
    sensitivity_flip: bool


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    policy_version: str
    total_cases: int
    eligible_cases: int
    out_of_scope_cases: int
    passed_cases: int
    pass_rate: float
    average_reward_gap: float
    sensitivity_flip_rate: float
    fairness_gap: float
    segment_exposure: dict[str, float]
    results: list[PolicyCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "total_cases": self.total_cases,
            "eligible_cases": self.eligible_cases,
            "out_of_scope_cases": self.out_of_scope_cases,
            "passed_cases": self.passed_cases,
            "pass_rate": self.pass_rate,
            "average_reward_gap": self.average_reward_gap,
            "sensitivity_flip_rate": self.sensitivity_flip_rate,
            "fairness_gap": self.fairness_gap,
            "segment_exposure": dict(self.segment_exposure),
            "results": [result.__dict__ for result in self.results],
        }


@dataclass(frozen=True)
class OfflineGoldenSetSummary:
    baseline: PolicyEvaluationSummary
    adaptive: PolicyEvaluationSummary
    policy_should_not_be_used_cases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "policy_should_not_be_used_cases": list(self.policy_should_not_be_used_cases),
        }


def load_golden_set_cases(path: Path = GOLDEN_SET_PATH) -> list[GoldenSetCase]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set nao encontrado: {path}")

    cases: list[GoldenSetCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        case = GoldenSetCase.from_payload(payload)
        if not case.candidate_offers:
            raise ValueError(f"case {case.case_id} sem candidate_offers na linha {line_number}")
        cases.append(case)

    return cases


def _build_scenarios(cases: list[GoldenSetCase]) -> list[BanditScenario]:
    return [
        BanditScenario(
            context=case.context,
            candidate_offers=case.candidate_offers,
            true_conversion_by_offer={
                offer_id: float(DEFAULT_HISTORICAL_CONVERSION.get(offer_id, 0.0))
                for offer_id in case.candidate_offers
            },
        )
        for case in cases
    ]


def _fresh_policy(policy_version: str, candidate_offers: list[str], seed: int) -> BanditPolicy:
    if policy_version == DeterministicBaselinePolicy().policy_version:
        return DeterministicBaselinePolicy()
    return ThompsonSamplingPolicy(arms=list(dict.fromkeys(candidate_offers)), seed=seed)


def _apply_probe(case: GoldenSetCase) -> list[str]:
    candidate_offers = list(case.candidate_offers)
    offer_id = case.sensitivity_probe.get("offer_id")
    if not offer_id:
        return candidate_offers
    filtered = [candidate for candidate in candidate_offers if candidate != offer_id]
    return filtered or candidate_offers


def _score_offer(offer_id: str) -> float:
    return float(DEFAULT_HISTORICAL_CONVERSION.get(offer_id, 0.0))


def _evaluate_policy(
    cases: list[GoldenSetCase],
    *,
    policy_version: str,
    rounds: list[dict[str, Any]],
    seed: int,
) -> PolicyEvaluationSummary:
    results: list[PolicyCaseResult] = []
    segment_totals: Counter[str] = Counter()
    segment_passes: Counter[str] = Counter()
    reward_gaps: list[float] = []
    flip_count = 0

    for case, round_result in zip(cases, rounds, strict=True):
        chosen_offer = str(round_result["chosen_offer"])
        chosen_score = _score_offer(chosen_offer)
        eligible = not case.policy_should_not_be_used
        passed = eligible and chosen_offer == case.expected_action
        reward_gap = case.expected_reward - chosen_score

        if eligible:
            segment_totals[case.segment] += 1
            if passed:
                segment_passes[case.segment] += 1
                reward_gaps.append(reward_gap)

        probe_offers = _apply_probe(case)
        probe_policy = _fresh_policy(policy_version, probe_offers, seed)
        probe_chosen, _ = probe_policy.recommend(case.context, probe_offers)
        sensitivity_flip = eligible and probe_chosen.offer_id != chosen_offer
        if sensitivity_flip:
            flip_count += 1

        results.append(
            PolicyCaseResult(
                case_id=case.case_id,
                segment=case.segment,
                chosen_offer=chosen_offer,
                expected_action=case.expected_action,
                expected_reward=case.expected_reward,
                chosen_score=chosen_score,
                reward_gap=reward_gap,
                passed=passed,
                out_of_scope=case.policy_should_not_be_used,
                sensitivity_flip=sensitivity_flip,
            )
        )

    eligible_cases = sum(1 for case in cases if not case.policy_should_not_be_used)
    out_of_scope_cases = len(cases) - eligible_cases
    passed_cases = sum(1 for result in results if result.passed)
    pass_rate = passed_cases / eligible_cases if eligible_cases else 0.0
    average_reward_gap = mean(reward_gaps) if reward_gaps else 0.0
    sensitivity_flip_rate = flip_count / eligible_cases if eligible_cases else 0.0
    segment_exposure = {
        segment: segment_passes[segment] / segment_totals[segment]
        for segment in segment_totals
        if segment_totals[segment]
    }
    fairness_gap = (
        max(segment_exposure.values()) - min(segment_exposure.values())
        if segment_exposure
        else 0.0
    )

    return PolicyEvaluationSummary(
        policy_version=policy_version,
        total_cases=len(cases),
        eligible_cases=eligible_cases,
        out_of_scope_cases=out_of_scope_cases,
        passed_cases=passed_cases,
        pass_rate=pass_rate,
        average_reward_gap=average_reward_gap,
        sensitivity_flip_rate=sensitivity_flip_rate,
        fairness_gap=fairness_gap,
        segment_exposure=segment_exposure,
        results=results,
    )


def run_offline_evaluation(
    *,
    path: Path = GOLDEN_SET_PATH,
    seed: int = 75,
) -> OfflineGoldenSetSummary:
    cases = load_golden_set_cases(path)
    scenarios = _build_scenarios(cases)
    comparison = compare_bandit_policies(scenarios, seed=seed)

    baseline_summary = comparison["baseline"]
    adaptive_summary = comparison["adaptive"]

    baseline = _evaluate_policy(
        cases,
        policy_version=baseline_summary.policy_version,
        rounds=[round_result.__dict__ for round_result in baseline_summary.rounds],
        seed=seed,
    )
    adaptive = _evaluate_policy(
        cases,
        policy_version=adaptive_summary.policy_version,
        rounds=[round_result.__dict__ for round_result in adaptive_summary.rounds],
        seed=seed,
    )

    policy_should_not_be_used_cases = [case.case_id for case in cases if case.policy_should_not_be_used]
    return OfflineGoldenSetSummary(
        baseline=baseline,
        adaptive=adaptive,
        policy_should_not_be_used_cases=policy_should_not_be_used_cases,
    )


def render_markdown_report(summary: OfflineGoldenSetSummary) -> str:
    lines = [
        "# Offline Golden Set Evaluation",
        "",
        f"- Baseline policy: `{summary.baseline.policy_version}`",
        f"- Adaptive policy: `{summary.adaptive.policy_version}`",
        f"- Cases out of scope: {len(summary.policy_should_not_be_used_cases)}",
        "",
        "## Key metrics",
        "",
        "| Policy | Pass rate | Sensitivity flips | Fairness gap | Avg reward gap |",
        "| --- | --- | --- | --- | --- |",
        f"| `{summary.baseline.policy_version}` | {summary.baseline.pass_rate:.3f} | {summary.baseline.sensitivity_flip_rate:.3f} | {summary.baseline.fairness_gap:.3f} | {summary.baseline.average_reward_gap:.4f} |",
        f"| `{summary.adaptive.policy_version}` | {summary.adaptive.pass_rate:.3f} | {summary.adaptive.sensitivity_flip_rate:.3f} | {summary.adaptive.fairness_gap:.3f} | {summary.adaptive.average_reward_gap:.4f} |",
        "",
        "## Segment exposure",
        "",
        "| Segment | Baseline | Adaptive |",
        "| --- | --- | --- |",
    ]

    all_segments = sorted(
        set(summary.baseline.segment_exposure) | set(summary.adaptive.segment_exposure)
    )
    for segment in all_segments:
        lines.append(
            f"| {segment} | {summary.baseline.segment_exposure.get(segment, 0.0):.3f} | {summary.adaptive.segment_exposure.get(segment, 0.0):.3f} |"
        )

    lines.extend([
        "",
        "## Out of scope cases",
        "",
    ])

    if summary.policy_should_not_be_used_cases:
        for case_id in summary.policy_should_not_be_used_cases:
            lines.append(f"- {case_id}")
    else:
        lines.append("- none")

    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline golden set evaluation.")
    parser.add_argument("--path", type=Path, default=GOLDEN_SET_PATH)
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--json", action="store_true", help="Print the raw JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_offline_evaluation(path=args.path, seed=args.seed)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_markdown_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())