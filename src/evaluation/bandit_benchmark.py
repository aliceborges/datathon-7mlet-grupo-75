from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol

import numpy as np

from src.api.schemas import CustomerContext, OfferDecision


DEFAULT_HISTORICAL_CONVERSION = {
    "loan_personal": 0.014,
    "loan_payroll": 0.029,
    "credit_card_gold": 0.011,
    "savings_premium": 0.022,
    "insurance_basic": 0.007,
}


class BanditPolicy(Protocol):
    policy_version: str

    def recommend(
        self,
        context: CustomerContext,
        candidates: list[str] | None = None,
    ) -> tuple[OfferDecision, list[OfferDecision]]: ...

    def update(self, offer_id: str, reward: float) -> None: ...

    def is_cold_start(self, offer_id: str) -> bool: ...


@dataclass
class DeterministicBaselinePolicy:
    historical_conversion: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_HISTORICAL_CONVERSION)
    )
    policy_version: str = "deterministic-baseline-v1"

    def recommend(
        self,
        context: CustomerContext,
        candidates: list[str] | None = None,
    ) -> tuple[OfferDecision, list[OfferDecision]]:
        pool = candidates or list(self.historical_conversion)
        ranked = sorted(
            (
                (offer_id, float(self.historical_conversion.get(offer_id, 0.0)))
                for offer_id in pool
            ),
            key=lambda item: (-item[1], pool.index(item[0])),
        )
        chosen = OfferDecision(
            offer_id=ranked[0][0],
            score=ranked[0][1],
            reason_codes=["deterministic_baseline", "highest_historical_conversion"],
        )
        alternatives = [OfferDecision(offer_id=offer, score=score) for offer, score in ranked[1:]]
        return chosen, alternatives

    def update(self, offer_id: str, reward: float) -> None:
        return None

    def is_cold_start(self, offer_id: str) -> bool:
        return offer_id not in self.historical_conversion


@dataclass
class ThompsonSamplingPolicy:
    arms: list[str]
    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    seed: int = 75
    policy_version: str = "thompson-sampling-v1"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        for arm in self.arms:
            self.alpha.setdefault(arm, 1.0)
            self.beta.setdefault(arm, 1.0)

    def _ensure_arm(self, offer_id: str) -> None:
        if offer_id not in self.alpha:
            self.alpha[offer_id] = 1.0
            self.beta[offer_id] = 1.0
            if offer_id not in self.arms:
                self.arms.append(offer_id)

    def recommend(
        self,
        context: CustomerContext,
        candidates: list[str] | None = None,
    ) -> tuple[OfferDecision, list[OfferDecision]]:
        pool = candidates or self.arms
        for offer_id in pool:
            self._ensure_arm(offer_id)

        samples = {
            offer_id: float(self._rng.beta(self.alpha[offer_id], self.beta[offer_id]))
            for offer_id in pool
        }
        ranked = sorted(samples.items(), key=lambda item: item[1], reverse=True)
        chosen_offer, chosen_score = ranked[0]
        chosen = OfferDecision(
            offer_id=chosen_offer,
            score=chosen_score,
            reason_codes=self._reason_codes_for(context, chosen_offer),
        )
        alternatives = [OfferDecision(offer_id=offer, score=score) for offer, score in ranked[1:]]
        return chosen, alternatives

    def update(self, offer_id: str, reward: float) -> None:
        self._ensure_arm(offer_id)
        if reward >= 0.5:
            self.alpha[offer_id] += 1.0
        else:
            self.beta[offer_id] += 1.0

    def is_cold_start(self, offer_id: str) -> bool:
        return offer_id not in self.alpha or (self.alpha[offer_id] == 1.0 and self.beta[offer_id] == 1.0)

    def _reason_codes_for(self, context: CustomerContext, offer_id: str) -> list[str]:
        codes = [f"thompson_sampling(seed={self.seed})"]
        if context.age >= 60:
            codes.append("senior_segment")
        if context.balance is not None and context.balance > 5000:
            codes.append("high_balance")
        if offer_id.startswith("loan") and context.housing:
            codes.append("housing_loan_synergy")
        return codes


@dataclass
class NilosUCBPolicy:
    arms: list[str]
    counts: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    c: float = 1.0
    total_rounds: int = 0
    policy_version: str = "nilos-ucb-v1"

    def __post_init__(self) -> None:
        for arm in self.arms:
            self.counts.setdefault(arm, 0)
            self.rewards.setdefault(arm, 0.0)

    def _ensure_arm(self, offer_id: str) -> None:
        if offer_id not in self.counts:
            self.counts[offer_id] = 0
            self.rewards[offer_id] = 0.0
            if offer_id not in self.arms:
                self.arms.append(offer_id)

    def recommend(
        self,
        context: CustomerContext,
        candidates: list[str] | None = None,
    ) -> tuple[OfferDecision, list[OfferDecision]]:
        pool = candidates or self.arms
        for offer_id in pool:
            self._ensure_arm(offer_id)

        scores: dict[str, float] = {}
        t = self.total_rounds
        for offer_id in pool:
            n_a = self.counts[offer_id]
            if n_a == 0:
                scores[offer_id] = float("inf")
            else:
                mu_a = self.rewards[offer_id] / n_a
                exploration_term = self.c * np.sqrt(np.log(max(1, t)) / n_a)
                scores[offer_id] = float(mu_a + exploration_term)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        chosen_offer, chosen_score = ranked[0]

        score_val = float(np.clip(chosen_score, 0.0, 1.0)) if chosen_score != float("inf") else 1.0
        chosen = OfferDecision(
            offer_id=chosen_offer,
            score=score_val,
            reason_codes=self._reason_codes_for(context, chosen_offer),
        )
        alternatives = [
            OfferDecision(
                offer_id=offer,
                score=float(np.clip(score, 0.0, 1.0)) if score != float("inf") else 1.0
            )
            for offer, score in ranked[1:]
        ]
        return chosen, alternatives

    def update(self, offer_id: str, reward: float) -> None:
        self._ensure_arm(offer_id)
        self.counts[offer_id] += 1
        self.rewards[offer_id] += reward
        self.total_rounds += 1

    def is_cold_start(self, offer_id: str) -> bool:
        return offer_id not in self.counts or self.counts[offer_id] == 0

    def _reason_codes_for(self, context: CustomerContext, offer_id: str) -> list[str]:
        codes = [f"nilos_ucb(c={self.c},rounds={self.total_rounds})"]
        if context.age >= 60:
            codes.append("senior_segment")
        if context.balance is not None and context.balance > 5000:
            codes.append("high_balance")
        if offer_id.startswith("loan") and context.housing:
            codes.append("housing_loan_synergy")
        return codes


@dataclass(frozen=True)
class BanditScenario:
    context: CustomerContext
    candidate_offers: list[str]
    true_conversion_by_offer: dict[str, float]
    reward_delay_by_offer: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BanditRoundResult:
    index: int
    chosen_offer: str
    baseline_offer: str
    best_offer: str
    reward: float
    regret: float
    exploration: bool
    cold_start: bool
    reward_delay_days: int | None


@dataclass
class BanditSimulationSummary:
    policy_version: str
    rounds: list[BanditRoundResult]

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def total_reward(self) -> float:
        return float(sum(round_result.reward for round_result in self.rounds))

    @property
    def total_regret(self) -> float:
        return float(sum(round_result.regret for round_result in self.rounds))

    @property
    def avg_regret(self) -> float:
        return self.total_regret / self.total_rounds if self.rounds else 0.0

    @property
    def conversion_rate(self) -> float:
        return self.total_reward / self.total_rounds if self.rounds else 0.0

    @property
    def exploration_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(round_result.exploration for round_result in self.rounds) / self.total_rounds

    @property
    def cold_start_rate(self) -> float:
        if not self.rounds:
            return 0.0
        return sum(round_result.cold_start for round_result in self.rounds) / self.total_rounds

    @property
    def avg_reward_delay_days(self) -> float:
        delays = [round_result.reward_delay_days for round_result in self.rounds if round_result.reward_delay_days is not None]
        return float(mean(delays)) if delays else 0.0

    @property
    def delayed_reward_rate(self) -> float:
        if not self.rounds:
            return 0.0
        delayed_rewards = [round_result for round_result in self.rounds if round_result.reward_delay_days not in (None, 0)]
        return len(delayed_rewards) / self.total_rounds

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "policy_version": self.policy_version,
            "total_rounds": self.total_rounds,
            "total_reward": self.total_reward,
            "total_regret": self.total_regret,
            "avg_regret": self.avg_regret,
            "conversion_rate": self.conversion_rate,
            "exploration_rate": self.exploration_rate,
            "cold_start_rate": self.cold_start_rate,
            "avg_reward_delay_days": self.avg_reward_delay_days,
            "delayed_reward_rate": self.delayed_reward_rate,
        }


def _best_offer(candidates: list[str], conversion_by_offer: dict[str, float]) -> str:
    return max(
        candidates,
        key=lambda offer_id: (
            conversion_by_offer.get(offer_id, 0.0),
            -candidates.index(offer_id),
        ),
    )


def simulate_bandit_policy(
    scenarios: list[BanditScenario],
    policy: BanditPolicy,
    *,
    seed: int = 75,
) -> BanditSimulationSummary:
    rng = np.random.default_rng(seed)
    rounds: list[BanditRoundResult] = []

    for index, scenario in enumerate(scenarios, start=1):
        chosen, _ = policy.recommend(scenario.context, scenario.candidate_offers)
        baseline_offer = _best_offer(
            scenario.candidate_offers,
            getattr(policy, "historical_conversion", scenario.true_conversion_by_offer),
        )
        best_offer = _best_offer(scenario.candidate_offers, scenario.true_conversion_by_offer)
        probability = float(scenario.true_conversion_by_offer.get(chosen.offer_id, 0.0))
        reward = float(rng.random() < probability)
        regret = float(scenario.true_conversion_by_offer.get(best_offer, 0.0) - probability)
        exploration = chosen.offer_id != baseline_offer
        cold_start = policy.is_cold_start(chosen.offer_id)
        reward_delay_days = scenario.reward_delay_by_offer.get(chosen.offer_id)
        if reward > 0.5:
            policy.update(chosen.offer_id, reward)
        rounds.append(
            BanditRoundResult(
                index=index,
                chosen_offer=chosen.offer_id,
                baseline_offer=baseline_offer,
                best_offer=best_offer,
                reward=reward,
                regret=regret,
                exploration=exploration,
                cold_start=cold_start,
                reward_delay_days=reward_delay_days if reward > 0.5 else None,
            )
        )

    return BanditSimulationSummary(policy_version=policy.policy_version, rounds=rounds)


def compare_bandit_policies(
    scenarios: list[BanditScenario],
    *,
    seed: int = 75,
) -> dict[str, BanditSimulationSummary | dict[str, float]]:
    baseline = simulate_bandit_policy(
        scenarios,
        DeterministicBaselinePolicy(),
        seed=seed,
    )
    adaptive = simulate_bandit_policy(
        scenarios,
        ThompsonSamplingPolicy(
            arms=list(
                dict.fromkeys(
                    offer_id
                    for scenario in scenarios
                    for offer_id in scenario.candidate_offers
                )
            ),
            seed=seed,
        ),
        seed=seed,
    )
    return {
        "baseline": baseline,
        "adaptive": adaptive,
        "delta": {
            "reward": adaptive.total_reward - baseline.total_reward,
            "regret": baseline.total_regret - adaptive.total_regret,
            "conversion_rate": adaptive.conversion_rate - baseline.conversion_rate,
            "exploration_rate": adaptive.exploration_rate - baseline.exploration_rate,
            "cold_start_rate": adaptive.cold_start_rate - baseline.cold_start_rate,
        },
    }