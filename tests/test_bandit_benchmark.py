from __future__ import annotations

import pytest

from src.api.schemas import CustomerContext
from src.evaluation.bandit_benchmark import (
    BanditScenario,
    DeterministicBaselinePolicy,
    ThompsonSamplingPolicy,
    NilosUCBPolicy,
    compare_bandit_policies,
    simulate_bandit_policy,
)



def _scenarios_for_regression() -> list[BanditScenario]:
    return [
        BanditScenario(
            context=CustomerContext(
                customer_id="C1",
                age=65,
                balance=8000.0,
                housing=True,
            ),
            candidate_offers=["loan_personal", "loan_payroll"],
            true_conversion_by_offer={"loan_personal": 0.0, "loan_payroll": 1.0},
            reward_delay_by_offer={"loan_payroll": 3},
        ),
        BanditScenario(
            context=CustomerContext(
                customer_id="C2",
                age=34,
                balance=1500.0,
                housing=False,
            ),
            candidate_offers=["credit_card_gold", "savings_premium"],
            true_conversion_by_offer={"credit_card_gold": 1.0, "savings_premium": 0.0},
            reward_delay_by_offer={"credit_card_gold": 2},
        ),
        BanditScenario(
            context=CustomerContext(
                customer_id="C3",
                age=28,
                balance=200.0,
                housing=False,
            ),
            candidate_offers=["new_offer"],
            true_conversion_by_offer={"new_offer": 1.0},
            reward_delay_by_offer={"new_offer": 1},
        ),
    ]


class TestDeterministicBaselinePolicy:
    def test_chooses_highest_historical_conversion(self):
        policy = DeterministicBaselinePolicy()
        context = CustomerContext(customer_id="C1", age=40)

        chosen, alternatives = policy.recommend(
            context,
            candidates=["loan_personal", "loan_payroll", "savings_premium"],
        )

        assert chosen.offer_id == "loan_payroll"
        assert chosen.score == pytest.approx(0.029)
        assert len(alternatives) == 2


class TestThompsonSamplingPolicy:
    def test_cold_start_is_cleared_after_update(self):
        policy = ThompsonSamplingPolicy(arms=["new_offer"], seed=75)

        assert policy.is_cold_start("new_offer") is True

        policy.update("new_offer", 1.0)

        assert policy.is_cold_start("new_offer") is False

    def test_recommend_is_reproducible_with_seed(self):
        context = CustomerContext(customer_id="C2", age=62, balance=12000.0)
        first = ThompsonSamplingPolicy(arms=["loan_personal", "loan_payroll"], seed=7)
        second = ThompsonSamplingPolicy(arms=["loan_personal", "loan_payroll"], seed=7)

        first_choice, _ = first.recommend(context)
        second_choice, _ = second.recommend(context)

        assert first_choice.offer_id == second_choice.offer_id


class TestBanditSimulation:
    def test_simulation_returns_expected_regression_metrics(self):
        scenarios = _scenarios_for_regression()
        baseline = simulate_bandit_policy(scenarios, DeterministicBaselinePolicy(), seed=75)
        adaptive = simulate_bandit_policy(
            scenarios,
            ThompsonSamplingPolicy(
                arms=[
                    "loan_personal",
                    "loan_payroll",
                    "credit_card_gold",
                    "savings_premium",
                    "new_offer",
                ],
                seed=75,
            ),
            seed=75,
        )

        assert baseline.total_rounds == 3
        assert baseline.total_reward == pytest.approx(2.0)
        assert baseline.total_regret == pytest.approx(1.0)
        assert baseline.exploration_rate == pytest.approx(0.0)
        assert baseline.cold_start_rate == pytest.approx(1 / 3)
        assert baseline.avg_reward_delay_days == pytest.approx(2.0)
        assert baseline.delayed_reward_rate == pytest.approx(2 / 3)

        assert adaptive.total_rounds == 3
        assert adaptive.total_reward == pytest.approx(2.0)
        assert adaptive.total_regret == pytest.approx(1.0)
        assert adaptive.exploration_rate == pytest.approx(1 / 3)
        assert adaptive.cold_start_rate == pytest.approx(1.0)
        assert adaptive.avg_reward_delay_days == pytest.approx(2.0)
        assert adaptive.delayed_reward_rate == pytest.approx(2 / 3)

    def test_comparison_reports_deltas(self):
        result = compare_bandit_policies(_scenarios_for_regression(), seed=75)

        assert result["delta"]["reward"] == pytest.approx(0.0)
        assert result["delta"]["regret"] == pytest.approx(0.0)
        assert result["delta"]["exploration_rate"] == pytest.approx(1 / 3)
        assert result["delta"]["cold_start_rate"] == pytest.approx(2 / 3)


class TestNilosUCBPolicy:
    def test_cold_start_is_cleared_after_update(self):
        policy = NilosUCBPolicy(arms=["new_offer"])
        assert policy.is_cold_start("new_offer") is True
        policy.update("new_offer", 1.0)
        assert policy.is_cold_start("new_offer") is False

    def test_recommend_returns_ucb_scores(self):
        context = CustomerContext(customer_id="C1", age=30)
        policy = NilosUCBPolicy(arms=["arm1", "arm2"], c=1.0)

        # When counts are 0, scores should be float("inf"), choosing the first (clipped to 1.0)
        chosen, alternatives = policy.recommend(context)
        assert chosen.score == 1.0

        # After updating both arms
        policy.update("arm1", 1.0)
        policy.update("arm2", 0.0)

        # arm1: count = 1, rewards = 1.0. arm2: count = 1, rewards = 0.0. total_rounds = 2.
        # Score arm1: 1.0/1 + 1.0 * sqrt(log(2)/1) = 1.0 + sqrt(log(2)) (clipped to 1.0)
        # Score arm2: 0.0/1 + 1.0 * sqrt(log(2)/1) = sqrt(log(2)) ≈ 0.8325
        chosen, alternatives = policy.recommend(context)
        assert chosen.offer_id == "arm1"
        assert chosen.score == 1.0
        assert alternatives[0].offer_id == "arm2"
        assert alternatives[0].score == pytest.approx(0.8325, abs=1e-3)

