from __future__ import annotations

import pytest

from src.api.model_loader import (
    DEFAULT_OFFERS,
    ThompsonSamplingStub,
    load_policy,
)
from src.api.schemas import CustomerContext


@pytest.fixture
def policy() -> ThompsonSamplingStub:
    return ThompsonSamplingStub(seed=123)


@pytest.fixture
def senior_with_housing() -> CustomerContext:
    return CustomerContext(
        customer_id="C-SR-01",
        age=65,
        balance=8000.0,
        housing=True,
    )


class TestThompsonSamplingStub:
    def test_default_offers_match_arms(self, policy):
        assert set(policy.arms) == set(DEFAULT_OFFERS)

    def test_priors_initialized_to_one(self, policy):
        for arm in policy.arms:
            assert policy.alpha[arm] == 1.0
            assert policy.beta[arm] == 1.0

    def test_recommend_returns_chosen_and_alternatives(
        self, policy, senior_with_housing
    ):
        chosen, alternatives = policy.recommend(senior_with_housing)
        assert chosen.offer_id in policy.arms
        assert len(alternatives) == len(policy.arms) - 1
        all_ids = {chosen.offer_id, *(a.offer_id for a in alternatives)}
        assert all_ids == set(policy.arms)

    def test_chosen_has_highest_score(self, policy, senior_with_housing):
        chosen, alternatives = policy.recommend(senior_with_housing)
        for alt in alternatives:
            assert chosen.score >= alt.score

    def test_reason_codes_flag_senior_segment(self, policy, senior_with_housing):
        chosen, _ = policy.recommend(senior_with_housing)
        assert any("senior" in code for code in chosen.reason_codes)

    def test_reason_codes_flag_high_balance(self, policy, senior_with_housing):
        chosen, _ = policy.recommend(senior_with_housing)
        assert any("high_balance" in code for code in chosen.reason_codes)

    def test_unknown_candidate_raises(self, policy, senior_with_housing):
        with pytest.raises(ValueError, match="desconhecidas"):
            policy.recommend(senior_with_housing, candidates=["fake_offer"])

    def test_candidates_filter_works(self, policy, senior_with_housing):
        chosen, alternatives = policy.recommend(
            senior_with_housing, candidates=["loan_personal", "savings_premium"]
        )
        assert chosen.offer_id in {"loan_personal", "savings_premium"}
        assert len(alternatives) == 1

    def test_seed_makes_recommendation_reproducible(self, senior_with_housing):
        p1 = ThompsonSamplingStub(seed=7)
        p2 = ThompsonSamplingStub(seed=7)
        c1, _ = p1.recommend(senior_with_housing)
        c2, _ = p2.recommend(senior_with_housing)
        assert c1.offer_id == c2.offer_id


def test_load_policy_without_env_returns_stub(monkeypatch):
    monkeypatch.delenv("CHAMPION_MODEL_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    policy = load_policy()
    assert isinstance(policy, ThompsonSamplingStub)


def test_load_policy_falls_back_when_mlflow_fails(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:9999-nope")
    monkeypatch.setenv("CHAMPION_MODEL_NAME", "nonexistent_model")
    policy = load_policy()
    assert isinstance(policy, ThompsonSamplingStub)


def test_load_policy_skips_unreachable_tracking_uri_without_warning(
    monkeypatch, caplog
):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("CHAMPION_MODEL_NAME", "nonexistent_model")

    def _fail_connection(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(
        "src.api.model_loader.socket.create_connection", _fail_connection
    )

    with caplog.at_level("INFO"):
        policy = load_policy()

    assert isinstance(policy, ThompsonSamplingStub)
    assert not any(record.levelname == "WARNING" for record in caplog.records)
