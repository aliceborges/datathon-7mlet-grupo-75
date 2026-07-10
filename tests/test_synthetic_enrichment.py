from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.synthetic_enrichment.generator import (
    SyntheticEnrichmentBundle,
    generate_synthetic_enrichment,
    write_synthetic_enrichment,
)


def _sample_source_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [31, 42, 56, 28, 61, 35],
            "job": [
                "admin.",
                "services",
                "retired",
                "technician",
                "management",
                "admin.",
            ],
            "marital": [
                "married",
                "single",
                "married",
                "single",
                "divorced",
                "married",
            ],
            "education": [
                "university.degree",
                "high.school",
                "basic.9y",
                "high.school",
                "university.degree",
                "basic.4y",
            ],
            "contact": [
                "cellular",
                "telephone",
                "cellular",
                "telephone",
                "cellular",
                "cellular",
            ],
            "campaign": [1, 2, 1, 3, 1, 2],
            "y": ["yes", "no", "no", "yes", "no", "yes"],
        }
    )


def test_generate_synthetic_enrichment_is_deterministic_by_seed():
    source_df = _sample_source_df()

    first = generate_synthetic_enrichment(source_df, seed=123, n_events=25)
    second = generate_synthetic_enrichment(source_df, seed=123, n_events=25)

    pd.testing.assert_frame_equal(first.offer_catalog, second.offer_catalog)
    pd.testing.assert_frame_equal(first.offer_events, second.offer_events)
    pd.testing.assert_frame_equal(first.delayed_rewards, second.delayed_rewards)


def test_generate_synthetic_enrichment_produces_expected_schema():
    bundle = generate_synthetic_enrichment(_sample_source_df(), seed=7, n_events=18)

    assert isinstance(bundle, SyntheticEnrichmentBundle)
    assert bundle.offer_catalog.columns.tolist() == [
        "offer_id",
        "offer_type",
        "channel",
        "reward_type",
        "reward_value",
        "discount_rate",
        "target_job",
        "target_age_min",
        "target_age_max",
        "valid_from",
        "valid_until",
        "expected_acceptance_rate",
        "expected_reward_delay_days",
    ]
    assert bundle.offer_events.columns.tolist() == [
        "event_id",
        "source_row_id",
        "client_id",
        "offer_id",
        "offer_type",
        "channel",
        "event_timestamp",
        "age",
        "age_band",
        "job",
        "marital",
        "education",
        "contact",
        "source_y",
        "propensity_score",
        "accepted_offer",
        "converted_with_delay",
        "reward_delay_days",
    ]
    assert bundle.delayed_rewards.columns.tolist() == [
        "reward_id",
        "event_id",
        "client_id",
        "offer_id",
        "reward_timestamp",
        "delay_days",
        "reward_value",
        "reward_type",
        "reward_reason",
    ]


def test_delayed_rewards_are_consistent_with_events():
    bundle = generate_synthetic_enrichment(_sample_source_df(), seed=99, n_events=40)
    events = bundle.offer_events.set_index("event_id")

    rewarded_events = set(bundle.delayed_rewards["event_id"])
    converted_events = set(events.index[events["converted_with_delay"]])

    assert rewarded_events.issubset(converted_events)
    assert rewarded_events == set(events.index[events["reward_delay_days"].notna()])

    for reward in bundle.delayed_rewards.itertuples(index=False):
        event = events.loc[reward.event_id]
        assert reward.delay_days == event.reward_delay_days
        assert reward.reward_timestamp > event.event_timestamp


def test_write_synthetic_enrichment_writes_three_csv_files(tmp_path: Path):
    bundle = generate_synthetic_enrichment(_sample_source_df(), seed=5, n_events=12)
    outputs = write_synthetic_enrichment(bundle, tmp_path)

    assert set(outputs.keys()) == {"offer_catalog", "offer_events", "delayed_rewards"}
    for path in outputs.values():
        assert path.exists()
        assert path.suffix == ".csv"
