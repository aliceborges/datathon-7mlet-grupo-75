"""Fixtures compartilhadas entre os testes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """DataFrame binário pequeno para smoke tests."""
    rng = np.random.default_rng(seed=42)
    n = 200
    return pd.DataFrame(
        {
            "feature_1": rng.normal(0.5, 0.15, size=n),
            "feature_2": rng.uniform(0, 10, size=n),
            "feature_3": rng.integers(0, 3, size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


@pytest.fixture
def reference_and_current_no_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Par (reference, current) com distribuições iguais."""
    rng = np.random.default_rng(seed=0)
    ref = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, size=1000),
            "x2": rng.normal(5, 2, size=1000),
        }
    )
    cur = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, size=1000),
            "x2": rng.normal(5, 2, size=1000),
        }
    )
    return ref, cur


@pytest.fixture
def reference_and_current_with_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Par (reference, current) com shift de média em x1."""
    rng = np.random.default_rng(seed=1)
    ref = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, size=1000),
            "x2": rng.normal(5, 2, size=1000),
        }
    )
    cur = pd.DataFrame(
        {
            "x1": rng.normal(2.5, 1, size=1000),
            "x2": rng.normal(5, 2, size=1000),
        }
    )
    return ref, cur
