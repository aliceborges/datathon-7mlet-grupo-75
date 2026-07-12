from __future__ import annotations

from evaluation.golden_set import (
    DEFAULT_PATH,
    GoldenSample,
    count_by_category,
    load_golden_set,
)


def test_golden_set_file_exists():
    assert DEFAULT_PATH.exists(), f"Golden set não encontrado em {DEFAULT_PATH}"


def test_golden_set_has_at_least_twenty_samples():
    samples = load_golden_set()
    assert len(samples) >= 20


def test_each_sample_has_required_fields():
    samples = load_golden_set()
    for s in samples:
        assert s.id
        assert s.query
        assert s.expected_answer
        assert isinstance(s.contexts, list)
        assert isinstance(s.expected_tools, list)


def test_sample_ids_are_unique():
    samples = load_golden_set()
    ids = [s.id for s in samples]
    assert len(set(ids)) == len(ids)


def test_categories_cover_multiple_types():
    samples = load_golden_set()
    coverage = count_by_category(samples)
    # pelo menos 4 categorias distintas
    assert len(coverage) >= 4


def test_count_by_category_sums_to_total():
    samples = load_golden_set()
    coverage = count_by_category(samples)
    assert sum(coverage.values()) == len(samples)


def test_contexts_are_non_empty_for_most_samples():
    samples = load_golden_set()
    with_contexts = [s for s in samples if s.contexts]
    # pelo menos 80% das amostras devem ter contexto
    assert len(with_contexts) / len(samples) >= 0.8


def test_golden_sample_is_frozen():
    sample = GoldenSample(id="t1", query="q", expected_answer="a")
    try:
        sample.id = "outro"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("GoldenSample deveria ser imutável (frozen)")
