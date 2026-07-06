from __future__ import annotations

import json

from evaluation.benchmark_agent import (
    DEFAULT_CONFIGS,
    DEFAULT_QUESTIONS,
    AgentConfig,
    ConfigSummary,
    RunResult,
    mock_agent_factory,
    render_markdown_table,
    run_benchmark,
)


def test_default_configs_have_at_least_three():
    assert len(DEFAULT_CONFIGS) >= 3


def test_default_configs_are_distinct():
    names = [c.name for c in DEFAULT_CONFIGS]
    assert len(set(names)) == len(names)


def test_run_benchmark_with_mock_returns_one_summary_per_config():
    summaries = run_benchmark(
        configs=DEFAULT_CONFIGS,
        questions=["q1", "q2"],
        agent_factory=mock_agent_factory,
    )
    assert len(summaries) == len(DEFAULT_CONFIGS)
    for s in summaries:
        assert len(s.runs) == 2
        assert s.avg_latency_ms > 0


def test_config_summary_aggregates_correctly():
    cfg = AgentConfig(name="test")
    summary = ConfigSummary(
        config=cfg,
        runs=[
            RunResult(question="q1", answer="ans1", tools_used=["a"], latency_ms=10.0),
            RunResult(
                question="q2", answer="ans22", tools_used=["a", "b"], latency_ms=20.0
            ),
        ],
    )
    assert summary.avg_latency_ms == 15.0
    assert summary.avg_tools_used == 1.5
    assert summary.avg_answer_length == 4.5


def test_render_markdown_table_has_header_and_one_row_per_config():
    summaries = run_benchmark(
        configs=DEFAULT_CONFIGS,
        questions=DEFAULT_QUESTIONS,
        agent_factory=mock_agent_factory,
    )
    table = render_markdown_table(summaries)
    assert "| Config |" in table
    for s in summaries:
        assert f"`{s.config.name}`" in table


def test_to_dict_is_json_serializable():
    summaries = run_benchmark(
        configs=[DEFAULT_CONFIGS[0]],
        questions=["q"],
        agent_factory=mock_agent_factory,
    )
    payload = [s.to_dict() for s in summaries]
    json.dumps(payload)  # não deve levantar


def test_higher_temperature_increases_latency_in_mock():
    cold = AgentConfig(name="cold", temperature=0.0)
    warm = AgentConfig(name="warm", temperature=1.0)
    [s_cold, s_warm] = run_benchmark(
        configs=[cold, warm],
        questions=["q"],
        agent_factory=mock_agent_factory,
    )
    assert s_warm.avg_latency_ms > s_cold.avg_latency_ms
