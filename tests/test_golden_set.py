from __future__ import annotations

from src.evaluation.offline_golden_set import (
	GOLDEN_SET_PATH,
	load_golden_set_cases,
	render_markdown_report,
	run_offline_evaluation,
)


def test_golden_set_has_minimum_coverage():
	cases = load_golden_set_cases(GOLDEN_SET_PATH)

	assert len(cases) >= 20
	assert any(case.policy_should_not_be_used for case in cases)
	assert any(case.segment == "senior_affluent" for case in cases)
	assert all(case.justification for case in cases)
	assert all(case.pass_criteria for case in cases)


def test_offline_evaluation_is_reproducible():
	first = run_offline_evaluation(seed=75)
	second = run_offline_evaluation(seed=75)

	assert first.to_dict() == second.to_dict()
	assert first.baseline.total_cases == first.adaptive.total_cases
	assert first.baseline.total_cases >= 20


def test_offline_evaluation_reports_fairness_and_sensitivity():
	summary = run_offline_evaluation(seed=75)
	report = render_markdown_report(summary)

	assert summary.baseline.fairness_gap >= 0.0
	assert summary.adaptive.sensitivity_flip_rate >= 0.0
	assert "Offline Golden Set Evaluation" in report
	assert "Out of scope cases" in report