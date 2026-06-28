from __future__ import annotations

import json

import pytest

from src.agent.tools import (
    MOCK_CUSTOMER_DB,
    StaticMetricsClient,
    StubRAGRetriever,
    build_default_tools,
)


def test_default_tools_count():
    tools = build_default_tools()
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {
        "get_customer_context",
        "lookup_offer_details",
        "historical_performance",
    }


class TestGetCustomerContext:
    def setup_method(self):
        self.tool = next(
            t for t in build_default_tools() if t.name == "get_customer_context"
        )

    def test_returns_existing_customer(self):
        out = self.tool.func("C001")
        data = json.loads(out)
        assert data["customer_id"] == "C001"
        assert data["age"] == MOCK_CUSTOMER_DB["C001"]["age"]

    def test_strips_quotes_from_input(self):
        out = self.tool.func('"C002"')
        data = json.loads(out)
        assert data["customer_id"] == "C002"

    def test_unknown_customer_returns_error(self):
        out = self.tool.func("UNKNOWN")
        data = json.loads(out)
        assert "error" in data


class TestLookupOfferDetails:
    def setup_method(self):
        self.tool = next(
            t for t in build_default_tools() if t.name == "lookup_offer_details"
        )

    def test_finds_matching_offer(self):
        out = self.tool.func("loan_payroll")
        assert "loan_payroll" in out or "consignado" in out.lower()

    def test_returns_fallback_for_no_match(self):
        out = self.tool.func("zzz-nada-aqui-zzz")
        assert out  # retorna alguma coisa, não vazio


class TestHistoricalPerformance:
    def setup_method(self):
        self.tool = next(
            t for t in build_default_tools() if t.name == "historical_performance"
        )

    def test_returns_metrics_for_known_offer(self):
        out = self.tool.func("loan_payroll")
        data = json.loads(out)
        assert set(data.keys()) == {"ctr", "conversion", "n"}
        assert data["ctr"] > 0

    def test_returns_error_for_unknown_offer(self):
        out = self.tool.func("offer_inexistente")
        data = json.loads(out)
        assert "error" in data


class TestStaticMetricsClient:
    def test_offer_performance_returns_copy(self):
        client = StaticMetricsClient()
        data = client.offer_performance("loan_personal")
        data["ctr"] = 99.0
        # garante que mutação não afeta o estado interno
        assert client.offer_performance("loan_personal")["ctr"] != 99.0

    def test_raises_for_unknown(self):
        client = StaticMetricsClient()
        with pytest.raises(KeyError):
            client.offer_performance("zzz")


class TestStubRAGRetriever:
    def test_retrieve_returns_at_most_k(self):
        r = StubRAGRetriever()
        out = r.retrieve("loan", k=2)
        assert len(out) <= 2

    def test_retrieve_falls_back_when_no_match(self):
        r = StubRAGRetriever()
        out = r.retrieve("xxxxx", k=3)
        assert len(out) == 1
