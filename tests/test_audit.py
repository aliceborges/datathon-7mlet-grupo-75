from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.api.audit import AuditLogger, build_audit_logger
from src.api.schemas import AuditLog


def _sample_entry(decision_id: str = "abc-123") -> AuditLog:
    return AuditLog(
        decision_id=decision_id,
        endpoint="/predict",
        policy_version="stub-thompson-v0",
        customer_id="C001",
        chosen_offer="loan_payroll",
        reason_codes=["senior_segment"],
        created_at=datetime.now(timezone.utc),
    )


def test_audit_logger_creates_file(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)
    logger.write(_sample_entry())
    assert audit_path.exists()


def test_audit_logger_writes_jsonl_line(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=audit_path)
    logger.write(_sample_entry("entry-1"))
    logger.write(_sample_entry("entry-2"))

    lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["decision_id"] == "entry-1"
    assert parsed[1]["decision_id"] == "entry-2"


def test_build_audit_logger_respects_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom" / "trail.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(custom))
    logger = build_audit_logger()
    assert logger.path == Path(str(custom))
    assert logger.path.parent.exists()


def test_build_audit_logger_default(monkeypatch, tmp_path):
    monkeypatch.delenv("AUDIT_LOG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    logger = build_audit_logger()
    assert logger.path.name == "audit.jsonl"
