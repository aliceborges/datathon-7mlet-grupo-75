"""Persistência de log auditável das decisões servidas pela API."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from src.api.schemas import AuditLog

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_PATH = Path("logs/audit.jsonl")


class AuditLogger:
    """Escreve uma entrada JSONL por decisão; thread-safe."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_AUDIT_PATH
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: AuditLog) -> None:
        line = entry.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)


def build_audit_logger() -> AuditLogger:
    """Factory que respeita a env var AUDIT_LOG_PATH."""
    path = os.environ.get("AUDIT_LOG_PATH")
    return AuditLogger(path=Path(path) if path else None)
