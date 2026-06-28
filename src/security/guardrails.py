"""Guardrails de input pra mitigar prompt injection e abuso de tamanho."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"forget\s+(everything|all|your\s+instructions|the\s+rules)",
    r"you\s+are\s+now\s+(a|an|in)\s+",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"reveal\s+(the\s+)?(system\s+)?prompt",
    r"act\s+as\s+(if\s+)?(a\s+)?(different|another|new)\s+",
    r"ignor[ae]\s+(toda|todas|tudo|qualquer)(\s+as)?\s+(instruções|regras|instrucoes)",
    r"esque[çc]a\s+(tudo|todas|suas\s+(instruções|instrucoes))",
    r"voc[êe]\b.{0,20}\b(um|uma)\s+(novo|outro|modelo|assistente|sistema|bot|ia)",
)


@dataclass(frozen=True)
class GuardrailResult:
    is_valid: bool
    reason: str | None = None
    matched_pattern: str | None = None

    @classmethod
    def ok(cls) -> GuardrailResult:
        return cls(is_valid=True)

    @classmethod
    def blocked(cls, reason: str, pattern: str | None = None) -> GuardrailResult:
        return cls(is_valid=False, reason=reason, matched_pattern=pattern)


class InputGuardrail:
    """Valida texto livre antes de enviar ao LLM."""

    def __init__(
        self,
        max_length: int = 4096,
        patterns: tuple[str, ...] = INJECTION_PATTERNS,
    ) -> None:
        self.max_length = max_length
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

    def validate(self, text: str) -> GuardrailResult:
        stripped = text.strip()
        if not stripped:
            return GuardrailResult.blocked("input vazio")

        if len(text) > self.max_length:
            return GuardrailResult.blocked(
                f"input excede tamanho máximo ({self.max_length} caracteres)"
            )

        for pattern in self._compiled:
            if pattern.search(text):
                logger.warning(
                    "Guardrail bloqueou padrão: %s (input[:80]=%r)",
                    pattern.pattern,
                    text[:80],
                )
                return GuardrailResult.blocked(
                    "padrão suspeito detectado no input",
                    pattern=pattern.pattern,
                )

        return GuardrailResult.ok()
