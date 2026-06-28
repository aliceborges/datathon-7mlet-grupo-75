"""Schema e loader do golden set usado na avaliação offline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "golden_set" / "datathon_v1.jsonl"
)


@dataclass(frozen=True)
class GoldenSample:
    id: str
    query: str
    expected_answer: str
    contexts: list[str] = field(default_factory=list)
    category: str = "uncategorized"
    expected_tools: list[str] = field(default_factory=list)


def load_golden_set(path: Path | None = None) -> list[GoldenSample]:
    """Lê o JSONL e devolve a lista de GoldenSample."""
    target = path or DEFAULT_PATH
    samples: list[GoldenSample] = []
    with target.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                GoldenSample(
                    id=row["id"],
                    query=row["query"],
                    expected_answer=row["expected_answer"],
                    contexts=row.get("contexts", []),
                    category=row.get("category", "uncategorized"),
                    expected_tools=row.get("expected_tools", []),
                )
            )
    return samples


def count_by_category(samples: list[GoldenSample]) -> dict[str, int]:
    """Conta amostras por categoria — útil pra checar cobertura."""
    out: dict[str, int] = {}
    for s in samples:
        out[s.category] = out.get(s.category, 0) + 1
    return out
