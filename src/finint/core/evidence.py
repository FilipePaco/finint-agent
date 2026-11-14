"""Per-investigation evidence registry (R3.1, R3.5).

Every tool output is registered here under a sequential id (``ev-001``...). The LLM sees the
bounded ``summary``; the grounding validator and the analyst see the raw ``data`` and its
extracted values. The store is the single source of truth for what the agent knows.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from finint.core.grounding import extract_values
from finint.core.models import Evidence

SUMMARY_MAX_CHARS = 6000


def render(data: Any, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """Compact JSON for the LLM, truncated with an explicit marker (never silently)."""
    text = json.dumps(data, default=str, separators=(",", ":"), ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f'..."[truncated {len(text) - max_chars} chars]"'


class EvidenceStore:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}

    def add(
        self,
        tool: str,
        args: dict[str, Any],
        data: Any,
        *,
        summary: str | None = None,
        untrusted: bool = False,
        injection_hits: list[dict[str, Any]] | None = None,
    ) -> Evidence:
        evidence_id = f"ev-{len(self._items) + 1:03d}"
        evidence = Evidence(
            evidence_id=evidence_id,
            tool=tool,
            args=args,
            summary=summary if summary is not None else render(data),
            # JSON-safe copy (Decimal/datetime -> str) so run state can be checkpointed;
            # values for grounding are extracted from the typed original first.
            data=json.loads(json.dumps(data, default=str)),
            values=extract_values(data),
            untrusted=untrusted,
            injection_hits=injection_hits or [],
        )
        self._items[evidence_id] = evidence
        return evidence

    @classmethod
    def from_dict(cls, items: dict[str, Evidence]) -> EvidenceStore:
        store = cls()
        store._items = dict(items)
        return store

    def get(self, evidence_id: str) -> Evidence:
        return self._items[evidence_id]

    def as_dict(self) -> dict[str, Evidence]:
        return dict(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._items.values())

    def injection_attempts(self) -> int:
        return sum(len(e.injection_hits) for e in self._items.values())
