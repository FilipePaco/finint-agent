"""Data minimization for LLM context (task 3.3 · R3.9 · constitution V).

Every row that reaches a prompt goes through :func:`minimize_row` with the levels from
``config/data_classification.yaml``: forbidden columns are dropped, masked columns are
reduced, untrusted columns are separated so the prompt builder wraps them as data.
Unknown columns are dropped (default deny).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

UNTRUSTED_OPEN = "<untrusted_data>"
UNTRUSTED_CLOSE = "</untrusted_data>"


class MinimizedRow(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    untrusted: dict[str, Any] = Field(default_factory=dict)


def mask(column: str, value: Any) -> Any:
    """Reduced form of a masked column: dates of birth become the year."""
    if isinstance(value, date | datetime):
        return value.year
    if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    raise ValueError(f"no masking rule for column {column!r}")


def minimize_row(row: Mapping[str, Any], levels: Mapping[str, str]) -> MinimizedRow:
    out = MinimizedRow()
    for column, value in row.items():
        level = levels.get(column, levels.get("_all", "forbidden"))
        if level == "allowed":
            out.data[column] = value
        elif level == "untrusted":
            out.untrusted[column] = value
        elif level == "masked" and value is not None:
            out.data[column] = mask(column, value)
    return out


def wrap_untrusted(text: str) -> str:
    """Delimit attacker-controllable text. Delimiters inside the text are neutralized."""
    cleaned = text.replace(UNTRUSTED_OPEN, "[removed tag]").replace(
        UNTRUSTED_CLOSE, "[removed tag]"
    )
    return f"{UNTRUSTED_OPEN}{cleaned}{UNTRUSTED_CLOSE}"
