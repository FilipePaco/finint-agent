"""Column-level data classification for LLM context (R1.7 · enforced by R3.9).

The build fails on any unclassified column, so new data cannot reach a prompt by accident.
Spec 03's context builder reads :class:`DataClassification` to decide what an LLM sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast, get_args

import duckdb

Level = Literal["allowed", "untrusted", "masked", "forbidden"]
_LEVELS: frozenset[str] = frozenset(get_args(Level))
_ALL = "_all"


class UnclassifiedColumnsError(ValueError):
    """Raised when the database has columns that the classification does not cover."""


class DataClassification:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._tables: dict[str, dict[str, str]] = raw["tables"]
        for table, columns in self._tables.items():
            for column, level in columns.items():
                if level not in _LEVELS:
                    raise ValueError(f"{table}.{column}: unknown level {level!r}")

    def level(self, table: str, column: str) -> Level | None:
        columns = self._tables.get(table)
        if columns is None:
            return None
        value = columns.get(column, columns.get(_ALL))
        return value if value is None else _as_level(value)

    def llm_visible(self, table: str) -> list[str]:
        """Columns of ``table`` that may reach an LLM in some form."""
        columns = self._tables.get(table, {})
        return [c for c, level in columns.items() if c != _ALL and level != "forbidden"]

    def unclassified(self, db_path: Path) -> list[str]:
        with duckdb.connect(str(db_path), read_only=True) as con:
            rows = con.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
            ).fetchall()
        return [f"{t}.{c}" for t, c in rows if self.level(t, c) is None]

    def enforce(self, db_path: Path) -> None:
        missing = self.unclassified(db_path)
        if missing:
            raise UnclassifiedColumnsError(
                "Columns missing from config/data_classification.yaml: " + ", ".join(missing)
            )


def _as_level(value: str) -> Level:
    if value not in _LEVELS:
        raise ValueError(f"unknown level {value!r}")
    return cast(Level, value)
