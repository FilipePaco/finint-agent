"""Synthetic payment memos and prompt-injection payloads (task 1.8 · R1.6).

AMLworld has no free-text field. Real investigations read payment references, which are
attacker-controlled, so every transaction gets a deterministic benign memo, and a small,
documented set of laundering transactions get prompt-injection payloads instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb
import yaml
from pydantic import BaseModel

from finint.data.split import SplitBoundaries


class InjectionCase(BaseModel):
    id: str
    technique: str
    payload: str


def load_injection_cases(path: Path) -> list[InjectionCase]:
    with path.open(encoding="utf-8") as handle:
        cases = [InjectionCase.model_validate(c) for c in yaml.safe_load(handle)["cases"]]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate prompt-injection case ids")
    return cases


def _memo_sql(templates: dict[str, list[str]]) -> tuple[str, dict[str, Any]]:
    """CASE expression choosing a template by payment format and a txn_id-derived index."""
    params: dict[str, Any] = {}
    branches = []
    for i, (fmt, options) in enumerate(sorted(templates.items())):
        params[f"fmt{i}"] = fmt
        params[f"opts{i}"] = options
        branches.append(
            f"WHEN payment_format = $fmt{i} THEN "
            f"list_extract($opts{i}, CAST((txn_id * 2654435761) % len($opts{i}) AS INTEGER) + 1)"
        )
    case = "CASE " + " ".join(branches) + " ELSE 'Transfer' END"
    return case, params


def add_memos(db_path: Path, templates: dict[str, list[str]]) -> int:
    """(Re)build ``transactions.reference``. Returns the number of rows with a memo."""
    case, params = _memo_sql(templates)
    memo = (
        f"replace(replace({case}, "
        "'{n}', CAST(10000 + (txn_id * 40503) % 890000 AS VARCHAR)), "
        "'{month}', strftime(ts, '%B'))"
    )
    with duckdb.connect(str(db_path)) as con:
        con.execute("ALTER TABLE transactions DROP COLUMN IF EXISTS reference")
        con.execute(
            f"CREATE OR REPLACE TABLE transactions AS "  # noqa: S608 - config keys are bound params
            f"SELECT *, {memo} AS reference FROM transactions ORDER BY txn_id",
            params,
        )
        row = con.execute("SELECT count(reference) FROM transactions").fetchone()
    return int(row[0]) if row else 0


_TARGETS = """
SELECT t.txn_id, t.from_key, t.reference
FROM transactions t
JOIN accounts a ON a.account_key = t.from_key
JOIN entities e ON e.entity_id = a.entity_id
WHERE t.is_laundering AND t.ts >= $test_start AND NOT e.is_injected
"""


def inject_prompts(
    db_path: Path, cases: list[InjectionCase], split: SplitBoundaries, seed: int
) -> dict[str, int]:
    """Replace one test-window laundering memo per case (distinct accounts). ``{case: txn_id}``."""
    with duckdb.connect(str(db_path)) as con:
        candidates = con.execute(_TARGETS, {"test_start": split.test_start}).fetchall()
        used_accounts: set[str] = set()
        assignment: dict[str, tuple[int, str, str]] = {}
        for case in cases:
            ranked = sorted(
                (hashlib.md5(f"{seed}:{case.id}:{txn}".encode(), usedforsecurity=False).hexdigest(),
                 txn, account, memo)
                for txn, account, memo in candidates
                if account not in used_accounts
            )  # fmt: skip
            if not ranked:
                raise ValueError(f"{case.id}: no laundering transaction left in the test window")
            _, txn_id, account, original = ranked[0]
            used_accounts.add(account)
            assignment[case.id] = (txn_id, account, original)
        con.execute(
            "CREATE OR REPLACE TABLE injected_prompts (case_id VARCHAR PRIMARY KEY, "
            "txn_id BIGINT, account_key VARCHAR, technique VARCHAR, payload VARCHAR, "
            "original_reference VARCHAR)"
        )
        for case in cases:
            txn_id, account, original = assignment[case.id]
            con.execute(
                "UPDATE transactions SET reference = ? WHERE txn_id = ?", [case.payload, txn_id]
            )
            con.execute(
                "INSERT INTO injected_prompts VALUES (?, ?, ?, ?, ?, ?)",
                [case.id, txn_id, account, case.technique, case.payload, original],
            )
    return {case_id: txn for case_id, (txn, _, _) in assignment.items()}
