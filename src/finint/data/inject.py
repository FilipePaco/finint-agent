"""Inject documented test cases into the synthetic data (tasks 1.7, 1.8 · R1.5, R1.6).

Sanctions cases rename one synthetic entity each, chosen deterministically among entities
active in the **test window**: ``subject`` cases go to entities that send laundering
transactions there, and ``counterparty`` cases go to clean counterparties of those
entities. The assignment is written to ``injected_sanctions`` so the evaluation (spec 04)
knows the ground truth.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Literal, Self

import duckdb
import yaml
from pydantic import BaseModel, model_validator

from finint.data.split import SplitBoundaries

PERSON_TYPES = ("Individual", "Sole Proprietorship")
COMPANY_TYPES = ("Corporation", "Partnership")

Variant = Literal[
    "transliteration",
    "missing_middle_name",
    "swapped_order",
    "diacritics_spacing",
    "alias",
    "legal_form",
    "exact_name_decoy",
    "similar_name_decoy",
]


class EntityCase(BaseModel):
    """A documented test case that renames one synthetic entity."""

    id: str
    kind: Literal["person", "company"]
    role: Literal["subject", "counterparty"]
    name: str
    country: str
    country_iso2: str
    dob: date | None = None

    @model_validator(mode="after")
    def _dob_matches_kind(self) -> Self:
        if (self.kind == "person") != (self.dob is not None):
            raise ValueError(f"{self.id}: persons need a dob, companies must not have one")
        return self


class SanctionsCase(EntityCase):
    sdn_ent_num: int
    variant: Variant
    expected: Literal["TRUE_MATCH", "FALSE_MATCH"]


class MediaCase(EntityCase):
    expected_media: Literal[
        "ALLEGATION", "CHARGE", "CONVICTION", "SANCTION", "SAME_NAME_DIFFERENT_ENTITY", "NONE"
    ]


def _load_cases[C: EntityCase](path: Path, model: type[C]) -> list[C]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cases = [model.model_validate(item) for item in raw["cases"]]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError(f"duplicate case ids in {path.name}")
    return cases


def load_sanctions_cases(path: Path) -> list[SanctionsCase]:
    return _load_cases(path, SanctionsCase)


def load_media_cases(path: Path) -> list[MediaCase]:
    return _load_cases(path, MediaCase)


_SUBJECTS = """
SELECT DISTINCT a.entity_id
FROM transactions t
JOIN accounts a ON a.account_key = t.from_key
JOIN entities e ON e.entity_id = a.entity_id
WHERE t.is_laundering AND t.ts >= $test_start AND list_contains($types, e.entity_type)
"""

_COUNTERPARTIES = """
WITH launderers AS (
    SELECT DISTINCT from_key AS k FROM transactions WHERE is_laundering AND ts >= $test_start
),
dirty AS (
    SELECT DISTINCT a.entity_id FROM transactions t JOIN accounts a ON a.account_key = t.from_key
    WHERE t.is_laundering
    UNION
    SELECT DISTINCT a.entity_id FROM transactions t JOIN accounts a ON a.account_key = t.to_key
    WHERE t.is_laundering
)
SELECT DISTINCT a.entity_id
FROM transactions t
JOIN launderers l ON l.k = t.from_key
JOIN accounts a ON a.account_key = t.to_key
JOIN entities e ON e.entity_id = a.entity_id
WHERE t.ts >= $test_start AND NOT t.is_laundering
  AND list_contains($types, e.entity_type)
  AND a.entity_id NOT IN (SELECT entity_id FROM dirty)
"""


def _rank_key(entity_id: str, case_id: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{case_id}:{entity_id}".encode(), usedforsecurity=False).hexdigest()


def assign_entities(
    con: duckdb.DuckDBPyConnection,
    cases: list[SanctionsCase] | list[MediaCase],
    split: SplitBoundaries,
    seed: int,
    exclude: set[str] | None = None,
) -> dict[str, str]:
    """Pick one entity per case, deterministically, never reusing an entity."""
    pools: dict[tuple[str, str], list[str]] = {}
    for role, sql in (("subject", _SUBJECTS), ("counterparty", _COUNTERPARTIES)):
        for kind, types in (("person", PERSON_TYPES), ("company", COMPANY_TYPES)):
            rows = con.execute(sql, {"test_start": split.test_start, "types": list(types)})
            pools[(role, kind)] = [r[0] for r in rows.fetchall()]
    used = set(exclude or ())
    assignment: dict[str, str] = {}
    for case in cases:
        pool = [e for e in pools[(case.role, case.kind)] if e not in used]
        if not pool:
            raise ValueError(f"{case.id}: no candidate entity for {case.role}/{case.kind}")
        chosen = sorted((_rank_key(entity, case.id, seed), entity) for entity in pool)[0][1]
        used.add(chosen)
        assignment[case.id] = chosen
    return assignment


def _rename(con: duckdb.DuckDBPyConnection, case: EntityCase, entity_id: str) -> None:
    con.execute(
        "UPDATE entities SET synthetic_name = ?, country = ?, country_iso2 = ?, "
        "date_of_birth = ?, is_injected = true, injection_case_id = ? WHERE entity_id = ?",
        [case.name, case.country, case.country_iso2, case.dob, case.id, entity_id],
    )


def inject_sanctions(
    db_path: Path, cases: list[SanctionsCase], split: SplitBoundaries, seed: int
) -> dict[str, str]:
    """Assign and rename sanctions-case entities. Returns ``{case_id: entity_id}``."""
    with duckdb.connect(str(db_path)) as con:
        assignment = assign_entities(con, cases, split, seed)
        con.execute(
            "CREATE OR REPLACE TABLE injected_sanctions (case_id VARCHAR PRIMARY KEY, "
            "entity_id VARCHAR, sdn_ent_num INTEGER, variant VARCHAR, expected VARCHAR, "
            "kind VARCHAR, role VARCHAR, injected_name VARCHAR)"
        )
        for case in cases:
            entity_id = assignment[case.id]
            _rename(con, case, entity_id)
            con.execute(
                "INSERT INTO injected_sanctions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    case.id, entity_id, case.sdn_ent_num, case.variant, case.expected,
                    case.kind, case.role, case.name,
                ],
            )  # fmt: skip
    return assignment


def inject_media(
    db_path: Path,
    cases: list[MediaCase],
    split: SplitBoundaries,
    seed: int,
    exclude: set[str],
) -> dict[str, str]:
    """Assign and rename adverse-media test entities, avoiding ``exclude`` (sanctions picks)."""
    with duckdb.connect(str(db_path)) as con:
        assignment = assign_entities(con, cases, split, seed, exclude)
        con.execute(
            "CREATE OR REPLACE TABLE injected_media (case_id VARCHAR PRIMARY KEY, "
            "entity_id VARCHAR, expected_media VARCHAR, kind VARCHAR, role VARCHAR, "
            "injected_name VARCHAR)"
        )
        for case in cases:
            entity_id = assignment[case.id]
            _rename(con, case, entity_id)
            con.execute(
                "INSERT INTO injected_media VALUES (?, ?, ?, ?, ?, ?)",
                [case.id, entity_id, case.expected_media, case.kind, case.role, case.name],
            )
    return assignment
