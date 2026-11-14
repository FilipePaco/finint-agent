"""Sanctions screening tool (task 3.8 · R3.4): the deterministic screener over OFAC data.

The full date of birth is used by the screener in code; the LLM only sees its year.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

from finint.core.models import Evidence
from finint.core.screening import EntityKind, SanctionsScreener, SdnRecord
from finint.tools.investigation import ToolContext, ToolError

PERSON_TYPES = ("Individual", "Sole Proprietorship")


@lru_cache(maxsize=2)
def load_screener(db_path: Path, config_items: tuple[tuple[str, Any], ...]) -> SanctionsScreener:
    with duckdb.connect(str(db_path), read_only=True) as con:
        aliases: dict[int, list[str]] = defaultdict(list)
        for ent, name in con.execute("SELECT ent_num, alt_name FROM sdn_aliases").fetchall():
            aliases[ent].append(name)
        countries: dict[int, list[str]] = defaultdict(list)
        for ent, country in con.execute(
            "SELECT ent_num, country FROM sdn_addresses WHERE country IS NOT NULL"
        ).fetchall():
            countries[ent].append(country)
        rows = con.execute(
            "SELECT ent_num, name, sdn_type, programs, dob_text, nationality FROM sdn_entries"
        ).fetchall()
    records = [
        SdnRecord(
            ent, name, sdn_type, programs, dob, nat, tuple(countries[ent]), tuple(aliases[ent])
        )
        for ent, name, sdn_type, programs, dob, nat in rows
    ]
    return SanctionsScreener(records, thaw(config_items))


def screen_party(ctx: ToolContext, account_key: str, config: Mapping[str, Any]) -> Evidence:
    """Screen the owner of an account against the OFAC SDN list (status decided by code)."""
    with ctx.connect() as con:
        row = con.execute(
            "SELECT e.entity_id, e.entity_type, e.synthetic_name, e.country, e.date_of_birth "
            "FROM accounts a JOIN entities e USING (entity_id) WHERE a.account_key = ?",
            [account_key],
        ).fetchone()
    if row is None:
        raise ToolError(f"unknown account {account_key}")
    entity_id, entity_type, name, country, dob = row
    kind: EntityKind = "person" if entity_type in PERSON_TYPES else "company"
    frozen = tuple(sorted((k, _freeze(v)) for k, v in config.items()))
    result = load_screener(ctx.db_path, frozen).screen(name, kind, country, dob)
    data = {
        "account_key": account_key,
        "entity_id": entity_id,
        "screened_name": name,
        "screened_kind": kind,
        "screened_country": country,
        "screened_birth_year": dob.year if dob else None,
        "status": result.status,
        "status_decided_by": "deterministic screener (thresholds in config/screening.yaml)",
        "candidates": [c.model_dump() for c in result.candidates],
    }
    return ctx.store.add("screen_sanctions", {"account_key": account_key}, data, untrusted=True)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def thaw(config_items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    """Inverse of ``_freeze`` for the one level of nesting the screening config has."""
    out: dict[str, Any] = {}
    for key, value in config_items:
        nested = isinstance(value, tuple) and value and isinstance(value[0], tuple)
        out[key] = dict(value) if nested else value
    return out
