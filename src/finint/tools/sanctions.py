"""Sanctions screening tool (task 3.8 · R3.4): the deterministic screener over OFAC data.

The full date of birth is used by the screener in code; the LLM only sees its year.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb

from finint.core.evidence import AGENT_PREFIX, HARNESS_PREFIX
from finint.core.models import Evidence
from finint.core.screening import (
    CounterpartyRule,
    EntityKind,
    SanctionsScreener,
    SdnRecord,
    SelectedCounterparty,
    select_counterparties,
)
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


def screen_party(
    ctx: ToolContext,
    account_key: str,
    config: Mapping[str, Any],
    *,
    prefix: str = AGENT_PREFIX,
    extra_args: Mapping[str, Any] | None = None,
) -> Evidence:
    """Screen the owner of an account against the OFAC SDN list (status decided by code).

    ``prefix`` and ``extra_args`` let the harness register its own screenings (``cp-`` ids,
    with the reason the counterparty was selected) without shifting the agent's ids.
    """
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
    args = {"account_key": account_key, **(extra_args or {})}
    return ctx.store.add("screen_sanctions", args, data, untrusted=True, prefix=prefix)


# Volume per counterparty of an account in [s, e), both directions. Fixed SQL, bound params.
_COUNTERPARTY_VOLUME_SQL = """
SELECT counterparty, sum(amount_usd) AS total_usd FROM (
    SELECT to_key AS counterparty, amount_usd
    FROM transactions WHERE from_key = $k AND ts >= $s AND ts < $e
    UNION ALL
    SELECT from_key, amount_usd
    FROM transactions WHERE to_key = $k AND ts >= $s AND ts < $e
)
WHERE counterparty <> $k
GROUP BY 1 ORDER BY total_usd DESC, counterparty LIMIT $top
"""


@dataclass(frozen=True)
class CounterpartyScreen:
    selected: SelectedCounterparty
    evidence: Evidence | None
    error: str | None = None


def screen_counterparties(
    ctx: ToolContext, account_key: str, alert_date: date, config: Mapping[str, Any]
) -> list[CounterpartyScreen]:
    """Harness step (R3.4, ADR-03.10): screen the alerted account's relevant counterparties.

    The selection is a code rule (:func:`select_counterparties`, caps in
    ``config/screening.yaml``); each screening is the same deterministic matcher the agent's
    ``screen_sanctions`` tool uses, registered under a ``cp-`` evidence id. Local list
    matching only: nothing leaves the process and no LLM is involved.
    """
    rule = CounterpartyRule.model_validate(config.get("counterparties") or {})
    day = datetime.combine(alert_date, datetime.min.time())
    end = day + timedelta(days=1)
    windows = {
        "alert_day": (day, rule.alert_day_max),
        "window": (day - timedelta(days=rule.lookback_days), rule.top_n),
    }
    volumes: dict[str, list[tuple[str, float]]] = {}
    with ctx.connect() as con:
        for name, (start, top) in windows.items():
            rows = con.execute(
                _COUNTERPARTY_VOLUME_SQL,
                {"k": account_key, "s": start, "e": end, "top": max(1, top)},
            ).fetchall()
            volumes[name] = [(str(k), float(v or 0)) for k, v in rows]
    selected = select_counterparties(account_key, volumes["alert_day"], volumes["window"], rule)
    out = []
    for party in selected:
        try:
            evidence = screen_party(
                ctx,
                party.account_key,
                config,
                prefix=HARNESS_PREFIX,
                extra_args={"selected_by": party.reasons},
            )
        except ToolError as error:
            out.append(CounterpartyScreen(party, None, str(error)))
            continue
        out.append(CounterpartyScreen(party, evidence))
    return out


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
