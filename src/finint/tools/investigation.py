"""Read-only investigation tools over DuckDB (tasks 3.8, 3.9, 2.11 · R3.1, R2.9).

Rules every tool follows:

* read-only connection, SQL fixed at module level with bound parameters, bounded windows
  (``MAX_WINDOW_DAYS``) and rows;
* output passes through :func:`finint.core.minimize.minimize_row` with the column levels
  from ``config/data_classification.yaml`` (labels and ground truth never leave);
* untrusted text (memos, names) is wrapped as data and scanned for injection attempts;
* the result is registered in the run's :class:`EvidenceStore` and returned as Evidence.

Totals are computed here, so the agent cites an aggregate instead of adding numbers itself
(the grounding validator rejects any amount it cannot find in evidence).
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import duckdb

from finint.config import PROJECT_ROOT, load_yaml
from finint.core.evidence import EvidenceStore
from finint.core.injection import scan
from finint.core.minimize import minimize_row, wrap_untrusted
from finint.core.models import Evidence
from finint.core.typology_signals import Rule, load_rules, typology_candidates

MAX_WINDOW_DAYS = 14
MAX_ROWS = 50
MAX_HOPS = 3

Direction = Literal["in", "out", "both"]

# One literal SQL string per direction: nothing is assembled at runtime.
_TOTALS_SQL = {
    "out": """
SELECT count(*) AS n_transactions, sum(amount_usd) AS total_amount_usd,
       sum(amount_usd) FILTER (WHERE from_key = $k) AS total_out_usd,
       sum(amount_usd) FILTER (WHERE to_key = $k) AS total_in_usd,
       min(ts) AS first_ts, max(ts) AS last_ts
FROM transactions WHERE from_key = $k AND ts >= $s AND ts < $e
""",
    "in": """
SELECT count(*) AS n_transactions, sum(amount_usd) AS total_amount_usd,
       sum(amount_usd) FILTER (WHERE from_key = $k) AS total_out_usd,
       sum(amount_usd) FILTER (WHERE to_key = $k) AS total_in_usd,
       min(ts) AS first_ts, max(ts) AS last_ts
FROM transactions WHERE to_key = $k AND ts >= $s AND ts < $e
""",
    "both": """
SELECT count(*) AS n_transactions, sum(amount_usd) AS total_amount_usd,
       sum(amount_usd) FILTER (WHERE from_key = $k) AS total_out_usd,
       sum(amount_usd) FILTER (WHERE to_key = $k) AS total_in_usd,
       min(ts) AS first_ts, max(ts) AS last_ts
FROM transactions WHERE (from_key = $k OR to_key = $k) AND ts >= $s AND ts < $e
""",
}
_ROWS_SQL = {
    "out": """
SELECT txn_id, ts, from_key, to_key, amount_paid, currency_paid, amount_received,
       currency_received, amount_usd, payment_format, cross_currency, cross_bank, reference
FROM transactions WHERE from_key = $k AND ts >= $s AND ts < $e
ORDER BY amount_usd DESC LIMIT $limit
""",
    "in": """
SELECT txn_id, ts, from_key, to_key, amount_paid, currency_paid, amount_received,
       currency_received, amount_usd, payment_format, cross_currency, cross_bank, reference
FROM transactions WHERE to_key = $k AND ts >= $s AND ts < $e
ORDER BY amount_usd DESC LIMIT $limit
""",
    "both": """
SELECT txn_id, ts, from_key, to_key, amount_paid, currency_paid, amount_received,
       currency_received, amount_usd, payment_format, cross_currency, cross_bank, reference
FROM transactions WHERE (from_key = $k OR to_key = $k) AND ts >= $s AND ts < $e
ORDER BY amount_usd DESC LIMIT $limit
""",
}
_SIDE = set(_ROWS_SQL)
_ALERT_TXNS_SQL = """
SELECT txn_id, ts, from_key, to_key, amount_paid, currency_paid, amount_received,
       currency_received, amount_usd, payment_format, cross_currency, cross_bank, reference
FROM transactions WHERE list_contains($ids, txn_id) ORDER BY ts
"""

_PROFILE_SQL = """
SELECT a.account_key, a.bank_name, e.entity_id, e.entity_type, e.country,
       e.synthetic_name, e.date_of_birth, e.n_accounts
FROM accounts a JOIN entities e USING (entity_id)
WHERE a.account_key = $k
"""
_SIBLINGS_SQL = """
SELECT account_key, bank_name FROM accounts
WHERE entity_id = $entity AND account_key <> $k
ORDER BY account_key LIMIT 20
"""
_ACTIVITY_SQL = """
SELECT min(ts) AS first_seen, max(ts) AS last_seen,
       count(*) FILTER (WHERE from_key = $k) AS n_out,
       count(*) FILTER (WHERE to_key = $k) AS n_in
FROM transactions WHERE from_key = $k OR to_key = $k
"""

_COUNTERPARTIES_SQL = """
SELECT counterparty, direction, count(*) AS n_transactions,
       sum(amount_usd) AS total_amount_usd, min(ts) AS first_ts, max(ts) AS last_ts
FROM (
    SELECT to_key AS counterparty, 'out' AS direction, amount_usd, ts
    FROM transactions WHERE from_key = $k AND ts >= $s AND ts < $e
    UNION ALL
    SELECT from_key, 'in', amount_usd, ts
    FROM transactions WHERE to_key = $k AND ts >= $s AND ts < $e
)
WHERE counterparty <> $k
GROUP BY ALL ORDER BY total_amount_usd DESC LIMIT $top
"""

_FLOW_OUT_SQL = """
SELECT to_key AS other, count(*) AS n, sum(amount_usd) AS amount_usd, min(ts) AS t
FROM transactions
WHERE from_key = $node AND ts >= $t AND ts < $e AND to_key <> $node
GROUP BY 1 ORDER BY amount_usd DESC LIMIT 5
"""
_FLOW_IN_SQL = """
SELECT from_key AS other, count(*) AS n, sum(amount_usd) AS amount_usd, max(ts) AS t
FROM transactions
WHERE to_key = $node AND ts <= $t AND ts >= $s AND from_key <> $node
GROUP BY 1 ORDER BY amount_usd DESC LIMIT 5
"""

_FAN_SQL = """
SELECT count(DISTINCT to_key) FILTER (WHERE from_key = $k) AS distinct_receivers,
       count(DISTINCT from_key) FILTER (WHERE to_key = $k) AS distinct_senders,
       count(*) FILTER (WHERE from_key = $k) AS n_out,
       count(*) FILTER (WHERE to_key = $k) AS n_in,
       sum(amount_usd) FILTER (WHERE from_key = $k) AS total_out_usd,
       sum(amount_usd) FILTER (WHERE to_key = $k) AS total_in_usd
FROM transactions
WHERE (from_key = $k OR to_key = $k) AND ts >= $s AND ts < $e AND from_key <> to_key
"""
_BURST_SQL = """
SELECT coalesce(max(n), 0) AS max_distinct_receivers_per_day FROM (
    SELECT count(DISTINCT to_key) AS n FROM transactions
    WHERE from_key = $k AND ts >= $s AND ts < $e AND to_key <> $k
    GROUP BY date_trunc('day', ts)
)
"""
# Edges within two hops of the account (plus edges back into it) are enough to find
# directed cycles of length <= 4 through the account.
_CYCLES_SQL = """
WITH w AS (
    SELECT DISTINCT from_key AS a, to_key AS b FROM transactions
    WHERE ts >= $s AND ts < $e AND from_key <> to_key
),
hop1 AS (SELECT b AS n FROM w WHERE a = $k),
hop2 AS (SELECT w.b AS n FROM w JOIN hop1 ON w.a = hop1.n),
e AS (
    SELECT * FROM w
    WHERE a = $k OR b = $k OR a IN (SELECT n FROM hop1) OR a IN (SELECT n FROM hop2)
)
SELECT
  (SELECT count(*) FROM e e1 JOIN e e2 ON e1.b = e2.a AND e2.b = e1.a
   WHERE e1.a = $k) AS cycles_len2,
  (SELECT count(*) FROM e e1 JOIN e e2 ON e1.b = e2.a
   JOIN e e3 ON e2.b = e3.a AND e3.b = e1.a
   WHERE e1.a = $k AND e2.b <> $k) AS cycles_len3,
  (SELECT count(*) FROM e e1 JOIN e e2 ON e1.b = e2.a JOIN e e3 ON e2.b = e3.a
   JOIN e e4 ON e3.b = e4.a AND e4.b = e1.a
   WHERE e1.a = $k AND e2.b <> $k AND e3.b <> $k AND e1.b <> e3.b) AS cycles_len4
"""
_FLOW_THROUGH_SQL = """
SELECT count(*) AS n_out_within_24h_of_inflow,
       sum(o.amount_usd) AS amount_out_within_24h_of_inflow_usd
FROM transactions o
WHERE o.from_key = $k AND o.ts >= $s AND o.ts < $e AND EXISTS (
    SELECT 1 FROM transactions i
    WHERE i.to_key = $k AND i.ts <= o.ts AND i.ts > o.ts - INTERVAL 1 DAY
      AND i.from_key <> $k
)
"""
_STRUCTURING_SQL = """
SELECT count(*) FILTER (WHERE amount_usd >= 9000 AND amount_usd < 10000) AS n_just_under_10k,
       count(*) FILTER (WHERE amount_paid % 1000 = 0) AS n_round_thousands,
       count(*) AS n_out
FROM transactions WHERE from_key = $k AND ts >= $s AND ts < $e
"""
_FORMATS_SQL = """
SELECT payment_format, count(*) AS n FROM transactions
WHERE from_key = $k AND ts >= $s AND ts < $e GROUP BY 1 ORDER BY 2 DESC, 1
"""
# Multi-account (two-hop) indicators, R3.15 / ADR-03.11. One literal query; `w` is the window.
# * scatter-gather, account as source: the account's receivers that pay one same destination;
# * scatter-gather, account as intermediary: accounts that, like this one, receive from one
#   same source and pay one same destination (this account included);
# * gather-scatter feeder: a receiver of the account that itself has many senders and receivers;
# * fan-in feeder: the receiver of the account with the most distinct senders.
_MULTI_SQL = """
WITH w AS (
    SELECT from_key AS a, to_key AS b, amount_usd AS amt FROM transactions
    WHERE ts >= $s AND ts < $e AND from_key <> to_key
),
kout AS (SELECT b, sum(amt) AS amt FROM w WHERE a = $k GROUP BY 1),
kin AS (SELECT DISTINCT a FROM w WHERE b = $k),
r_in AS (
    SELECT w.b AS h, count(DISTINCT w.a) AS senders FROM w JOIN kout ON w.b = kout.b GROUP BY 1
),
r_out AS (
    SELECT w.a AS h, count(DISTINCT w.b) AS receivers FROM w JOIN kout ON w.a = kout.b
    WHERE w.b <> $k GROUP BY 1
),
r AS (
    SELECT r_in.h, r_in.senders, coalesce(r_out.receivers, 0) AS receivers
    FROM r_in LEFT JOIN r_out USING (h)
),
cm AS (
    SELECT DISTINCT w.b AS dest, kout.b AS mid, kout.amt FROM kout JOIN w ON w.a = kout.b
    WHERE w.b <> $k AND w.b <> kout.b
),
conv AS (SELECT dest, count(*) AS n, sum(amt) AS amt FROM cm GROUP BY 1),
sj AS (SELECT DISTINCT w.a AS s, w.b AS j FROM w JOIN kin ON w.a = kin.a WHERE w.b <> $k),
jd AS (SELECT DISTINCT w.a AS j, w.b AS d FROM w JOIN kout ON w.b = kout.b WHERE w.a <> $k),
par AS (
    SELECT sj.s, jd.d, count(*) + 1 AS n FROM sj JOIN jd ON sj.j = jd.j
    WHERE sj.s <> jd.d AND sj.s <> $k AND jd.d <> $k GROUP BY 1, 2
),
top_conv AS (SELECT * FROM conv ORDER BY n DESC, amt DESC, dest LIMIT 1),
top_par AS (SELECT * FROM par ORDER BY n DESC, s, d LIMIT 1),
top_hub AS (
    SELECT h, least(senders, receivers) AS degree FROM r ORDER BY degree DESC, h LIMIT 1
),
top_fan_in AS (SELECT h, senders FROM r ORDER BY senders DESC, h LIMIT 1)
SELECT
  (SELECT sum(amt) FROM kout) AS total_out_usd,
  (SELECT least(count(*), (SELECT count(*) FROM kin)) FROM kout) AS own_hub_degree,
  (SELECT n FROM top_conv) AS src_n,
  (SELECT dest FROM top_conv) AS src_dest,
  (SELECT amt FROM top_conv) AS src_amt,
  (SELECT n FROM top_par) AS mid_n,
  (SELECT s FROM top_par) AS mid_source,
  (SELECT d FROM top_par) AS mid_dest,
  (SELECT degree FROM top_hub) AS hub_degree,
  (SELECT h FROM top_hub) AS hub,
  (SELECT senders FROM top_fan_in) AS fan_in_senders,
  (SELECT h FROM top_fan_in) AS fan_in_receiver
"""
# One level of an amount-matched chain (layering, long cycles): from each frontier node, the
# transfers that leave it after the money arrived (within CHAIN_MAX_DELAY) for a similar amount.
_CHAIN_FWD_SQL = """
WITH f AS (SELECT unnest($nodes) AS node, unnest($amts) AS amt, unnest($times) AS t)
SELECT x.to_key AS node, x.amount_usd::DOUBLE AS amt, x.ts AS t
FROM f JOIN transactions x ON x.from_key = f.node
WHERE x.ts > f.t AND x.ts <= f.t + to_hours($delay) AND x.ts >= $s AND x.ts < $e
  AND x.to_key <> x.from_key
  AND x.amount_usd BETWEEN $lo * f.amt AND $hi * f.amt
ORDER BY x.ts, node, amt LIMIT $cap
"""
_CHAIN_BWD_SQL = """
WITH f AS (SELECT unnest($nodes) AS node, unnest($amts) AS amt, unnest($times) AS t)
SELECT x.from_key AS node, x.amount_usd::DOUBLE AS amt, x.ts AS t
FROM f JOIN transactions x ON x.to_key = f.node
WHERE x.ts < f.t AND x.ts >= f.t - to_hours($delay) AND x.ts >= $s AND x.ts < $e
  AND x.to_key <> x.from_key
  AND f.amt BETWEEN $lo * x.amount_usd AND $hi * x.amount_usd
ORDER BY x.ts DESC, node, amt LIMIT $cap
"""
_CHAIN_START_SQL = {
    "out": """
SELECT to_key AS node, amount_usd::DOUBLE AS amt, ts AS t FROM transactions
WHERE from_key = $k AND ts >= $s AND ts < $e AND to_key <> from_key
ORDER BY amount_usd DESC, ts, node LIMIT $cap
""",
    "in": """
SELECT from_key AS node, amount_usd::DOUBLE AS amt, ts AS t FROM transactions
WHERE to_key = $k AND ts >= $s AND ts < $e AND to_key <> from_key
ORDER BY amount_usd DESC, ts, node LIMIT $cap
""",
}
CHAIN_MAX_DELAY_HOURS = 72
CHAIN_AMOUNT_BAND = (0.8, 1.2)  # next hop amount / previous hop amount
CHAIN_FRONTIER = 25
CHAIN_MAX_DEPTH = {"out": 10, "in": 5}


@lru_cache(maxsize=1)
def default_rules() -> tuple[Rule, ...]:
    return tuple(load_rules(load_yaml("typology", PROJECT_ROOT / "config")))


@dataclass
class ToolContext:
    db_path: Path
    classification: Mapping[str, Mapping[str, str]]
    store: EvidenceStore = field(default_factory=EvidenceStore)
    # Evaluation only: replace stored memos by txn_id (paired prompt-injection runs, R4.2).
    memo_overrides: Mapping[int, str] = field(default_factory=dict)
    # Multi-account indicators and typology candidates in detect_patterns (R3.15); the v1
    # baseline turns them off to stay comparable with its recorded results.
    multi_account: bool = True
    typology_rules: Sequence[Rule] | None = None  # default: config/typology.yaml

    def candidate_rules(self) -> Sequence[Rule]:
        return self.typology_rules if self.typology_rules is not None else default_rules()

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=True)

    def levels(self, table: str) -> Mapping[str, str]:
        return self.classification.get(table, {})


class ToolError(ValueError):
    """A bad tool argument. The message goes back to the agent so it can correct itself."""


def _window(start: str | date, end: str | date) -> tuple[datetime, datetime]:
    try:
        s = date.fromisoformat(str(start)[:10])
        e = date.fromisoformat(str(end)[:10])
    except ValueError as error:
        raise ToolError(f"dates must be ISO YYYY-MM-DD: {error}") from error
    if e < s:
        raise ToolError("end date is before start date")
    if (e - s).days > MAX_WINDOW_DAYS:
        raise ToolError(f"window is limited to {MAX_WINDOW_DAYS} days")
    first = datetime.combine(s, datetime.min.time())
    return first, datetime.combine(e, datetime.min.time()) + timedelta(days=1)


def _span(s: datetime, e: datetime) -> list[date]:
    return [s.date(), (e - timedelta(days=1)).date()]


def _rows(
    con: duckdb.DuckDBPyConnection, sql: str, params: Mapping[str, Any]
) -> list[dict[str, Any]]:
    cursor = con.execute(sql, dict(params))
    names = [d[0] for d in cursor.description or []]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _apply_overrides(ctx: ToolContext, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ctx.memo_overrides:
        return rows
    return [
        {**r, "reference": ctx.memo_overrides.get(r["txn_id"], r.get("reference"))}
        if "reference" in r
        else r
        for r in rows
    ]


def _scan_memos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for row in rows:
        memo = row.get("reference")
        if memo and (result := scan(str(memo))).is_attempt:
            hits.append({"txn_id": row["txn_id"], "field": "reference", "rules": result.rules})
    return hits


def _transaction_view(ctx: ToolContext, row: dict[str, Any]) -> dict[str, Any]:
    minimized = minimize_row(row, ctx.levels("transactions"))
    view = dict(minimized.data)
    memo = minimized.untrusted.get("reference")
    if memo is not None:
        view["reference"] = wrap_untrusted(str(memo))
    return view


def get_alert(ctx: ToolContext, alert_id: str) -> Evidence:
    """The alert, its score explanation (SHAP top-5) and its top contributing transactions."""
    with ctx.connect() as con:
        rows = _rows(con, "SELECT * FROM alerts WHERE alert_id = $a", {"a": alert_id})
        if not rows:
            raise ToolError(f"unknown alert {alert_id}")
        alert = rows[0]
        ids = list(alert["contributing_txn_ids"])[:10]
        txns = _apply_overrides(ctx, _rows(con, _ALERT_TXNS_SQL, {"ids": ids}))
    view = minimize_row(alert, ctx.levels("alerts")).data
    data = {
        "alert_id": alert_id,
        "account_key": view.get("account_key"),
        "alert_date": view.get("alert_date"),
        "risk_score": round(float(view["score"]), 4),
        "threshold": round(float(view["threshold"]), 4),
        "model_version": view.get("model_version"),
        "priority_rank": view.get("priority_rank"),
        "n_transactions_that_day": view.get("n_txn"),
        "score_drivers_shap_top5": json.loads(view["shap_top5_json"]),
        "top_transactions": [_transaction_view(ctx, t) for t in txns],
    }
    return ctx.store.add(
        "get_alert", {"alert_id": alert_id}, data, untrusted=True, injection_hits=_scan_memos(txns)
    )


def get_account_profile(ctx: ToolContext, account_key: str) -> Evidence:
    """Owner entity (type, country, synthetic name), the owner's other accounts, activity span."""
    with ctx.connect() as con:
        found = _rows(con, _PROFILE_SQL, {"k": account_key})
        if not found:
            raise ToolError(f"unknown account {account_key}")
        profile = found[0]
        siblings = _rows(con, _SIBLINGS_SQL, {"k": account_key, "entity": profile["entity_id"]})
        activity = _rows(con, _ACTIVITY_SQL, {"k": account_key})[0]
    account = minimize_row(
        {k: profile[k] for k in ("account_key", "bank_name")}, ctx.levels("accounts")
    ).data
    owner_fields = (
        "entity_id",
        "entity_type",
        "country",
        "synthetic_name",
        "date_of_birth",
        "n_accounts",
    )
    owner = minimize_row({k: profile[k] for k in owner_fields}, ctx.levels("entities"))
    name = owner.untrusted.get("synthetic_name")
    data = {
        **account,
        "owner": {**owner.data, "name": wrap_untrusted(str(name)) if name else None},
        "other_accounts_of_owner": siblings,
        "activity": activity,
    }
    return ctx.store.add("get_account_profile", {"account_key": account_key}, data, untrusted=True)


def get_transactions(
    ctx: ToolContext,
    account_key: str,
    start: str,
    end: str,
    direction: Direction = "both",
    limit: int = 30,
) -> Evidence:
    """Transactions of an account in a window (max 14 days, max 50 rows), plus exact totals."""
    if direction not in _SIDE:
        raise ToolError("direction must be 'in', 'out' or 'both'")
    s, e = _window(start, end)
    params = {"k": account_key, "s": s, "e": e}
    with ctx.connect() as con:
        totals = _rows(con, _TOTALS_SQL[direction], params)[0]
        rows = _apply_overrides(
            ctx, _rows(con, _ROWS_SQL[direction], {**params, "limit": max(1, min(limit, MAX_ROWS))})
        )
    data = {
        "account_key": account_key,
        "window": _span(s, e),
        "direction": direction,
        "totals": totals,
        "shown": len(rows),
        "ordering": "largest amount_usd first",
        "transactions": [_transaction_view(ctx, r) for r in rows],
    }
    args = {"account_key": account_key, "start": start, "end": end, "direction": direction}
    return ctx.store.add(
        "get_transactions", args, data, untrusted=True, injection_hits=_scan_memos(rows)
    )


def get_counterparties(
    ctx: ToolContext, account_key: str, start: str, end: str, top: int = 10
) -> Evidence:
    """Counterparties by volume in a window, with direction, counts, totals and first/last time."""
    s, e = _window(start, end)
    params = {"k": account_key, "s": s, "e": e, "top": max(1, min(top, 25))}
    with ctx.connect() as con:
        rows = _rows(con, _COUNTERPARTIES_SQL, params)
    data = {"account_key": account_key, "window": _span(s, e), "counterparties": rows}
    args = {"account_key": account_key, "start": start, "end": end}
    return ctx.store.add("get_counterparties", args, data)


def trace_flows(
    ctx: ToolContext,
    account_key: str,
    start: str,
    end: str,
    direction: Literal["in", "out"] = "out",
    hops: int = 2,
) -> Evidence:
    """Multi-hop money flow from (``out``) or into (``in``) an account, max 3 hops.

    Causality is respected: going forward, a next hop must happen after the money reached
    the node; going backward, a previous hop must happen before the money left the node.
    Each step keeps the 5 largest edges.
    """
    if direction not in ("in", "out"):
        raise ToolError("direction must be 'in' or 'out'")
    s, e = _window(start, end)
    hops = max(1, min(hops, MAX_HOPS))
    sql = _FLOW_OUT_SQL if direction == "out" else _FLOW_IN_SQL
    edges: list[dict[str, Any]] = []
    seen = {account_key}
    first_time = s if direction == "out" else e
    frontier: deque[tuple[str, datetime, int]] = deque([(account_key, first_time, 0)])
    with ctx.connect() as con:
        while frontier:
            node, t, depth = frontier.popleft()
            if depth >= hops:
                continue
            bound = (
                {"node": node, "t": t, "e": e}
                if direction == "out"
                else {"node": node, "t": t, "s": s}
            )
            for row in _rows(con, sql, bound):
                src, dst = (node, row["other"]) if direction == "out" else (row["other"], node)
                edges.append(
                    {
                        "hop": depth + 1,
                        "from": src,
                        "to": dst,
                        "n_transactions": row["n"],
                        "amount_usd": row["amount_usd"],
                        "time": row["t"],
                    }
                )
                if row["other"] not in seen:
                    seen.add(row["other"])
                    frontier.append((row["other"], row["t"], depth + 1))
    endpoint = "to" if direction == "out" else "from"
    data = {
        "account_key": account_key,
        "direction": direction,
        "hops": hops,
        "window": _span(s, e),
        "edges": edges[:40],
        "n_accounts_reached": len(seen) - 1,
        "returns_to_origin": any(x[endpoint] == account_key for x in edges if x["hop"] > 1),
    }
    args = {
        "account_key": account_key,
        "start": start,
        "end": end,
        "direction": direction,
        "hops": hops,
    }
    return ctx.store.add("trace_flows", args, data)


def _chain(
    con: duckdb.DuckDBPyConnection,
    account_key: str,
    s: datetime,
    e: datetime,
    direction: Literal["in", "out"],
) -> tuple[dict[str, int], int | None]:
    """Amount-matched chain from (``out``) or into (``in``) the account, level by level.

    Level 1 is the account's own transfers; each next level keeps only transfers that follow
    (``out``) or precede (``in``) the previous hop within ``CHAIN_MAX_DELAY_HOURS`` for an
    amount within ``CHAIN_AMOUNT_BAND``. Bounded: ``CHAIN_FRONTIER`` rows per level,
    ``CHAIN_MAX_DEPTH`` levels. Returns the first level at which each account was reached
    and the level at which the chain came back to the account itself (if it did).
    """
    lo, hi = CHAIN_AMOUNT_BAND
    base = {"s": s, "e": e, "cap": CHAIN_FRONTIER}
    frontier = _rows(con, _CHAIN_START_SQL[direction], {**base, "k": account_key})
    sql = _CHAIN_FWD_SQL if direction == "out" else _CHAIN_BWD_SQL
    levels: dict[str, int] = {}
    back_home: int | None = None
    level = 1
    while frontier:
        if level > 1 and back_home is None and any(r["node"] == account_key for r in frontier):
            back_home = level
        fresh: dict[str, dict[str, Any]] = {}
        for r in frontier:
            if r["node"] != account_key and r["node"] not in levels and r["node"] not in fresh:
                fresh[r["node"]] = r
        if not fresh:
            break
        levels.update(dict.fromkeys(fresh, level))
        if level >= CHAIN_MAX_DEPTH[direction]:
            break
        level += 1
        frontier = _rows(
            con,
            sql,
            {
                **base,
                "nodes": list(fresh),
                "amts": [float(r["amt"]) for r in fresh.values()],
                "times": [r["t"] for r in fresh.values()],
                "delay": CHAIN_MAX_DELAY_HOURS,
                "lo": lo,
                "hi": hi,
            },
        )
    return levels, back_home


def _cycle_length(
    forward: dict[str, int], backward: dict[str, int], home: int | None
) -> int | None:
    """Shortest amount-matched cycle through the account.

    In a time-ordered cycle v0 -> ... -> v0 the account is usually in the middle: the money
    reaches the first node v0 again going forward, and v0 is also where it came from going
    backward. So a node met by both chains closes a cycle of forward + backward hops; a
    forward chain that comes back to the account itself is a cycle starting at the account.
    """
    met = [forward[n] + backward[n] for n in forward.keys() & backward.keys()]
    lengths = [*met, *([home] if home else [])]
    return min(lengths) if lengths else None


def multi_account_indicators(
    con: duckdb.DuckDBPyConnection, account_key: str, s: datetime, e: datetime
) -> dict[str, Any]:
    """Two-hop and chain indicators of multi-account typologies (R3.15, ADR-03.11).

    Counts and ratios, plus at most one counterparty account per indicator (the one that
    makes the number), so the output stays small and minimal (R3.9).
    """
    m = _rows(con, _MULTI_SQL, {"k": account_key, "s": s, "e": e})[0]
    forward, home = _chain(con, account_key, s, e, "out")
    backward, _ = _chain(con, account_key, s, e, "in")
    total_out = float(m["total_out_usd"] or 0)
    share = round(float(m["src_amt"] or 0) / total_out, 3) if total_out and m["src_n"] else None
    return {
        "scatter_gather_source": {
            "converging_intermediaries": m["src_n"] or 0,
            "destination": m["src_dest"],
            "share_of_outflow_converging": share,
        },
        "scatter_gather_intermediary": {
            "parallel_intermediaries": m["mid_n"] or 0,
            "source": m["mid_source"],
            "destination": m["mid_dest"],
        },
        "gather_scatter_hub": {"hub_degree": m["own_hub_degree"] or 0},
        "gather_scatter_feeder": {"hub_degree": m["hub_degree"] or 0, "hub": m["hub"]},
        "fan_in_feeder": {
            "receiver_distinct_senders": m["fan_in_senders"] or 0,
            "receiver": m["fan_in_receiver"],
        },
        "amount_matched_chain": {
            "forward_depth": max(forward.values(), default=0),
            "backward_depth": max(backward.values(), default=0),
            "cycle_length": _cycle_length(forward, backward, home),
        },
    }


def detect_patterns(ctx: ToolContext, account_key: str, start: str, end: str) -> Evidence:
    """Code-computed typology indicators: fan-in/out, cycles, flow-through, structuring,
    multi-account shapes (scatter-gather, gather-scatter, fan-in feeder, amount-matched
    chains) and the typology candidates they support (R3.15, ADR-03.11).

    The agent tests its hypotheses against these numbers instead of estimating them.
    """
    s, e = _window(start, end)
    params = {"k": account_key, "s": s, "e": e}
    with ctx.connect() as con:

        def one(sql: str) -> dict[str, Any]:
            return _rows(con, sql, params)[0]

        fan = one(_FAN_SQL)
        burst = one(_BURST_SQL)
        cycles = one(_CYCLES_SQL)
        flow = one(_FLOW_THROUGH_SQL)
        structuring = one(_STRUCTURING_SQL)
        formats = _rows(con, _FORMATS_SQL, params)
        multi = multi_account_indicators(con, account_key, s, e) if ctx.multi_account else None
    receivers, senders = fan["distinct_receivers"] or 0, fan["distinct_senders"] or 0
    total_in, total_out = fan["total_in_usd"] or 0, fan["total_out_usd"] or 0
    ratio = round(float(total_out) / float(total_in), 3) if total_in else None
    data = {
        "account_key": account_key,
        "window": _span(s, e),
        "fan": {
            **fan,
            **burst,
            "fan_out_ratio": round(receivers / (1 + senders), 2),
            "fan_in_ratio": round(senders / (1 + receivers), 2),
        },
        "cycles_through_account": cycles,
        "flow_through": {**flow, "out_to_in_amount_ratio": ratio},
        "structuring": structuring,
        "payment_formats_out": formats,
        "thresholds_note": "just-under-10k = amount_usd in [9000, 10000)",
    }
    if multi is not None:
        # New keys go last, so the earlier part of the summary the agent sees is unchanged.
        data["multi_account"] = multi
        candidates = typology_candidates(
            {"fan": data["fan"], "cycles": cycles, "multi": multi}, ctx.candidate_rules()
        )
        data["typology_candidates"] = [
            {
                "typology": c.typology,
                "indicator": c.indicator,
                "observed": c.observed,
                "rule": c.rule,
            }
            for c in candidates
        ]
    args = {"account_key": account_key, "start": start, "end": end}
    return ctx.store.add("detect_patterns", args, data)
