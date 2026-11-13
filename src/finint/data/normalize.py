"""Normalize ingested transactions (task 1.4 · R1.2).

* ``fx_rates``: USD rate per currency, **derived from the dataset itself**. The AMLworld
  simulator converts with fixed rates, so the median ``received / paid`` ratio over
  cross-currency transfers into USD recovers them. ``config/fx.yaml`` is the documented
  fallback for a currency without enough USD pairs.
* ``transactions`` gains ``amount_usd``, ``cross_currency``, ``cross_bank``,
  ``self_transfer`` and ``ts_shifted``.

``ts_shifted`` moves the dataset so its last day lands on (or just before) the configured
as-of date, shifting by **whole weeks** so weekday patterns stay intact. Charts use
``ts_shifted``; models always use ``ts``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
from pydantic import BaseModel

MIN_FX_OBSERVATIONS = 10

# Columns written by ingest. Listing them keeps normalize idempotent: re-running it
# rebuilds the derived columns instead of appending duplicates.
_RAW_COLUMNS = (
    "txn_id, ts, from_bank, from_account, to_bank, to_account, amount_received, "
    "currency_received, amount_paid, currency_paid, payment_format, is_laundering, "
    "from_key, to_key, attempt_id, typology"
)

_DERIVE_FX = f"""
CREATE OR REPLACE TABLE fx_rates AS
WITH pairs AS (
    SELECT currency_paid AS currency,
           count(*) AS n_obs,
           median(CAST(amount_received AS DOUBLE) / CAST(amount_paid AS DOUBLE)) AS usd_rate
    FROM transactions
    WHERE currency_received = 'US Dollar'
      AND currency_paid <> 'US Dollar'
      AND amount_paid > 0
    GROUP BY currency_paid
)
SELECT 'US Dollar' AS currency, 1.0 AS usd_rate, NULL::BIGINT AS n_obs, 'identity' AS source
UNION ALL
SELECT currency, usd_rate, n_obs, 'dataset_pairs' AS source
FROM pairs WHERE n_obs >= {MIN_FX_OBSERVATIONS}
"""  # noqa: S608 - module constants only

_REBUILD = f"""
CREATE OR REPLACE TABLE transactions AS
SELECT
    {_RAW_COLUMNS},
    CAST(round(CAST(t.amount_paid AS DOUBLE) * fx.usd_rate, 2) AS DECIMAL(38, 2)) AS amount_usd,
    t.currency_paid <> t.currency_received AS cross_currency,
    t.from_bank <> t.to_bank AS cross_bank,
    t.from_key = t.to_key AS self_transfer,
    t.ts + to_days($shift_days) AS ts_shifted
FROM (SELECT {_RAW_COLUMNS} FROM transactions) AS t
LEFT JOIN fx_rates AS fx ON fx.currency = t.currency_paid
ORDER BY t.txn_id
"""  # noqa: S608 - module constants only


class NormalizeReport(BaseModel):
    fx_rates: dict[str, float]
    fx_fallbacks: list[str]
    shift_days: int
    shifted_range: tuple[date, date]
    n_missing_usd: int
    cross_currency_share: float
    cross_bank_share: float
    self_transfer_share: float


def week_aligned_shift(last_day: date, as_of: date) -> int:
    """Days to add so ``last_day`` lands on the latest date ≤ ``as_of`` with the same weekday."""
    return ((as_of - last_day).days // 7) * 7


def normalize(db_path: Path, as_of: date, fallback_rates: dict[str, float]) -> NormalizeReport:
    with duckdb.connect(str(db_path)) as con:
        con.execute(_DERIVE_FX)
        fallbacks = _apply_fallbacks(con, fallback_rates)
        last_day = _fetch_date(con, "SELECT max(ts)::DATE FROM transactions")
        shift_days = week_aligned_shift(last_day, as_of)
        con.execute(_REBUILD, {"shift_days": shift_days})
        return _report(con, fallbacks, shift_days)


def _apply_fallbacks(con: duckdb.DuckDBPyConnection, fallback_rates: dict[str, float]) -> list[str]:
    missing = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT currency_paid FROM transactions "
            "WHERE currency_paid NOT IN (SELECT currency FROM fx_rates) ORDER BY 1"
        ).fetchall()
    ]
    used = [currency for currency in missing if currency in fallback_rates]
    if used:
        con.executemany(
            "INSERT INTO fx_rates VALUES (?, ?, NULL, 'config')",
            [(currency, fallback_rates[currency]) for currency in used],
        )
    return used


def _fetch_date(con: duckdb.DuckDBPyConnection, sql: str) -> date:
    row = con.execute(sql).fetchone()
    if row is None or row[0] is None:
        raise ValueError("transactions table is empty; run ingest first")
    value: date = row[0]
    return value


def _report(
    con: duckdb.DuckDBPyConnection, fallbacks: list[str], shift_days: int
) -> NormalizeReport:
    rates = dict(con.execute("SELECT currency, usd_rate FROM fx_rates ORDER BY 1").fetchall())
    stats = con.execute(
        "SELECT min(ts_shifted)::DATE, max(ts_shifted)::DATE, "
        "count(*) FILTER (WHERE amount_usd IS NULL), "
        "avg(cross_currency::INT), avg(cross_bank::INT), avg(self_transfer::INT) "
        "FROM transactions"
    ).fetchone()
    if stats is None:  # an aggregate always returns one row; guard for the type checker
        raise RuntimeError("transaction stats query returned no row")
    first, last, n_missing, cross_cur, cross_bank, self_tx = stats
    return NormalizeReport(
        fx_rates={k: float(v) for k, v in rates.items()},
        fx_fallbacks=fallbacks,
        shift_days=shift_days,
        shifted_range=(first, last),
        n_missing_usd=int(n_missing),
        cross_currency_share=float(cross_cur),
        cross_bank_share=float(cross_bank),
        self_transfer_share=float(self_tx),
    )
