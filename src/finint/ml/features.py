"""Per-transaction tabular features (task 2.2 · R2.1, R2.2).

All account-history windows look **backwards only**: ``RANGE BETWEEN <w> PRECEDING AND
CURRENT ROW EXCLUDE TIES``. ``EXCLUDE TIES`` matters: AMLworld has ~238k same-minute groups
from one sender, and without it a transaction would "see" its siblings in the same
minute, which a real-time scorer could not.

Flow-through features (money in shortly before money out) come from a unified per-account
event stream, so a sender's recent *incoming* funds are visible when it pays out.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

CATEGORICAL = ("payment_format", "currency_paid", "currency_received")

NUMERIC = (
    "amount_usd",
    "log_amount_usd",
    "cross_currency",
    "cross_bank",
    "self_transfer",
    "hour",
    "weekday",
    "is_whole_amount",
    "is_round_100",
    "just_under_10k",
    "out_cnt_1h",
    "out_sum_1h",
    "out_cnt_24h",
    "out_sum_24h",
    "out_cnt_7d",
    "out_sum_7d",
    "amount_vs_out_mean_7d",
    "new_pair",
    "sender_age_hours",
    "in_cnt_24h_receiver",
    "in_sum_24h_receiver",
    "in_cnt_7d_receiver",
    "sender_in_sum_24h",
    "sender_in_cnt_24h",
    "flow_through_ratio_24h",
    "receiver_out_cnt_24h",
)

FEATURES = (*NUMERIC, *CATEGORICAL)


def _window(partition: str, span: str) -> str:
    return (
        f"PARTITION BY {partition} ORDER BY ts "
        f"RANGE BETWEEN INTERVAL {span} PRECEDING AND CURRENT ROW EXCLUDE TIES"
    )


_SQL = f"""
CREATE OR REPLACE TABLE features_txn AS
WITH base AS (
    SELECT txn_id, ts, from_key, to_key, CAST(amount_usd AS DOUBLE) AS amt,
           amount_paid, payment_format, currency_paid, currency_received,
           cross_currency, cross_bank, self_transfer
    FROM transactions
),
sender AS (
    SELECT txn_id,
        count(*) OVER w1h AS out_cnt_1h,  sum(amt) OVER w1h AS out_sum_1h,
        count(*) OVER w24h AS out_cnt_24h, sum(amt) OVER w24h AS out_sum_24h,
        count(*) OVER w7d AS out_cnt_7d,  sum(amt) OVER w7d AS out_sum_7d,
        row_number() OVER (PARTITION BY from_key, to_key ORDER BY ts, txn_id) = 1 AS new_pair,
        date_diff('hour', min(ts) OVER (PARTITION BY from_key), ts) AS sender_age_hours
    FROM base
    WINDOW w1h AS ({_window("from_key", "1 HOUR")}),
           w24h AS ({_window("from_key", "1 DAY")}),
           w7d AS ({_window("from_key", "7 DAYS")})
),
receiver AS (
    SELECT txn_id,
        count(*) OVER w24h AS in_cnt_24h_receiver, sum(amt) OVER w24h AS in_sum_24h_receiver,
        count(*) OVER w7d AS in_cnt_7d_receiver
    FROM base
    WINDOW w24h AS ({_window("to_key", "1 DAY")}),
           w7d AS ({_window("to_key", "7 DAYS")})
),
events AS (
    SELECT txn_id, to_key AS acct, ts, amt, 1 AS is_in FROM base
    UNION ALL
    SELECT txn_id, from_key AS acct, ts, amt, 0 AS is_in FROM base
),
event_windows AS (
    SELECT txn_id, is_in,
        sum(amt * is_in) OVER w AS in_sum_24h,
        sum(is_in) OVER w AS in_cnt_24h,
        sum(1 - is_in) OVER w AS out_cnt_24h
    FROM events
    WINDOW w AS ({_window("acct", "1 DAY")})
),
flow AS (
    SELECT o.txn_id,
        coalesce(o.in_sum_24h, 0) AS sender_in_sum_24h,
        coalesce(o.in_cnt_24h, 0) AS sender_in_cnt_24h,
        coalesce(i.out_cnt_24h, 0) AS receiver_out_cnt_24h
    FROM event_windows o
    JOIN event_windows i ON i.txn_id = o.txn_id AND i.is_in = 1
    WHERE o.is_in = 0
)
SELECT
    b.txn_id,
    b.amt AS amount_usd,
    ln(1 + b.amt) AS log_amount_usd,
    b.cross_currency::INT AS cross_currency,
    b.cross_bank::INT AS cross_bank,
    b.self_transfer::INT AS self_transfer,
    hour(b.ts) AS hour,
    dayofweek(b.ts) AS weekday,
    (b.amount_paid = round(b.amount_paid, 0))::INT AS is_whole_amount,
    (b.amount_paid % 100 = 0)::INT AS is_round_100,
    (b.amt >= 9000 AND b.amt < 10000)::INT AS just_under_10k,
    s.out_cnt_1h, s.out_sum_1h, s.out_cnt_24h, s.out_sum_24h, s.out_cnt_7d, s.out_sum_7d,
    b.amt / nullif(s.out_sum_7d / s.out_cnt_7d, 0) AS amount_vs_out_mean_7d,
    s.new_pair::INT AS new_pair,
    s.sender_age_hours,
    r.in_cnt_24h_receiver, r.in_sum_24h_receiver, r.in_cnt_7d_receiver,
    f.sender_in_sum_24h, f.sender_in_cnt_24h,
    f.sender_in_sum_24h / nullif(b.amt, 0) AS flow_through_ratio_24h,
    f.receiver_out_cnt_24h,
    b.payment_format, b.currency_paid, b.currency_received
FROM base b
JOIN sender s USING (txn_id)
JOIN receiver r USING (txn_id)
JOIN flow f USING (txn_id)
ORDER BY b.txn_id
"""  # noqa: S608 - module constants only


def build_tabular_features(db_path: Path) -> int:
    """(Re)create ``features_txn``. Returns its row count."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(_SQL)
        row = con.execute("SELECT count(*) FROM features_txn").fetchone()
    return int(row[0]) if row else 0
