"""Load AMLworld raw files into DuckDB (tasks 1.2, 1.3 · R1.1).

Tables written:

* ``transactions``: one row per CSV line, stable ``txn_id`` (file order), typed columns,
  account keys ``bank:account`` and the attempt link (``attempt_id``, ``typology``).
* ``accounts``: account → owning entity.
* ``attempts``: laundering attempts from the patterns file.
* ``pattern_unmatched``: pattern lines that matched no transaction (reported, never dropped
  silently).

All SQL is static. The only runtime values are file paths, passed as bound parameters.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import duckdb
from pydantic import BaseModel

from finint.data.patterns import Attempt, PatternTxn, iter_pattern_file, parse_patterns
from finint.data.sources import AmlWorldFiles, MissingRawDataError

# The CSV header repeats "Account", so columns are named explicitly by position.
_LOAD_TRANSACTIONS = """
CREATE OR REPLACE TABLE transactions AS
WITH raw AS (
    SELECT * FROM read_csv(
        $path,
        header = true,
        columns = {
            'ts_raw': 'VARCHAR',
            'from_bank_raw': 'VARCHAR',
            'from_account': 'VARCHAR',
            'to_bank_raw': 'VARCHAR',
            'to_account': 'VARCHAR',
            'amount_received': 'DECIMAL(38, 8)',
            'currency_received': 'VARCHAR',
            'amount_paid': 'DECIMAL(38, 8)',
            'currency_paid': 'VARCHAR',
            'payment_format': 'VARCHAR',
            'is_laundering_raw': 'INTEGER'
        }
    )
)
SELECT
    row_number() OVER () AS txn_id,
    strptime(ts_raw, '%Y/%m/%d %H:%M') AS ts,
    coalesce(nullif(ltrim(from_bank_raw, '0'), ''), '0') AS from_bank,
    from_account,
    coalesce(nullif(ltrim(to_bank_raw, '0'), ''), '0') AS to_bank,
    to_account,
    amount_received,
    currency_received,
    amount_paid,
    currency_paid,
    payment_format,
    is_laundering_raw = 1 AS is_laundering
FROM raw
"""

_ADD_KEYS = """
ALTER TABLE transactions ADD COLUMN from_key VARCHAR;
ALTER TABLE transactions ADD COLUMN to_key VARCHAR;
UPDATE transactions SET from_key = from_bank || ':' || from_account,
                        to_key = to_bank || ':' || to_account;
ALTER TABLE transactions ADD COLUMN attempt_id INTEGER;
ALTER TABLE transactions ADD COLUMN typology VARCHAR;
"""

_LOAD_ACCOUNTS = """
CREATE OR REPLACE TABLE accounts AS
SELECT
    coalesce(nullif(ltrim(CAST("Bank ID" AS VARCHAR), '0'), ''), '0') AS bank_id,
    "Account Number" AS account_number,
    coalesce(nullif(ltrim(CAST("Bank ID" AS VARCHAR), '0'), ''), '0')
        || ':' || "Account Number" AS account_key,
    "Entity ID" AS entity_id,
    "Bank Name" AS bank_name,
    "Entity Name" AS entity_label
FROM read_csv($path, header = true, all_varchar = true)
"""

_CREATE_PATTERN_STAGE = """
CREATE OR REPLACE TEMP TABLE pattern_stage (
    line_idx INTEGER, attempt_id INTEGER, typology VARCHAR, ts TIMESTAMP,
    from_bank VARCHAR, from_account VARCHAR, to_bank VARCHAR, to_account VARCHAR,
    amount_received DECIMAL(38, 8), currency_received VARCHAR,
    amount_paid DECIMAL(38, 8), currency_paid VARCHAR, payment_format VARCHAR
)
"""

_MATCH = """
CREATE OR REPLACE TEMP TABLE pattern_match AS
SELECT p.line_idx, p.attempt_id, p.typology, t.txn_id
FROM pattern_stage p
JOIN transactions t
  ON t.ts = p.ts
 AND t.from_bank = p.from_bank AND t.from_account = p.from_account
 AND t.to_bank = p.to_bank AND t.to_account = p.to_account
 AND t.amount_paid = p.amount_paid AND t.currency_paid = p.currency_paid
 AND t.amount_received = p.amount_received AND t.currency_received = p.currency_received
 AND t.payment_format = p.payment_format
"""

# A transaction can (rarely) be listed in more than one attempt: keep the first attempt.
_LINK = """
UPDATE transactions AS t
SET attempt_id = m.attempt_id, typology = m.typology
FROM (
    SELECT txn_id, arg_min(attempt_id, attempt_id) AS attempt_id,
           arg_min(typology, attempt_id) AS typology
    FROM pattern_match GROUP BY txn_id
) AS m
WHERE t.txn_id = m.txn_id
"""

_UNMATCHED = """
CREATE OR REPLACE TABLE pattern_unmatched AS
SELECT p.* FROM pattern_stage p
WHERE NOT EXISTS (SELECT 1 FROM pattern_match m WHERE m.line_idx = p.line_idx)
"""


class IngestReport(BaseModel):
    """Counts that feed ``reports/data_quality.md`` (task 1.10)."""

    n_transactions: int
    n_laundering: int
    n_accounts: int
    n_entities: int
    n_attempts: int
    n_pattern_lines: int
    n_pattern_unmatched: int
    n_linked_transactions: int
    n_multi_attempt_transactions: int
    n_laundering_without_attempt: int
    attempts_by_typology: dict[str, int]


def ingest(files: AmlWorldFiles, db_path: Path) -> IngestReport:
    """Rebuild the raw tables from scratch (idempotent)."""
    missing = files.missing()
    if missing:
        raise MissingRawDataError(missing, files.transactions.parent)

    attempts, pattern_txns = parse_patterns(iter_pattern_file(files.patterns))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute("SET preserve_insertion_order = true")
        con.execute(_LOAD_TRANSACTIONS, {"path": str(files.transactions)})
        con.execute(_ADD_KEYS)
        con.execute(_LOAD_ACCOUNTS, {"path": str(files.accounts)})
        _write_attempts(con, attempts)
        _stage_patterns(con, pattern_txns)
        con.execute(_MATCH)
        con.execute(_LINK)
        con.execute(_UNMATCHED)
        return _report(con, attempts, len(pattern_txns))


def _write_attempts(con: duckdb.DuckDBPyConnection, attempts: list[Attempt]) -> None:
    con.execute(
        "CREATE OR REPLACE TABLE attempts "
        "(attempt_id INTEGER PRIMARY KEY, typology VARCHAR, detail VARCHAR)"
    )
    if attempts:
        con.executemany(
            "INSERT INTO attempts VALUES (?, ?, ?)",
            [(a.attempt_id, a.typology, a.detail) for a in attempts],
        )


def _stage_patterns(con: duckdb.DuckDBPyConnection, txns: list[PatternTxn]) -> None:
    con.execute(_CREATE_PATTERN_STAGE)
    if txns:
        con.executemany(
            "INSERT INTO pattern_stage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    idx, t.attempt_id, t.typology, t.ts, t.from_bank, t.from_account,
                    t.to_bank, t.to_account, t.amount_received, t.currency_received,
                    t.amount_paid, t.currency_paid, t.payment_format,
                )
                for idx, t in enumerate(txns)
            ],
        )  # fmt: skip


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _report(
    con: duckdb.DuckDBPyConnection, attempts: list[Attempt], n_pattern_lines: int
) -> IngestReport:
    return IngestReport(
        n_transactions=_scalar(con, "SELECT count(*) FROM transactions"),
        n_laundering=_scalar(con, "SELECT count(*) FROM transactions WHERE is_laundering"),
        n_accounts=_scalar(con, "SELECT count(*) FROM accounts"),
        n_entities=_scalar(con, "SELECT count(DISTINCT entity_id) FROM accounts"),
        n_attempts=len(attempts),
        n_pattern_lines=n_pattern_lines,
        n_pattern_unmatched=_scalar(con, "SELECT count(*) FROM pattern_unmatched"),
        n_linked_transactions=_scalar(
            con, "SELECT count(*) FROM transactions WHERE attempt_id IS NOT NULL"
        ),
        n_multi_attempt_transactions=_scalar(
            con,
            "SELECT count(*) FROM (SELECT txn_id FROM pattern_match "
            "GROUP BY txn_id HAVING count(DISTINCT attempt_id) > 1)",
        ),
        n_laundering_without_attempt=_scalar(
            con, "SELECT count(*) FROM transactions WHERE is_laundering AND attempt_id IS NULL"
        ),
        attempts_by_typology=dict(Counter(a.typology for a in attempts)),
    )
