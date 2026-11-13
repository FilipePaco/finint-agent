"""Raw ingestion into DuckDB (tasks 1.2, 1.3 · R1.1)."""

from __future__ import annotations

import shutil
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from finint.data.ingest import IngestReport, ingest
from finint.data.sources import AmlWorldFiles, MissingRawDataError

FIXTURES = Path(__file__).parent / "fixtures" / "amlworld"


@pytest.fixture
def db(tmp_path: Path) -> tuple[Path, IngestReport]:
    path = tmp_path / "finint.duckdb"
    return path, ingest(AmlWorldFiles.in_dir(FIXTURES), path)


def test_report_counts(db: tuple[Path, IngestReport]) -> None:
    _, report = db
    assert report.n_transactions == 9
    assert report.n_laundering == 5
    assert report.n_accounts == 10
    assert report.n_entities == 9  # one entity holds accounts in two banks
    assert report.n_attempts == 2
    assert report.attempts_by_typology == {"FAN-OUT": 1, "CYCLE": 1}
    assert report.n_pattern_lines == 6
    assert report.n_linked_transactions == 5
    assert report.n_pattern_unmatched == 1
    assert report.n_laundering_without_attempt == 0


def test_txn_ids_follow_file_order(db: tuple[Path, IngestReport]) -> None:
    path, _ = db
    with duckdb.connect(str(path), read_only=True) as con:
        rows = con.execute(
            "SELECT txn_id, from_account FROM transactions ORDER BY txn_id"
        ).fetchall()
    assert [r[0] for r in rows] == list(range(1, 10))
    assert rows[0][1] == "8000EBD30"
    assert rows[-1][1] == "8000EBD30"


def test_typed_columns_and_keys(db: tuple[Path, IngestReport]) -> None:
    path, _ = db
    with duckdb.connect(str(path), read_only=True) as con:
        row = con.execute(
            "SELECT ts, from_key, to_key, amount_paid, currency_paid, amount_received, "
            "currency_received, is_laundering, typology FROM transactions WHERE txn_id = 6"
        ).fetchone()
    assert row is not None
    ts, from_key, to_key, paid, cur_paid, received, cur_rec, laundering, typology = row
    assert ts.isoformat() == "2022-09-02T10:15:00"
    assert (from_key, to_key) == ("11:800A12340", "24:800B56780")
    assert (paid, cur_paid) == (Decimal("9400.25"), "US Dollar")
    assert (received, cur_rec) == (Decimal("8800.00"), "Euro")
    assert laundering is True
    assert typology == "CYCLE"


def test_small_crypto_amounts_keep_precision(db: tuple[Path, IngestReport]) -> None:
    path, _ = db
    with duckdb.connect(str(path), read_only=True) as con:
        row = con.execute("SELECT amount_paid FROM transactions WHERE txn_id = 5").fetchone()
    assert row is not None
    assert row[0] == Decimal("0.000412")


def test_account_keys_join_transactions(db: tuple[Path, IngestReport]) -> None:
    path, _ = db
    with duckdb.connect(str(path), read_only=True) as con:
        orphans = con.execute(
            "SELECT count(*) FROM transactions t "
            "LEFT JOIN accounts a ON a.account_key = t.from_key WHERE a.account_key IS NULL"
        ).fetchone()
    assert orphans == (0,)


def test_unmatched_pattern_line_is_kept_for_reporting(db: tuple[Path, IngestReport]) -> None:
    path, _ = db
    with duckdb.connect(str(path), read_only=True) as con:
        rows = con.execute("SELECT from_account, typology FROM pattern_unmatched").fetchall()
    assert rows == [("800DEAD00", "CYCLE")]


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "finint.duckdb"
    first = ingest(AmlWorldFiles.in_dir(FIXTURES), path)
    second = ingest(AmlWorldFiles.in_dir(FIXTURES), path)
    assert first == second


def test_missing_files_explain_the_kaggle_download(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIXTURES / "HI-Small_Trans.csv", raw)
    with pytest.raises(MissingRawDataError) as excinfo:
        ingest(AmlWorldFiles.in_dir(raw), tmp_path / "x.duckdb")
    message = str(excinfo.value)
    assert "HI-Small_accounts.csv" in message
    assert "HI-Small_Patterns.txt" in message
    assert "finint data download" in message
    assert "HI-Small_Trans.csv," not in message
