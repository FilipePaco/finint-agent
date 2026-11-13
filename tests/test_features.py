"""Tabular features and leakage control (task 2.2 · R2.1, R2.2)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest

from finint.data.ingest import ingest
from finint.data.normalize import normalize
from finint.data.sources import AmlWorldFiles
from finint.ml.features import FEATURES, build_tabular_features
from synthetic_amlworld import write_dataset

FX = {"US Dollar": 1.0}


@pytest.fixture(scope="module")
def featured(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("features")
    write_dataset(root / "raw")
    db = root / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(root / "raw"), db)
    normalize(db, date(2026, 11, 13), FX)
    build_tabular_features(db)
    return db


def _row(db: Path, txn_id: int) -> dict[str, Any]:
    with duckdb.connect(str(db), read_only=True) as con:
        cursor = con.execute("SELECT * FROM features_txn WHERE txn_id = ?", [txn_id])
        names = [d[0] for d in cursor.description]
        values = cursor.fetchone()
    assert values is not None
    return dict(zip(names, values, strict=True))


def test_one_feature_row_per_transaction(featured: Path) -> None:
    with duckdb.connect(str(featured), read_only=True) as con:
        n_txn = con.execute("SELECT count(*) FROM transactions").fetchone()
        n_feat = con.execute("SELECT count(DISTINCT txn_id) FROM features_txn").fetchone()
        columns = {r[0] for r in con.execute("DESCRIBE features_txn").fetchall()}
    assert n_txn == n_feat
    assert set(FEATURES) <= columns


@pytest.mark.parametrize("quantile", [0.05, 0.3, 0.6, 0.9, 0.97])
def test_features_use_only_the_past(featured: Path, tmp_path: Path, quantile: float) -> None:
    """Recompute on data truncated at the transaction's timestamp: features must not change."""
    with duckdb.connect(str(featured), read_only=True) as con:
        row = con.execute(
            "SELECT txn_id, ts FROM transactions ORDER BY ts, txn_id "
            "LIMIT 1 OFFSET (SELECT CAST(count(*) * ? AS BIGINT) FROM transactions)",
            [quantile],
        ).fetchone()
    assert row is not None
    txn_id, ts = row
    truncated = tmp_path / "truncated.duckdb"
    shutil.copy(featured, truncated)
    with duckdb.connect(str(truncated)) as con:
        con.execute("DELETE FROM transactions WHERE ts > ?", [ts])
    build_tabular_features(truncated)
    assert _row(truncated, txn_id) == _row(featured, txn_id)


def test_same_minute_siblings_are_not_visible(tmp_path: Path) -> None:
    db = tmp_path / "ties.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute(
            "CREATE TABLE transactions AS SELECT * FROM (VALUES "
            "(1, TIMESTAMP '2022-09-01 10:00', 'A', 'B', 100.00, 100.00),"
            "(2, TIMESTAMP '2022-09-01 10:00', 'A', 'C', 200.00, 200.00),"
            "(3, TIMESTAMP '2022-09-01 10:30', 'A', 'D', 300.00, 300.00)) "
            "v(txn_id, ts, from_key, to_key, amount_usd, amount_paid)"
        )
        con.execute(
            "ALTER TABLE transactions ADD COLUMN payment_format VARCHAR DEFAULT 'Wire';"
            "ALTER TABLE transactions ADD COLUMN currency_paid VARCHAR DEFAULT 'US Dollar';"
            "ALTER TABLE transactions ADD COLUMN currency_received VARCHAR DEFAULT 'US Dollar';"
            "ALTER TABLE transactions ADD COLUMN cross_currency BOOLEAN DEFAULT false;"
            "ALTER TABLE transactions ADD COLUMN cross_bank BOOLEAN DEFAULT true;"
            "ALTER TABLE transactions ADD COLUMN self_transfer BOOLEAN DEFAULT false;"
        )
    build_tabular_features(db)
    assert _row(db, 1)["out_cnt_1h"] == 1
    assert _row(db, 2)["out_cnt_1h"] == 1
    assert _row(db, 3)["out_cnt_1h"] == 3
    assert _row(db, 3)["out_sum_1h"] == pytest.approx(600.0)
