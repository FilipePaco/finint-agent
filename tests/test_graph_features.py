"""Graph features on sliding windows (task 2.4 · R2.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from finint.ml.graph_features import GRAPH_FEATURES, build_graph_features

# A -> B -> C -> A is a 3-cycle on day 1; D fans out to 4 accounts on day 1;
# E <-> F is a 2-cycle; on day 2 only "probe" transactions exist, so day-2 features
# come entirely from day 1. A day-2 edge G -> H must not appear in day-2 features.
EDGES = [
    (1, "2022-09-01 09:00", "1:A", "1:B", 100.0),
    (2, "2022-09-01 10:00", "1:B", "1:C", 90.0),
    (3, "2022-09-01 11:00", "1:C", "1:A", 80.0),
    (4, "2022-09-01 09:00", "1:D", "1:W", 10.0),
    (5, "2022-09-01 09:01", "1:D", "1:X", 10.0),
    (6, "2022-09-01 09:02", "1:D", "1:Y", 10.0),
    (7, "2022-09-01 09:03", "1:D", "1:Z", 10.0),
    (8, "2022-09-01 12:00", "1:E", "1:F", 50.0),
    (9, "2022-09-01 13:00", "1:F", "1:E", 50.0),
    (10, "2022-09-02 08:00", "1:A", "1:D", 5.0),
    (11, "2022-09-02 08:00", "1:G", "1:H", 5.0),
    (12, "2022-09-02 09:00", "1:E", "2:E2", 5.0),
]


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "g.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute(
            "CREATE TABLE transactions (txn_id BIGINT, ts TIMESTAMP, from_key VARCHAR, "
            "to_key VARCHAR, amount_usd DECIMAL(38, 2))"
        )
        con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?)", EDGES)
        keys = sorted({e[2] for e in EDGES} | {e[3] for e in EDGES})
        # E and E2 belong to the same entity (two banks).
        con.execute("CREATE TABLE accounts (account_key VARCHAR, entity_id VARCHAR)")
        con.executemany(
            "INSERT INTO accounts VALUES (?, ?)",
            [(k, "ENT-E" if k in ("1:E", "2:E2") else k) for k in keys],
        )
    build_graph_features(path, window_days=3, log=lambda _m: None)
    return path


def _features(db: Path, txn_id: int) -> dict[str, Any]:
    with duckdb.connect(str(db), read_only=True) as con:
        cursor = con.execute("SELECT * FROM features_graph WHERE txn_id = ?", [txn_id])
        names = [d[0] for d in cursor.description]
        row = cursor.fetchone()
    assert row is not None
    return dict(zip(names, row, strict=True))


def test_one_row_per_transaction_with_all_columns(db: Path) -> None:
    with duckdb.connect(str(db), read_only=True) as con:
        n = con.execute("SELECT count(*) FROM features_graph").fetchone()
        cols = {r[0] for r in con.execute("DESCRIBE features_graph").fetchall()}
    assert n == (len(EDGES),)
    assert set(GRAPH_FEATURES) <= cols


def test_first_day_has_no_history(db: Path) -> None:
    features = _features(db, 1)
    assert all(features[c] == 0 for c in GRAPH_FEATURES)


def test_cycles_fan_out_and_flow_from_previous_days(db: Path) -> None:
    probe = _features(db, 10)  # A -> D on day 2
    assert probe["src_in_cycle_3"] == 1
    assert probe["src_in_cycle_2"] == 0
    assert probe["dst_out_degree"] == 4
    assert probe["dst_fan_out_ratio"] == pytest.approx(4.0)
    assert probe["src_flow_through"] == pytest.approx(100.0 / 80.0)
    assert probe["src_pagerank"] > 0


def test_same_day_edges_are_not_used(db: Path) -> None:
    probe = _features(db, 11)  # G -> H exists only on day 2
    assert probe["src_out_degree"] == 0
    assert probe["dst_in_degree"] == 0


def test_two_cycle_and_entity_aggregation(db: Path) -> None:
    probe = _features(db, 12)  # E -> E2 on day 2
    assert probe["src_in_cycle_2"] == 1
    assert probe["src_entity_out_degree"] == 1
    assert probe["src_entity_n_accounts_active"] == 1  # E2 had no activity on day 1
