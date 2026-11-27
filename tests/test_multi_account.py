"""Multi-account typology indicators on a hand-built graph (task 3.23 · R3.15 · ADR-03.11)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pytest

from finint.tools.investigation import multi_account_indicators

T0 = datetime(2022, 9, 5, 8, 0)
S, E = datetime(2022, 9, 1), datetime(2022, 9, 12)


def _edges() -> list[tuple[str, str, float, datetime]]:
    e: list[tuple[str, str, float, datetime]] = []
    # Scatter-gather: S scatters to M1..M5, each re-gathers on D a day later.
    for i in range(1, 6):
        e.append(("1:S", f"1:M{i}", 1000.0 + i, T0 + timedelta(hours=i)))
        e.append((f"1:M{i}", "1:D", 990.0 + i, T0 + timedelta(days=1, hours=i)))
    # Cycle A -> B -> C -> A with similar, slowly decreasing amounts; B sits mid-cycle.
    e += [
        ("2:A", "2:B", 500.0, T0),
        ("2:B", "2:C", 490.0, T0 + timedelta(hours=10)),
        ("2:C", "2:A", 480.0, T0 + timedelta(hours=20)),
    ]
    # Stack: X -> Y -> Z -> W, each layer forwards a similar amount within a day.
    e += [
        ("3:X", "3:Y", 7000.0, T0),
        ("3:Y", "3:Z", 6900.0, T0 + timedelta(hours=12)),
        ("3:Z", "3:W", 6800.0, T0 + timedelta(hours=30)),
    ]
    # Fan-in: F1..F12 all pay H.
    e += [(f"4:F{i}", "4:H", 300.0 * i, T0 + timedelta(hours=i)) for i in range(1, 13)]
    return e


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Any:
    path: Path = tmp_path_factory.mktemp("multi") / "g.duckdb"
    with duckdb.connect(str(path)) as w:
        w.execute(
            "CREATE TABLE transactions (from_key VARCHAR, to_key VARCHAR, "
            "amount_usd DECIMAL(38,2), ts TIMESTAMP)"
        )
        w.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?)", _edges())
    c = duckdb.connect(str(path), read_only=True)
    yield c
    c.close()


def test_scatter_gather_seen_from_the_source(con: Any) -> None:
    m = multi_account_indicators(con, "1:S", S, E)["scatter_gather_source"]
    assert m["converging_intermediaries"] == 5
    assert m["destination"] == "1:D"
    assert m["share_of_outflow_converging"] == 1.0


def test_scatter_gather_seen_from_an_intermediary(con: Any) -> None:
    """An intermediary alone looks like 1 in / 1 out; two hops show 5 parallel siblings."""
    m = multi_account_indicators(con, "1:M3", S, E)["scatter_gather_intermediary"]
    assert m == {"parallel_intermediaries": 5, "source": "1:S", "destination": "1:D"}


def test_cycle_through_an_account_in_the_middle(con: Any) -> None:
    """B neither starts nor ends the cycle in time, yet the 3-hop cycle is found."""
    chain = multi_account_indicators(con, "2:B", S, E)["amount_matched_chain"]
    assert chain["cycle_length"] == 3


def test_stack_depth_and_no_cycle(con: Any) -> None:
    chain = multi_account_indicators(con, "3:X", S, E)["amount_matched_chain"]
    assert chain["forward_depth"] == 3
    assert chain["cycle_length"] is None
    assert multi_account_indicators(con, "3:Z", S, E)["amount_matched_chain"]["backward_depth"] == 2


def test_fan_in_seen_from_a_feeder(con: Any) -> None:
    m = multi_account_indicators(con, "4:F1", S, E)["fan_in_feeder"]
    assert m == {"receiver_distinct_senders": 12, "receiver": "4:H"}


def test_unrelated_account_has_no_multi_account_shape(con: Any) -> None:
    m = multi_account_indicators(con, "4:H", S, E)
    assert m["scatter_gather_source"]["converging_intermediaries"] == 0
    assert m["scatter_gather_intermediary"]["parallel_intermediaries"] == 0
    assert m["amount_matched_chain"] == {
        "forward_depth": 0,
        "backward_depth": 1,
        "cycle_length": None,
    }
