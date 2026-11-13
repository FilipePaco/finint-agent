"""Chronological split by transaction count (R2.1, ADR-02.6)."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pytest
from pydantic import ValidationError

from finint.data.split import SplitFractions, split_boundaries


def test_boundaries_follow_transaction_count_not_days() -> None:
    con = duckdb.connect()
    start = datetime(2022, 9, 1)
    # 90 transactions on day 1, 10 spread over days 2-10: a day-based split would differ.
    stamps = [start + timedelta(minutes=i) for i in range(90)]
    stamps += [start + timedelta(days=d) for d in range(1, 11)]
    con.execute("CREATE TABLE transactions (ts TIMESTAMP)")
    con.executemany("INSERT INTO transactions VALUES (?)", [(s,) for s in stamps])
    b = split_boundaries(con, SplitFractions(train=0.6, validation=0.2, test=0.2))
    assert b.validation_start == start + timedelta(minutes=59)
    assert b.test_start == start + timedelta(minutes=79)


@pytest.mark.parametrize(("train", "val", "test"), [(0.6, 0.2, 0.1), (0.8, 0.2, 0.0)])
def test_fractions_must_be_positive_and_sum_to_one(train: float, val: float, test: float) -> None:
    with pytest.raises(ValidationError):
        SplitFractions(train=train, validation=val, test=test)


def test_tail_start_marks_low_activity_days() -> None:
    con = duckdb.connect()
    start = datetime(2022, 9, 1)
    stamps = [start + timedelta(days=d, minutes=m) for d in range(5) for m in range(500)]
    stamps += [start + timedelta(days=5 + d) for d in range(3)]  # 1 row/day: the tail
    con.execute("CREATE TABLE transactions (ts TIMESTAMP)")
    con.executemany("INSERT INTO transactions VALUES (?)", [(s,) for s in stamps])
    b = split_boundaries(con, SplitFractions(train=0.6, validation=0.2, test=0.2))
    assert b.tail_start == datetime(2022, 9, 6)


def test_no_tail_when_activity_is_even() -> None:
    con = duckdb.connect()
    start = datetime(2022, 9, 1)
    con.execute("CREATE TABLE transactions (ts TIMESTAMP)")
    con.executemany(
        "INSERT INTO transactions VALUES (?)",
        [(start + timedelta(hours=h),) for h in range(24 * 5)],
    )
    b = split_boundaries(con, SplitFractions(train=0.6, validation=0.2, test=0.2))
    assert b.tail_start is None
