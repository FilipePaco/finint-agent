"""Chronological split by cumulative transaction count (ADR-02.6 · R2.1).

Shared by spec 01 (injected test cases go to entities active in the test window) and
spec 02 (model training and evaluation). Transactions at a boundary timestamp belong to
the later split.

``tail_start`` marks the simulator's low-activity tail (HI-Small: Sep 11-18, ~1.1k rows,
59% laundering). Background activity stops there while laundering attempts finish, so
including it inflates test metrics. Primary metrics exclude it; full-test metrics are
reported alongside (ADR-02.7).
"""

from __future__ import annotations

from datetime import datetime

import duckdb
from pydantic import BaseModel, model_validator


class SplitFractions(BaseModel):
    train: float
    validation: float
    test: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitFractions:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9 or min(self.train, self.validation, self.test) <= 0:
            raise ValueError(f"split fractions must be positive and sum to 1, got {total}")
        return self


class SplitBoundaries(BaseModel):
    validation_start: datetime
    test_start: datetime
    # Start of the low-activity tail (ADR-02.7): days whose volume is below
    # ``TAIL_MIN_SHARE`` of the median day. None if the data has no such tail.
    tail_start: datetime | None = None


TAIL_MIN_SHARE = 0.01

_TAIL = """
WITH daily AS (SELECT ts::DATE AS day, count(*) AS n FROM transactions GROUP BY 1),
active AS (SELECT max(day) AS last_active FROM daily
           WHERE n >= $share * (SELECT median(n) FROM daily))
SELECT CASE WHEN (SELECT max(day) FROM daily) > last_active
            THEN CAST(last_active + INTERVAL 1 DAY AS TIMESTAMP) END
FROM active
"""


def split_boundaries(con: duckdb.DuckDBPyConnection, fractions: SplitFractions) -> SplitBoundaries:
    row = con.execute(
        "SELECT quantile_disc(ts, $q1), quantile_disc(ts, $q2) FROM transactions",
        {"q1": fractions.train, "q2": fractions.train + fractions.validation},
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("transactions table is empty")
    tail = con.execute(_TAIL, {"share": TAIL_MIN_SHARE}).fetchone()
    return SplitBoundaries(
        validation_start=row[0], test_start=row[1], tail_start=tail[0] if tail else None
    )
