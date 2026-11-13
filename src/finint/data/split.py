"""Chronological split by cumulative transaction count (ADR-02.6 · R2.1).

Shared by spec 01 (injected test cases go to entities active in the test window) and
spec 02 (model training and evaluation). Transactions at a boundary timestamp belong to
the later split.
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


def split_boundaries(con: duckdb.DuckDBPyConnection, fractions: SplitFractions) -> SplitBoundaries:
    row = con.execute(
        "SELECT quantile_disc(ts, $q1), quantile_disc(ts, $q2) FROM transactions",
        {"q1": fractions.train, "q2": fractions.train + fractions.validation},
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("transactions table is empty")
    return SplitBoundaries(validation_start=row[0], test_start=row[1])
