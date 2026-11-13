"""Account-day alerts and the capacity-based threshold (task 2.9 · R2.7, ADR-02.1, ADR-02.3).

An alert is one sending account on one day, scored with the maximum calibrated score of
its transactions that day. The threshold is not a probability chosen in the abstract: it
is the score at which the alert volume on validation matches what the analyst team can
review per day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CAPACITY_GRID = (50, 100, 150, 200, 300, 400, 600, 800, 1000)


def account_days(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction scores to account-days.

    Input columns: ``txn_id, from_key, day, score (calibrated), label, typology``.
    Output: one row per (from_key, day) with ``score`` = max, ``label`` = any laundering,
    ``n_txn``, ``n_laundering`` and ``top_txn_id`` (the transaction that set the score).
    """
    ordered = scored.sort_values(["from_key", "day", "score"], ascending=[True, True, False])
    grouped = ordered.groupby(["from_key", "day"], sort=False, observed=True)
    out = grouped.agg(
        score=("score", "first"),
        top_txn_id=("txn_id", "first"),
        n_txn=("txn_id", "size"),
        n_laundering=("label", "sum"),
    ).reset_index()
    out["label"] = out["n_laundering"] > 0
    return out


@dataclass(frozen=True)
class ThresholdChoice:
    capacity_per_day: int
    threshold: float
    days: float
    expected_alerts_per_day: float
    precision: float
    recall_account_days: float
    recall_transactions: float
    marginal_precision_next_50: float


def choose_threshold(
    validation: pd.DataFrame, capacity_per_day: int, n_laundering_txn: int, days: float
) -> ThresholdChoice:
    """Threshold so that validation yields ≈ ``capacity_per_day`` alerts per day.

    ``days`` is the validation window's length in days (fractional: it starts and ends
    mid-day), so the volume is per 24 h of activity rather than per calendar date.
    """
    ranked = validation.sort_values("score", ascending=False).reset_index(drop=True)
    k = min(len(ranked), max(1, round(capacity_per_day * days)))
    threshold = float(ranked.loc[k - 1, "score"])
    flagged = ranked.iloc[:k]
    next_50 = ranked.iloc[k : k + round(50 * days)]
    positives = int(ranked["label"].sum())
    return ThresholdChoice(
        capacity_per_day=capacity_per_day,
        threshold=threshold,
        days=days,
        expected_alerts_per_day=k / days,
        precision=float(flagged["label"].mean()),
        recall_account_days=float(flagged["label"].sum() / positives) if positives else 0.0,
        recall_transactions=(
            float(flagged["n_laundering"].sum() / n_laundering_txn) if n_laundering_txn else 0.0
        ),
        marginal_precision_next_50=float(next_50["label"].mean()) if len(next_50) else 0.0,
    )


def capacity_tradeoff(
    validation: pd.DataFrame, n_laundering_txn: int, days: float
) -> list[dict[str, Any]]:
    """What each analyst capacity buys: precision and recall per daily alert budget."""
    return [
        choose_threshold(validation, capacity, n_laundering_txn, days).__dict__
        for capacity in CAPACITY_GRID
    ]


def apply_threshold(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Alerts = account-days at or above the threshold, ranked by score within each day."""
    alerts = frame[frame["score"] >= threshold].copy()
    alerts["priority_rank"] = (
        alerts.groupby("day")["score"].rank(ascending=False, method="first").astype(np.int64)
    )
    return alerts.sort_values(["day", "priority_rank"]).reset_index(drop=True)
