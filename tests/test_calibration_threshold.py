"""Calibration and capacity threshold (tasks 2.8, 2.9 · R2.6, R2.7)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finint.ml.calibration import IsotonicCalibrator, brier, reliability
from finint.ml.threshold import account_days, apply_threshold, choose_threshold


def test_isotonic_is_monotone_improves_brier_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    raw = rng.random(5000)
    labels = rng.random(5000) < raw**3  # true probability = raw^3: raw is overconfident
    cal = IsotonicCalibrator.fit(raw, labels)
    probs = cal.transform(raw)
    assert np.all(np.diff(cal.transform(np.linspace(0, 1, 50))) >= 0)
    assert brier(labels, probs) < brier(labels, raw)
    cal.save(tmp_path / "cal.json")
    again = IsotonicCalibrator.load(tmp_path / "cal.json")
    assert np.allclose(again.transform(raw), probs)


def test_calibrator_refuses_unknown_kind(tmp_path: Path) -> None:
    (tmp_path / "c.json").write_text('{"kind": "pickle", "x": [], "y": []}', "utf-8")
    with pytest.raises(ValueError, match="unexpected calibrator kind"):
        IsotonicCalibrator.load(tmp_path / "c.json")


def test_reliability_bins_cover_all_rows() -> None:
    probs = np.array([0.00005, 0.0005, 0.03, 0.5, 0.95, 1.0])
    rows = reliability(np.array([0, 0, 0, 1, 1, 1], dtype=bool), probs)
    assert sum(r["n"] for r in rows) == len(probs)


@pytest.fixture
def scored() -> pd.DataFrame:
    # 3 accounts x 2 days; account B launders on day 1.
    return pd.DataFrame(
        {
            "txn_id": [1, 2, 3, 4, 5, 6, 7],
            "from_key": ["A", "A", "B", "B", "C", "A", "B"],
            "day": [date(2022, 9, 1)] * 5 + [date(2022, 9, 2)] * 2,
            "score": [0.1, 0.4, 0.9, 0.2, 0.05, 0.3, 0.6],
            "label": [False, False, True, True, False, False, False],
        }
    )


def test_account_days_take_max_score_and_any_label(scored: pd.DataFrame) -> None:
    ad = account_days(scored).set_index(["from_key", "day"])
    b1 = ad.loc[("B", date(2022, 9, 1))]
    assert b1["score"] == 0.9
    assert b1["top_txn_id"] == 3
    assert b1["n_laundering"] == 2
    assert bool(b1["label"]) is True
    assert len(ad) == 5


def test_threshold_matches_capacity_per_day(scored: pd.DataFrame) -> None:
    ad = account_days(scored)
    choice = choose_threshold(ad, capacity_per_day=1, n_laundering_txn=2, days=2.0)
    assert choice.expected_alerts_per_day == 1.0
    assert choice.threshold == 0.6  # top-2 account-days over 2 days: 0.9 and 0.6
    assert choice.precision == 0.5
    assert choice.recall_transactions == 1.0
    alerts = apply_threshold(ad, choice.threshold)
    assert alerts["priority_rank"].tolist() == [1, 1]
