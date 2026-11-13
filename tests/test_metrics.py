"""Imbalanced-classification metrics (task 2.6 · R2.5)."""

from __future__ import annotations

import numpy as np
import pytest

from finint.ml.metrics import (
    budget_metrics,
    pr_auc,
    precision_recall_at_k,
    recall_by_group,
    top_k_mask,
)

LABELS = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])


def test_pr_auc_perfect_and_known_value() -> None:
    assert pr_auc(LABELS, LABELS.astype(float)) == pytest.approx(1.0)
    # positives at ranks 1, 3, 7 -> AP = (1/1 + 2/3 + 3/7) / 3
    assert pr_auc(LABELS, SCORES) == pytest.approx((1 + 2 / 3 + 3 / 7) / 3)


def test_top_k_mask_breaks_ties_by_position() -> None:
    scores = np.array([0.5, 0.9, 0.5, 0.5])
    assert top_k_mask(scores, 2).tolist() == [True, True, False, False]
    assert not top_k_mask(scores, 0).any()


@pytest.mark.parametrize(
    ("k", "precision", "recall"), [(1, 1.0, 1 / 3), (3, 2 / 3, 2 / 3), (10, 0.3, 1.0)]
)
def test_precision_recall_at_k(k: int, precision: float, recall: float) -> None:
    assert precision_recall_at_k(LABELS, SCORES, k) == pytest.approx((precision, recall))


def test_recall_by_group_reports_unlisted_positives() -> None:
    groups = ["CYCLE", None, float("nan"), None, None, None, "CYCLE", None, None, None]
    flagged = top_k_mask(SCORES, 3)
    assert recall_by_group(LABELS, flagged, groups) == {
        "CYCLE": (2, 0.5),
        "UNLISTED": (1, 1.0),
    }


def test_budget_metrics_rounds_k_from_fraction() -> None:
    out = budget_metrics(LABELS, SCORES, [0.1, 0.3])
    assert out["10.00%"] == {"k": 1, "precision": 1.0, "recall": pytest.approx(1 / 3)}
    assert out["30.00%"]["k"] == 3
