"""Imbalanced-classification metrics (task 2.6 · R2.5).

Accuracy and ROC-AUC are not used for decisions: with a 0.1% positive rate they reward
models that are useless at the top of the ranking, where analysts actually work.

Pure numpy on purpose: this module is imported by the GNN process, which must not load
scikit-learn's or LightGBM's OpenMP runtime next to PyTorch's (segfault on macOS).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.bool_]


def pr_auc(labels: Labels, scores: Array) -> float:
    """Average precision, same definition as ``sklearn.metrics.average_precision_score``.

    Sum over distinct score thresholds of (recall gain) x (precision at that threshold).
    """
    y = np.asarray(labels, dtype=bool)
    if not y.any():
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    hits = np.cumsum(y[order])
    # last index of each distinct score value (ties share one threshold)
    last = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tps = hits[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    gains = np.diff(np.r_[0.0, recall])
    return float(np.sum(gains * precision))


def top_k_mask(scores: Array, k: int) -> Labels:
    """Mask of the ``k`` highest scores (ties broken by position, deterministically)."""
    if k <= 0:
        return np.zeros(len(scores), dtype=bool)
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:k]] = True
    return mask


def precision_recall_at_k(labels: Labels, scores: Array, k: int) -> tuple[float, float]:
    mask = top_k_mask(scores, k)
    hits = int(labels[mask].sum())
    positives = int(labels.sum())
    precision = hits / k if k else 0.0
    recall = hits / positives if positives else 0.0
    return precision, recall


def recall_by_group(
    labels: Labels, flagged: Labels, groups: Sequence[object]
) -> dict[str, tuple[int, float]]:
    """Recall within each group of positives: ``{group: (n_positives, recall)}``.

    Positives without a group are reported as ``UNLISTED`` (laundering transactions that the
    patterns file does not list). Missing values may arrive as ``None`` or pandas ``NaN``.
    """
    out: dict[str, list[int]] = {}
    for is_pos, hit, group in zip(labels, flagged, groups, strict=True):
        if not is_pos:
            continue
        name = group if isinstance(group, str) and group else "UNLISTED"
        counts = out.setdefault(name, [0, 0])
        counts[0] += 1
        counts[1] += int(hit)
    return {g: (n, hits / n) for g, (n, hits) in sorted(out.items())}


def budget_metrics(
    labels: Labels, scores: Array, budgets: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Precision and recall when flagging the top ``budget`` fraction of transactions."""
    out = {}
    for budget in budgets:
        k = max(1, round(len(scores) * budget))
        precision, recall = precision_recall_at_k(labels, scores, k)
        out[f"{budget:.2%}"] = {"k": k, "precision": precision, "recall": recall}
    return out
