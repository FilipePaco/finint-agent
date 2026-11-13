"""Model-agnostic evaluation from scores (R2.5, ADR-02.7).

Shared by LightGBM and the GNN. Imports neither LightGBM nor scikit-learn (see metrics.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from finint.ml.metrics import budget_metrics, pr_auc, recall_by_group, top_k_mask

BUDGETS = (0.001, 0.005, 0.01)
TYPOLOGY_BUDGET = 0.005


def evaluate_scores(scored: pd.DataFrame, split: str = "test") -> dict[str, Any]:
    """Shared by every model (LightGBM and GNN): columns score, label, in_tail, typology."""
    core = scored[~scored["in_tail"]]
    metrics = _metrics(core, split)
    full = _metrics(scored, split)
    metrics["excludes_tail"] = bool(len(core) < len(scored))
    metrics["full_split"] = {k: full[k] for k in ("n", "n_positive", "pr_auc", "budgets")}
    return metrics


def _metrics(part: pd.DataFrame, split: str) -> dict[str, Any]:
    labels = part["label"].to_numpy(bool)
    scores = part["score"].to_numpy(np.float64)
    k = max(1, round(len(part) * TYPOLOGY_BUDGET))
    flagged = top_k_mask(scores, k)
    return {
        "split": split,
        "n": len(part),
        "n_positive": int(labels.sum()),
        "pr_auc": pr_auc(labels, scores),
        "budgets": budget_metrics(labels, scores, BUDGETS),
        "typology_recall_at": f"{TYPOLOGY_BUDGET:.2%}",
        "typology_recall": {
            g: {"n_positive": n, "recall": r}
            for g, (n, r) in recall_by_group(labels, flagged, part["typology"].tolist()).items()
        },
    }
