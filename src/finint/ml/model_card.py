"""Model card for the champion (task 2.13 · R2.8), written next to its artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from finint.ml.registry import ArtifactInfo


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_model_card(
    champion: ArtifactInfo, why: str, run: dict[str, Any], split: dict[str, Any]
) -> str:
    test = champion.metrics["test"]
    budgets = test["budgets"]
    choice = run["threshold"]
    lines = [
        f"# Model card: `{champion.version}`",
        "",
        "## Intended use",
        "Rank transactions by laundering risk and raise **account-day alerts** for a FinCrime "
        "analyst queue. The model never takes action: alerts feed an investigation agent and a "
        "human analyst makes every decision.",
        "",
        "## Data",
        "IBM AMLworld HI-Small (synthetic). Chronological 60/20/20 split by transaction count: "
        f"validation from {split['validation_start']}, test from {split['test_start']}. "
        "Primary test metrics exclude the low-activity tail from "
        f"{split['tail_start']} (ADR-02.7).",
        "",
        "## Selection",
        f"Champion: **{champion.name}**, {why}.",
        "",
        "## Test performance (without the tail)",
        f"- PR-AUC: **{test['pr_auc']:.3f}** (full test {test['full_split']['pr_auc']:.3f})",
    ]
    for budget, row in budgets.items():
        lines.append(
            f"- Top {budget} of transactions: precision {_pct(row['precision'])}, "
            f"recall {_pct(row['recall'])}"
        )
    lines += ["", f"Recall by typology at the top {test['typology_recall_at']}:", ""]
    lines += ["| Typology | Positives | Recall |", "|---|---|---|"]
    lines += [
        f"| {g} | {v['n_positive']} | {_pct(v['recall'])} |"
        for g, v in test["typology_recall"].items()
    ]
    lines += [
        "",
        "## Calibration and threshold",
        f"- Isotonic calibration on validation: Brier {run['brier_raw']:.5f} → "
        f"{run['brier_calibrated']:.5f}.",
        f"- Analyst capacity **{choice['capacity_per_day']} alerts/day** → threshold "
        f"{choice['threshold']:.4f} on the calibrated score.",
        f"- Validation at that threshold: precision {_pct(choice['precision'])}, recall "
        f"{_pct(choice['recall_transactions'])} of laundering transactions, marginal precision "
        f"of the next 50 alerts/day {_pct(choice['marginal_precision_next_50'])}.",
        f"- Test alerts: precision {_pct(run['test_precision'])}, recall "
        f"{_pct(run['test_recall_transactions'])} of laundering transactions.",
        "",
        "## Limitations",
        "- Synthetic data with about ten active days; patterns and rates may not transfer.",
        "- Graph features use only closed previous days, so same-day structure is seen only "
        "through the tabular windows.",
        f"- {_pct(test['typology_recall'].get('UNLISTED', {}).get('recall', 0.0))} recall on "
        "laundering not listed in the patterns file: the model is weakest on unlabelled "
        "typologies.",
        "- Calibration and threshold are both fitted on validation, so validation precision is "
        "optimistic; the test figures are the honest estimate.",
        "- Calibrated on validation; a prior shift in production requires recalibration.",
    ]
    return "\n".join(lines) + "\n"


def write_model_card(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
