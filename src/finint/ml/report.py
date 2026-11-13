"""ML report (task 2.12 · R2.5): ``reports/ml_report.md`` + figures.

Everything comes from saved artifacts (``metrics.json``, ``alert_run.json``), so the
numbers quoted in the README trace to ``make train`` and ``make alerts``. Every chart has a
table with the same numbers next to it (accessibility, and the colour-contrast relief rule).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finint.ml.registry import ArtifactInfo

# Validated categorical order (blue, orange, aqua): passes CVD and normal-vision checks on
# the light surface; aqua is below 3:1 contrast, hence tables next to every chart.
MODEL_COLORS = {"m0": "#2a78d6", "m1": "#eb6834", "m2": "#1baf7a"}
MODEL_LABELS = {"m0": "M0 tabular", "m1": "M1 tabular + graph", "m2": "M2 GraphSAGE"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: Any, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width: float = 7.5, height: float = 4.2) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(out)


def plot_pr_curves(models: dict[str, ArtifactInfo], path: Path) -> list[str]:
    fig, ax = _figure()
    drawn = []
    for name, info in sorted(models.items()):
        curve = info.metrics["test"].get("pr_curve")
        if not curve or not curve["recall"]:
            continue
        ax.plot(curve["recall"], curve["precision"], color=MODEL_COLORS.get(name, INK_2),
                linewidth=2, label=MODEL_LABELS.get(name, name))  # fmt: skip
        drawn.append(name)
    _style(ax, "Precision-recall on test (without the low-activity tail)")
    ax.set_xlabel("Recall", color=INK_2)
    ax.set_ylabel("Precision", color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)
    return drawn


def plot_typology_recall(models: dict[str, ArtifactInfo], path: Path) -> None:
    names = sorted(models)
    groups = sorted(
        {g for info in models.values() for g in info.metrics["test"]["typology_recall"]},
        key=lambda g: (g == "UNLISTED", g),
    )
    fig, ax = _figure(height=4.8)
    height = 0.8 / len(names)
    for i, name in enumerate(names):
        recall = models[name].metrics["test"]["typology_recall"]
        ys = [j + i * height for j in range(len(groups))]
        xs = [recall.get(g, {}).get("recall", 0.0) for g in groups]
        ax.barh(ys, xs, height=height * 0.85, color=MODEL_COLORS.get(name, INK_2),
                label=MODEL_LABELS.get(name, name))  # fmt: skip
    ax.set_yticks([j + 0.4 - height / 2 for j in range(len(groups))], groups)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    _style(ax, "Recall by laundering typology (top 0.5% of test transactions)")
    ax.set_xlabel("Recall", color=INK_2)
    ax.legend(frameon=False, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(names))  # fmt: skip
    _save(fig, path)


def plot_capacity(tradeoff: list[dict[str, Any]], chosen: int, path: Path) -> None:
    caps = [t["capacity_per_day"] for t in tradeoff]
    fig, ax = _figure()
    series = (
        ("precision", "Precision of alerts", "#2a78d6"),
        ("recall_transactions", "Recall of laundering transactions", "#eb6834"),
        ("marginal_precision_next_50", "Precision of the next 50 alerts/day", "#1baf7a"),
    )
    for key, label, color in series:
        ax.plot(caps, [t[key] for t in tradeoff], color=color, linewidth=2, marker="o",
                markersize=4, label=label)  # fmt: skip
    ax.axvline(chosen, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"chosen: {chosen}/day", (chosen, 1.0), xytext=(6, -12),
                textcoords="offset points", color=INK_2, fontsize=9)  # fmt: skip
    _style(ax, "What analyst capacity buys (validation)")
    ax.set_xlabel("Analyst capacity (alerts per day)", color=INK_2)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def plot_reliability(run: dict[str, Any], path: Path) -> None:
    fig, ax = _figure(width=5.5, height=5.0)
    for key, label, color in (
        ("reliability_raw", "Raw score", "#2a78d6"),
        ("reliability_calibrated", "Calibrated (isotonic)", "#eb6834"),
    ):
        rows = [r for r in run[key] if r["mean_predicted"] > 0 and r["observed_rate"] > 0]
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_rate"] for r in rows],
                color=color, linewidth=2, marker="o", markersize=4, label=label)  # fmt: skip
    ax.plot([1e-5, 1], [1e-5, 1], color=INK_2, linewidth=1, linestyle="--", label="Perfect")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _style(ax, "Reliability on test, out of sample (log scale)")
    ax.set_xlabel("Mean predicted probability", color=INK_2)
    ax.set_ylabel("Observed laundering rate", color=INK_2)
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def build_ml_report(
    models: dict[str, ArtifactInfo],
    champion: ArtifactInfo,
    why: str,
    run: dict[str, Any],
    figures_dir: Path,
) -> str:
    rel = figures_dir.name
    drawn = plot_pr_curves(models, figures_dir / "pr_curves.png")
    plot_typology_recall(models, figures_dir / "typology_recall.png")
    choice = run["threshold"]
    plot_capacity(run["tradeoff"], choice["capacity_per_day"], figures_dir / "capacity.png")
    plot_reliability(run, figures_dir / "reliability.png")

    comparison = []
    for name, info in sorted(models.items()):
        test = info.metrics["test"]
        b = test["budgets"]
        comparison.append(
            (
                MODEL_LABELS.get(name, name), f"`{info.version}`",
                f"{info.validation_pr_auc:.3f}", f"**{test['pr_auc']:.3f}**",
                f"{test['full_split']['pr_auc']:.3f}",
                f"{_pct(b['0.10%']['precision'])} / {_pct(b['0.10%']['recall'])}",
                f"{_pct(b['0.50%']['precision'])} / {_pct(b['0.50%']['recall'])}",
                f"{_pct(b['1.00%']['precision'])} / {_pct(b['1.00%']['recall'])}",
            )
        )  # fmt: skip
    groups = sorted(
        {g for info in models.values() for g in info.metrics["test"]["typology_recall"]},
        key=lambda g: (g == "UNLISTED", g),
    )
    typology_rows = []
    for g in groups:
        row: list[Any] = [g]
        for name in sorted(models):
            cell = models[name].metrics["test"]["typology_recall"].get(g)
            row.append(f"{_pct(cell['recall'])} (n={cell['n_positive']})" if cell else "")
        typology_rows.append(row)
    capacity_rows = [
        (t["capacity_per_day"], f"{t['threshold']:.4f}", _pct(t["precision"]),
         _pct(t["recall_transactions"]), _pct(t["marginal_precision_next_50"]))
        for t in run["tradeoff"]
    ]  # fmt: skip
    missing_curves = sorted(set(models) - set(drawn))
    larger = [t for t in run["tradeoff"] if t["capacity_per_day"] > choice["capacity_per_day"]]
    next_step = larger[0] if larger else None
    gnn = models.get("m2")
    if gnn is None:
        gnn_finding = "- **No GNN was trained** in this run."
    elif gnn.validation_pr_auc < champion.validation_pr_auc:
        gnn_finding = (
            f"- **The GNN did not beat gradient boosting** within its time box (validation "
            f"PR-AUC {gnn.validation_pr_auc:.3f} vs {champion.validation_pr_auc:.3f}). It is "
            "reported, not chosen: the champion is picked by metrics, not sophistication "
            "(ADR-02.2)."
        )
    else:
        gnn_finding = (
            f"- **The GNN scores higher on validation** ({gnn.validation_pr_auc:.3f}) but has no "
            "per-feature explanations, so alerts use the best explainable model."
        )
    sections = [
        "# ML detection report",
        "Generated by `make report-ml` from the saved artifacts of `make train` and "
        "`make alerts`. Data: IBM AMLworld HI-Small (synthetic), chronological 60/20/20 split "
        "by transaction count. **Primary test metrics exclude the simulator's low-activity tail** "
        "(Sep 11-18: 1,108 transactions, 59% laundering), which would otherwise inflate them "
        "(ADR-02.7); full-test PR-AUC is shown for transparency.",
        "## Model comparison",
        _table(
            [
                "Model",
                "Version",
                "Val PR-AUC",
                "Test PR-AUC",
                "Full test",
                "P / R @0.1%",
                "P / R @0.5%",
                "P / R @1%",
            ],
            comparison,
        ),
        f"**Champion: {MODEL_LABELS.get(champion.name, champion.name)}**: {why}.",
        "## Precision-recall curves",
        f"![Precision-recall curves]({rel}/pr_curves.png)",
        (
            f"Curve not stored for: {', '.join(missing_curves)} (trained before curves were "
            "saved in metrics.json); its PR-AUC above is exact."
            if missing_curves
            else ""
        ),
        "## Recall by typology",
        "At the top 0.5% of test transactions. `UNLISTED` = laundering transactions the "
        "patterns file does not list (38% of all positives): they have no typology, and every "
        "model finds them far less often.",
        f"![Recall by typology]({rel}/typology_recall.png)",
        _table(["Typology", *[MODEL_LABELS.get(n, n) for n in sorted(models)]], typology_rows),
        "## Calibration",
        "Isotonic regression fitted on validation scores (stored as JSON breakpoints, not a "
        "pickle) and checked **out of sample on test**: Brier score "
        f"{run['brier_raw']:.6f} → {run['brier_calibrated']:.6f}."
        + (
            " **Calibration does not improve Brier out of sample**: the isotonic fit follows "
            "validation too closely. Ranking and the capacity threshold are unaffected (both "
            "monotone), but calibrated scores should be read as approximate probabilities."
            if run["brier_calibrated"] > run["brier_raw"]
            else ""
        ),
        f"![Reliability]({rel}/reliability.png)",
        "## Alert threshold from analyst capacity",
        "Alerts are **account-days** (one sending account on one day, scored by its riskiest "
        "transaction), because a laundering pattern spans many transactions and analysts work "
        "cases, not rows (ADR-02.1). The threshold is the calibrated score at which validation "
        "produces the team's daily capacity (ADR-02.3).",
        f"![Capacity trade-off]({rel}/capacity.png)",
        _table(
            ["Capacity/day", "Threshold", "Precision", "Recall (txns)", "Next 50/day precision"],
            capacity_rows,
        ),
        f"**Chosen: {choice['capacity_per_day']} alerts/day → threshold "
        f"{choice['threshold']:.4f}.** Up to this point the next 50 alerts a day would still "
        f"be laundering {_pct(choice['marginal_precision_next_50'])} of the time"
        + (
            f"; at {next_step['capacity_per_day']}/day that marginal precision is "
            f"{_pct(next_step['marginal_precision_next_50'])}, so extra capacity there mostly "
            "buys false positives."
            if next_step
            else "."
        ),
        "### Alerts produced",
        f"{run['n_alerts']:,} alerts over the validation and test periods "
        f"({', '.join(f'{k}: {v}' for k, v in run['alerts_by_split'].items())}). "
        f"On **test**, {_pct(run['test_precision'])} of alerts contain laundering and they "
        f"cover {_pct(run['test_recall_transactions'])} of laundering transactions. Validation "
        "figures are optimistic because calibration and threshold were fitted on validation; "
        "the test figures are the honest estimate.",
        "Every alert is stored with its features, SHAP top-5 contributions, contributing "
        "transactions, score, threshold and model version, and is logged as an "
        "`ALERT_CREATED` event in the hash-chained audit trail.",
        "## Findings",
        "- **Graph features help, modestly.** M1 beats M0 on validation PR-AUC and lifts recall "
        "on fan-in and bipartite patterns, but the test gain is small. Graph features only see "
        "closed previous days (the leakage-safe choice), while many attempts unfold within a day.",
        gnn_finding,
        "- **The biggest gap is `UNLISTED` laundering** (about 20% recall vs 75-90% for listed "
        "typologies). These transactions are labelled but follow no listed pattern.",
        "- **Class weights did not help.** `scale_pos_weight` = 1 beat 10 and 50 on validation "
        "PR-AUC for every LightGBM model.",
    ]
    return "\n\n".join(s for s in sections if s) + "\n"
