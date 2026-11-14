"""Weekly operations report (task 4.8 · R4.5): ``reports/ops_report_<week>.md`` + 2 charts.

Observed values come from the alerts table and the audit trail. HI-Small has about five
days of model alerts, so earlier weeks and the 12-month history are **simulated** from the
observed days and labelled as such everywhere (ADR-01.3).
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from finint.core.audit import AuditLog
from finint.ml.report import INK, INK_2, SURFACE, _figure, _save, _style

OBSERVED, SIMULATED = "#2a78d6", "#b5b3ad"


def observed_daily(db_path: Path) -> dict[date, int]:
    """Model alerts per shifted day (forced evaluation alerts excluded)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        shift = con.execute(
            "SELECT any_value(date_diff('day', ts, ts_shifted)) FROM transactions"
        ).fetchone()
        days = int(shift[0]) if shift else 0
        rows = con.execute(
            "SELECT alert_date, count(*) FROM alerts "
            "WHERE split IN ('validation', 'test') GROUP BY 1"
        ).fetchall()
    return {d + timedelta(days=days): int(n) for d, n in rows}


def simulate_history(
    observed: dict[date, int], end: date, days: int, noise: float, seed: int
) -> dict[date, tuple[int, bool]]:
    """Daily counts for ``days`` days ending at ``end``; observed where available."""
    rng = random.Random(seed)  # noqa: S311 - seeded simulation, not security
    full_days = sorted(observed.values())[1:] or list(
        observed.values()
    )  # drop the partial first day
    out: dict[date, tuple[int, bool]] = {}
    for i in range(days):
        day = end - timedelta(days=days - 1 - i)
        if day in observed:
            out[day] = (observed[day], False)
        else:
            base = rng.choice(full_days)
            out[day] = (max(0, round(base * (1 + rng.uniform(-noise, noise)))), True)
    return out


def _week(day: date) -> tuple[int, int]:
    iso = day.isocalendar()
    return iso.year, iso.week


def _bar_chart(
    labels: list[str],
    observed: list[int],
    simulated: list[int],
    title: str,
    path: Path,
    rotate: int = 0,
) -> None:
    """Stacked bars: observed part (solid) and simulated part (hatched), never mixed up."""
    from matplotlib.patches import Patch

    fig, ax = _figure(width=8.5, height=4.2)
    x = np.arange(len(labels))
    ax.bar(x, observed, color=OBSERVED, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.bar(
        x,
        simulated,
        bottom=observed,
        color=SIMULATED,
        width=0.8,
        edgecolor="#8f8d87",
        hatch="///",
        linewidth=0,
    )
    step = max(1, len(labels) // 10)
    ax.set_xticks(x[::step], labels[::step], rotation=rotate)
    _style(ax, title)
    handles = [
        Patch(color=OBSERVED, label="Observed (HI-Small alerts)"),
        Patch(facecolor=SIMULATED, hatch="///", edgecolor="#8f8d87", label="SIMULATED"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        labelcolor=INK,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
    )
    ax.set_ylabel("Alerts", color=INK_2)
    _save(fig, path)
    plt.close("all")


def build_ops_report(
    db_path: Path, audit: AuditLog, as_of: date, cfg: dict[str, Any], figures: Path
) -> tuple[str, str]:
    observed = observed_daily(db_path)
    report_day = max(observed)
    year, week = _week(report_day)
    week_label = f"{year}-W{week:02d}"
    history = simulate_history(
        observed, as_of, 366, cfg["simulation"]["noise"], cfg["simulation"]["seed"]
    )
    this_week = [(n, s) for d, (n, s) in history.items() if _week(d) == (year, week)]
    observed_part = sum(n for n, s in this_week if not s)
    simulated_part = sum(n for n, s in this_week if s)
    prev = report_day - timedelta(days=7)
    prev_week = [(n, s) for d, (n, s) in history.items() if _week(d) == _week(prev)]
    volume, prev_volume = sum(n for n, _ in this_week), sum(n for n, _ in prev_week)
    prev_simulated = any(s for _, s in prev_week)

    with duckdb.connect(str(db_path), read_only=True) as con:
        shift_row = con.execute(
            "SELECT any_value(date_diff('day', ts, ts_shifted)) FROM transactions"
        ).fetchone()
        shift = int(shift_row[0]) if shift_row else 0
        precision_row = con.execute(
            "SELECT count(*), avg(l.is_laundering::INT) "
            "FROM alerts a JOIN alert_labels l USING (alert_id) "
            "WHERE a.split IN ('validation', 'test') "
            "AND a.alert_date + to_days($s) BETWEEN $a AND $b",
            {"s": shift, "a": report_day - timedelta(days=report_day.weekday()), "b": report_day},
        ).fetchone()
    reviewed, precision = (
        (int(precision_row[0]), float(precision_row[1] or 0)) if precision_row else (0, 0.0)
    )

    events = audit.events()
    completed = [e for e in events if e.event_type == "RUN_COMPLETED"]
    decisions = [e for e in events if e.event_type == "ANALYST_DECISION"]
    gathering = [float(e.payload.get("gathering_seconds") or 0) for e in completed]
    costs: list[tuple[str, float]] = [
        (e.occurred_at[:10], float(e.payload["cost_usd"]))
        for e in completed
        if e.payload.get("cost_usd") is not None
    ]
    agree = [
        e.payload.get("recommended_disposition") == e.payload.get("final_disposition")
        for e in decisions
    ]
    manual_min = float(cfg["manual_minutes_per_alert"])
    after_min = (float(np.mean(gathering)) / 60) if gathering else None
    after_median = (float(np.median(gathering)) / 60) if gathering else None
    by_day: dict[str, list[float]] = {}
    for day, cost in costs:
        by_day.setdefault(day, []).append(float(cost))

    daily = {d: v for d, v in history.items() if d > as_of - timedelta(days=30)}
    _bar_chart(
        [d.strftime("%d %b") for d in daily],
        [0 if sim else v for v, sim in daily.values()],
        [v if sim else 0 for v, sim in daily.values()],
        "Daily alerts, last 30 days",
        figures / "ops_daily_alerts.png",
        45,
    )
    months: dict[str, list[tuple[int, bool]]] = {}
    for d, v in history.items():
        months.setdefault(d.strftime("%Y-%m"), []).append(v)
    month_keys = sorted(months)[-12:]
    _bar_chart(
        [date.fromisoformat(m + "-01").strftime("%b %Y") for m in month_keys],
        [sum(n for n, sim in months[m] if not sim) for m in month_keys],
        [sum(n for n, sim in months[m] if sim) for m in month_keys],
        "Monthly alerts, last 12 months",
        figures / "ops_monthly_alerts.png",
        45,
    )

    change = (volume - prev_volume) / prev_volume if prev_volume else None
    rows = [
        (
            "Alerts this week",
            f"{volume:,}",
            f"{observed_part:,} observed"
            + (f" + {simulated_part:,} SIMULATED (days without data)" if simulated_part else ""),
        ),
        (
            "Change vs previous week",
            f"{change:+.0%}" if change is not None else "-",
            "previous week SIMULATED" if prev_simulated else "observed",
        ),
        (
            "Precision of reviewed alerts (confirmed laundering)",
            f"{precision:.0%} of {reviewed:,}",
            "observed (dataset labels as analyst outcome)",
        ),
        (
            "Evidence-gathering time per alert: before",
            f"{manual_min:.0f} min",
            "ASSUMPTION (config/ops.yaml)",
        ),
        (
            "Evidence-gathering time per alert: after (agent)",
            f"{after_min:.1f} min mean, {after_median:.1f} min median"
            if after_min is not None and after_median is not None
            else "-",
            f"observed over {len(gathering)} investigations; the mean includes runs that "
            "waited on rate limits or a sleeping machine",
        ),
        (
            "Agent-analyst disposition agreement",
            f"{np.mean(agree):.0%}" if agree else "-",
            f"observed over {len(agree)} decisions (simulated analyst = ground truth)",
        ),
        (
            "Cost per investigated alert",
            f"${np.mean([c for _, c in costs]):.4f}" if costs else "-",
            "observed, list prices (free tier in this PoC)",
        ),
    ]
    trend = [(day, f"${np.mean(v):.4f}", len(v)) for day, v in sorted(by_day.items())]
    text = "\n\n".join(
        [
            f"# FinCrime operations report: week {week_label}",
            f"Generated by `make report` on {as_of}. Alert dates are the dataset's timestamps "
            "shifted to the present (weekday-preserving). **Observed** figures come from the "
            "model's alerts and the audit trail; **SIMULATED** figures fill history the dataset "
            "does not have.",
            "## Key metrics",
            "| Metric | Value | Basis |\n|---|---|---|\n"
            + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows),
            "## Daily alerts, last 30 days",
            "![Daily alerts](figures/ops_daily_alerts.png)",
            "## Monthly alerts, last 12 months",
            "![Monthly alerts](figures/ops_monthly_alerts.png)",
            "## Cost per investigated alert, by day",
            "| Day | Mean cost | Investigations |\n|---|---|---|\n"
            + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in trend),
            "## Notes",
            "- Volume counts model alerts only (forced evaluation alerts are excluded).\n"
            "- The 'before' time is an assumption, not a measurement; the 'after' time is the "
            "agent's evidence-gathering time and excludes the analyst's review.\n"
            "- Precision uses dataset labels as the reviewed outcome. Agreement uses the "
            "evaluation's simulated analyst, who follows the label.",
        ]
    )
    return week_label, text + "\n"
