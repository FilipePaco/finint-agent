"""Alert generation and logging (tasks 2.8-2.10 · R2.6-R2.8).

Pipeline for the champion LightGBM model:

1. score validation and the alert period (validation start → low-activity tail);
2. fit an isotonic calibrator on validation;
3. aggregate to account-days and pick the threshold from analyst capacity;
4. write ``alerts`` (one row per alert, with features, SHAP top-5, model version and
   threshold) and ``alert_labels`` (ground truth, LLM-forbidden), and log one
   ``ALERT_CREATED`` audit event per alert.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from finint.core.audit import AuditLog, NewEvent
from finint.data.split import SplitBoundaries
from finint.ml.calibration import IsotonicCalibrator, brier, reliability
from finint.ml.threshold import (
    ThresholdChoice,
    account_days,
    apply_threshold,
    capacity_tradeoff,
    choose_threshold,
)

SHAP_TOP = 5
MAX_CONTRIBUTING = 20


@dataclass(frozen=True)
class Champion:
    version: str
    path: Path
    booster: lgb.Booster
    features: list[str]


def load_champion(path: Path) -> Champion:
    booster = lgb.Booster(model_file=str(path / "model.txt"))
    features = json.loads((path / "features.json").read_text("utf-8"))
    return Champion(path.name, path, booster, features)


def _score(champion: Champion, frame: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(champion.booster.predict(frame[champion.features]), dtype=np.float64)


def _shap_top(champion: Champion, rows: pd.DataFrame) -> list[list[dict[str, Any]]]:
    contrib = np.asarray(
        champion.booster.predict(rows[champion.features], pred_contrib=True), dtype=np.float64
    )[:, :-1]  # last column is the bias term
    out = []
    for i in range(len(rows)):
        order = np.argsort(-np.abs(contrib[i]))[:SHAP_TOP]
        out.append(
            [
                {
                    "feature": champion.features[j],
                    "value": _jsonable(rows.iloc[i][champion.features[j]]),
                    "contribution": round(float(contrib[i, j]), 4),
                }
                for j in order
            ]
        )
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value if isinstance(value, int | float | str | bool) or value is None else str(value)


@dataclass(frozen=True)
class AlertRun:
    model_version: str
    threshold: ThresholdChoice
    tradeoff: list[dict[str, Any]]
    brier_raw: float
    brier_calibrated: float
    reliability_raw: list[dict[str, Any]]
    reliability_calibrated: list[dict[str, Any]]
    n_alerts: int
    alerts_by_split: dict[str, int]
    test_precision: float
    test_recall_transactions: float


def generate_alerts(
    db_path: Path,
    frame: pd.DataFrame,
    bounds: SplitBoundaries,
    champion: Champion,
    capacity_per_day: int,
    audit: AuditLog,
) -> AlertRun:
    """``frame`` must hold features, label, typology, split, in_tail, from_key, ts."""
    period = frame[(frame["split"] != "train") & (~frame["in_tail"])].copy()
    period["raw"] = _score(champion, period)
    val_mask = period["split"] == "validation"

    calibrator = IsotonicCalibrator.fit(
        period.loc[val_mask, "raw"].to_numpy(), period.loc[val_mask, "label"].to_numpy(bool)
    )
    period["score"] = calibrator.transform(period["raw"].to_numpy())
    calibrator.save(champion.path / "calibrator.json")

    period["day"] = period["ts"].dt.date
    val_days = (bounds.test_start - bounds.validation_start).total_seconds() / 86400
    val_ad = account_days(period[val_mask])
    n_val_laundering = int(period.loc[val_mask, "label"].sum())
    choice = choose_threshold(val_ad, capacity_per_day, n_val_laundering, val_days)
    tradeoff = capacity_tradeoff(val_ad, n_val_laundering, val_days)

    all_ad = account_days(period)
    split_of_day = period.groupby("day")["split"].agg(lambda s: s.mode().iloc[0])
    alerts = apply_threshold(all_ad, choice.threshold)
    alerts["split"] = alerts["day"].map(split_of_day).astype(str)

    _write_alerts(db_path, period, alerts, champion, choice, audit)

    test_alerts = alerts[alerts["split"] == "test"]
    n_test_laundering = int(period.loc[period["split"] == "test", "label"].sum())
    labels_val = period.loc[val_mask, "label"].to_numpy(bool)
    return AlertRun(
        model_version=champion.version,
        threshold=choice,
        tradeoff=tradeoff,
        brier_raw=brier(labels_val, period.loc[val_mask, "raw"].to_numpy()),
        brier_calibrated=brier(labels_val, period.loc[val_mask, "score"].to_numpy()),
        reliability_raw=reliability(labels_val, period.loc[val_mask, "raw"].to_numpy()),
        reliability_calibrated=reliability(labels_val, period.loc[val_mask, "score"].to_numpy()),
        n_alerts=len(alerts),
        alerts_by_split=alerts["split"].value_counts().to_dict(),
        test_precision=float(test_alerts["label"].mean()) if len(test_alerts) else 0.0,
        test_recall_transactions=(
            float(test_alerts["n_laundering"].sum() / n_test_laundering)
            if n_test_laundering
            else 0.0
        ),
    )


def _write_alerts(
    db_path: Path,
    period: pd.DataFrame,
    alerts: pd.DataFrame,
    champion: Champion,
    choice: ThresholdChoice,
    audit: AuditLog,
) -> None:
    by_txn = period.set_index("txn_id")
    top_rows = by_txn.loc[alerts["top_txn_id"]]
    shap = _shap_top(champion, top_rows)
    contributing = (
        period.sort_values("score", ascending=False)
        .groupby(["from_key", "day"], observed=True)["txn_id"]
        .apply(lambda s: s.head(MAX_CONTRIBUTING).tolist())
    )
    records, labels, events = [], [], []
    for i, alert in enumerate(alerts.itertuples(index=False)):
        alert_id = f"AL-{alert.day:%Y%m%d}-{alert.priority_rank:04d}"
        snapshot = {f: _jsonable(top_rows.iloc[i][f]) for f in champion.features}
        txns = contributing.loc[(alert.from_key, alert.day)]
        records.append(
            (
                alert_id, alert.from_key, alert.day, alert.split, float(alert.score),
                float(by_txn.loc[alert.top_txn_id, "raw"]), choice.threshold,
                champion.version, int(alert.priority_rank), int(alert.top_txn_id), txns,
                int(alert.n_txn), json.dumps(snapshot), json.dumps(shap[i]),
            )
        )  # fmt: skip
        labels.append((alert_id, bool(alert.label), int(alert.n_laundering)))
        events.append(
            NewEvent(
                event_type="ALERT_CREATED",
                actor=f"ml:{champion.version}",
                alert_id=alert_id,
                payload={
                    "account_key": alert.from_key,
                    "alert_date": alert.day.isoformat(),
                    "score": round(float(alert.score), 6),
                    "threshold": round(choice.threshold, 6),
                    "capacity_per_day": choice.capacity_per_day,
                    "model_version": champion.version,
                    "priority_rank": int(alert.priority_rank),
                    "top_txn_id": int(alert.top_txn_id),
                    "shap_top5": shap[i],
                },
            )
        )
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "CREATE OR REPLACE TABLE alerts (alert_id VARCHAR PRIMARY KEY, account_key VARCHAR, "
            "alert_date DATE, split VARCHAR, score DOUBLE, score_raw DOUBLE, threshold DOUBLE, "
            "model_version VARCHAR, priority_rank INTEGER, top_txn_id BIGINT, "
            "contributing_txn_ids BIGINT[], n_txn INTEGER, features_json VARCHAR, "
            "shap_top5_json VARCHAR)"
        )
        con.executemany(
            "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records
        )
        con.execute(
            "CREATE OR REPLACE TABLE alert_labels (alert_id VARCHAR PRIMARY KEY, "
            "is_laundering BOOLEAN, n_laundering_txn INTEGER)"
        )
        con.executemany("INSERT INTO alert_labels VALUES (?, ?, ?)", labels)
    audit.append_many(events)


def save_run(run: AlertRun, path: Path) -> None:
    payload = asdict(run)
    path.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
