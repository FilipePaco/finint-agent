"""Evaluation set builder (task 4.1 · R4.1).

Two parts, both from the **test** period, with ground truth from the dataset labels and the
documented injections:

* **natural alerts**: alerts the champion actually raised. Laundering alerts are stratified
  by typology group (seeded), plus a seeded sample of false-positive alerts;
* **forced alerts**: account-days that carry an injected test case (prompt-injection memo,
  sanctions near-match, adverse-media entity) but were not alerted. They are scored by the
  same champion and calibrator and marked ``source="forced"``, so coverage of every guardrail
  is deliberate rather than left to chance. Metrics are reported for both parts.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from finint.core.audit import AuditLog, NewEvent
from finint.ml.alerts import Champion, _jsonable, _shap_top
from finint.ml.calibration import IsotonicCalibrator
from finint.ml.features import CATEGORICAL, NUMERIC
from finint.ml.graph_features import GRAPH_FEATURES

# Dataset typologies -> investigation skills (what the agent can name).
TYPOLOGY_TO_SKILL = {
    "FAN-OUT": "fan-out",
    "FAN-IN": "fan-in",
    "CYCLE": "cycle",
    "GATHER-SCATTER": "scatter-gather",
    "SCATTER-GATHER": "scatter-gather",
    "STACK": "stack",
    "RANDOM": "stack",
    "BIPARTITE": "bipartite",
}
Disposition = Literal["ESCALATE", "CLOSE_FALSE_POSITIVE"]


class SanctionsLink(BaseModel):
    case_id: str
    role: str
    expected: str  # TRUE_MATCH | FALSE_MATCH
    account_key: str


class EvalCase(BaseModel):
    alert_id: str
    source: Literal["natural", "forced"]
    reasons: list[str]
    account_key: str
    alert_date: date
    is_laundering: bool
    expected_disposition: Disposition
    expected_typologies: list[str] = Field(default_factory=list)
    injection_case: str | None = None
    injection_txn: int | None = None
    original_memo: str | None = None
    sanctions: list[SanctionsLink] = Field(default_factory=list)
    media_case: str | None = None
    expected_media: str | None = None


def _rows(
    con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    cursor = con.execute(sql, params or {})
    names = [d[0] for d in cursor.description or []]
    return [dict(zip(names, r, strict=True)) for r in cursor.fetchall()]


def _group(typology: str | None) -> str:
    return TYPOLOGY_TO_SKILL.get(typology or "", "unlisted")


def _ground_truth(con: duckdb.DuckDBPyConnection, account: str, day: date) -> dict[str, Any]:
    row = _rows(
        con,
        "SELECT count(*) FILTER (WHERE is_laundering) AS n_laundering, "
        "list(DISTINCT typology) FILTER (WHERE is_laundering AND typology IS NOT NULL) AS typ "
        "FROM transactions WHERE from_key = $k AND ts::DATE = $d",
        {"k": account, "d": day},
    )[0]
    typologies = sorted(
        {TYPOLOGY_TO_SKILL[t] for t in (row["typ"] or []) if t in TYPOLOGY_TO_SKILL}
    )
    laundering = bool(row["n_laundering"])
    return {
        "is_laundering": laundering,
        "expected_typologies": typologies,
        "expected_disposition": "ESCALATE" if laundering else "CLOSE_FALSE_POSITIVE",
    }


def _score_account_day(
    con: duckdb.DuckDBPyConnection,
    champion: Champion,
    calibrator: IsotonicCalibrator,
    account: str,
    day: date,
) -> dict[str, Any]:
    cols = ", ".join(
        [
            *(f"f.{c}" for c in NUMERIC),
            *(f"f.{c}" for c in CATEGORICAL),
            *(f"g.{c}" for c in GRAPH_FEATURES),
        ]
    )
    frame = con.execute(
        f"SELECT t.txn_id, {cols} FROM transactions t "  # noqa: S608 - module constants only
        "JOIN features_txn f USING (txn_id) JOIN features_graph g USING (txn_id) "
        "WHERE t.from_key = $k AND t.ts::DATE = $d",
        {"k": account, "d": day},
    ).df()
    for c in CATEGORICAL:
        frame[c] = frame[c].astype("category")
    for c in (*NUMERIC, *GRAPH_FEATURES):
        frame[c] = frame[c].astype(np.float32)
    raw = np.asarray(champion.booster.predict(frame[champion.features]), dtype=np.float64)
    frame["score"] = calibrator.transform(raw)
    frame["raw"] = raw
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    top = frame.iloc[[0]]
    return {
        "score": float(frame.loc[0, "score"]),
        "score_raw": float(frame.loc[0, "raw"]),
        "top_txn_id": int(frame.loc[0, "txn_id"]),
        "contributing": [int(t) for t in frame["txn_id"].head(20)],
        "n_txn": len(frame),
        "features": {f: _jsonable(top.iloc[0][f]) for f in champion.features},
        "shap": _shap_top(champion, top)[0],
    }


def build_eval_set(
    db_path: Path,
    champion: Champion,
    calibrator_path: Path,
    audit: AuditLog,
    seed: int,
    n_laundering: int = 30,
    n_false_positive: int = 20,
    per_group_min: int = 3,
) -> list[EvalCase]:
    """Select natural alerts, add forced alerts for injected cases, write them to ``alerts``."""
    rng = random.Random(seed)  # noqa: S311 - seeded sampling, not security
    calibrator = IsotonicCalibrator.load(calibrator_path)
    with duckdb.connect(str(db_path)) as con:
        con.execute("DELETE FROM alert_labels WHERE alert_id LIKE 'EV-%'")
        con.execute("DELETE FROM alerts WHERE alert_id LIKE 'EV-%'")
        natural = _rows(
            con,
            """
            SELECT a.alert_id, a.account_key, a.alert_date, a.contributing_txn_ids, l.is_laundering,
                   t.typology AS top_typology, a.score
            FROM alerts a JOIN alert_labels l USING (alert_id)
            JOIN transactions t ON t.txn_id = a.top_txn_id
            WHERE a.split = 'test' ORDER BY a.alert_id""",
        )
        cases: dict[str, EvalCase] = {}
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in natural:
            if a["is_laundering"]:
                by_group[_group(a["top_typology"])].append(a)
        chosen: list[dict[str, Any]] = []
        for group in sorted(by_group):
            pool = by_group[group][:]
            rng.shuffle(pool)
            chosen += pool[:per_group_min]
        rest = [a for g in sorted(by_group) for a in by_group[g] if a not in chosen]
        rng.shuffle(rest)
        chosen += rest[: max(0, n_laundering - len(chosen))]
        fps = [a for a in natural if not a["is_laundering"]]
        rng.shuffle(fps)
        chosen += fps[:n_false_positive]
        for a in chosen:
            gt = _ground_truth(con, a["account_key"], a["alert_date"])
            reason = (
                f"laundering:{_group(a['top_typology'])}"
                if a["is_laundering"]
                else "false_positive"
            )
            cases[a["alert_id"]] = EvalCase(
                alert_id=a["alert_id"],
                source="natural",
                reasons=[reason],
                account_key=a["account_key"],
                alert_date=a["alert_date"],
                **gt,
            )

        def account_day_case(account: str, day: date, reason: str) -> EvalCase:
            for case in cases.values():
                if case.account_key == account and case.alert_date == day:
                    case.reasons.append(reason)
                    return case
            existing = _rows(
                con,
                "SELECT alert_id FROM alerts WHERE account_key = $k AND alert_date = $d",
                {"k": account, "d": day},
            )
            if existing:
                alert_id, source = existing[0]["alert_id"], "natural"
            else:
                alert_id = (
                    f"EV-{day:%Y%m%d}-{len([c for c in cases if c.startswith('EV-')]) + 1:04d}"
                )
                source = "forced"
                _insert_forced(con, audit, champion, calibrator, alert_id, account, day)
            case = EvalCase(
                alert_id=alert_id,
                source=source,
                reasons=[reason],
                account_key=account,
                alert_date=day,
                **_ground_truth(con, account, day),
            )
            cases[alert_id] = case
            return case

        for p in _rows(
            con,
            """
                SELECT p.case_id, p.txn_id, p.original_reference, t.from_key, t.ts::DATE AS day
                FROM injected_prompts p JOIN transactions t USING (txn_id) ORDER BY p.case_id""",
        ):
            case = account_day_case(p["from_key"], p["day"], f"injection:{p['case_id']}")
            case.injection_case, case.injection_txn = p["case_id"], p["txn_id"]
            case.original_memo = p["original_reference"]

        first_test = con.execute(
            "SELECT min(alert_date) FROM alerts WHERE split = 'test'"
        ).fetchone()
        window = {"test_start": first_test[0] if first_test else None}
        subjects = _rows(
            con,
            """
            SELECT s.case_id, s.role, s.expected, a.account_key,
                   min(t.ts::DATE) FILTER (WHERE t.is_laundering) AS day
            FROM injected_sanctions s JOIN accounts a USING (entity_id)
            JOIN transactions t ON t.from_key = a.account_key
            WHERE s.role = 'subject' AND t.ts >= $test_start
            GROUP BY s.case_id, s.role, s.expected, a.account_key
            QUALIFY row_number() OVER (
                PARTITION BY s.case_id ORDER BY day NULLS LAST, a.account_key) = 1
            ORDER BY s.case_id""",
            window,
        )
        rng.shuffle(subjects)
        for s in [x for x in subjects if x["day"] is not None][:5]:
            case = account_day_case(s["account_key"], s["day"], f"sanctions:{s['case_id']}")
            case.sanctions.append(
                SanctionsLink(
                    case_id=s["case_id"],
                    role="subject",
                    expected=s["expected"],
                    account_key=s["account_key"],
                )
            )
        counterparties = _rows(
            con,
            """
            SELECT s.case_id, s.expected, a.account_key AS cp, t.from_key AS payer,
                   t.ts::DATE AS day
            FROM injected_sanctions s JOIN accounts a USING (entity_id)
            JOIN transactions t ON t.to_key = a.account_key
            WHERE s.role = 'counterparty' AND t.ts >= $test_start
            QUALIFY row_number() OVER (PARTITION BY s.case_id ORDER BY t.ts) = 1
            ORDER BY s.case_id""",
            window,
        )
        decoys = [c for c in counterparties if c["expected"] == "FALSE_MATCH"]
        trues = [c for c in counterparties if c["expected"] == "TRUE_MATCH"]
        rng.shuffle(decoys)
        rng.shuffle(trues)
        for c in decoys[:3] + trues[:2]:
            case = account_day_case(c["payer"], c["day"], f"sanctions:{c['case_id']}")
            case.sanctions.append(
                SanctionsLink(
                    case_id=c["case_id"],
                    role="counterparty",
                    expected=c["expected"],
                    account_key=c["cp"],
                )
            )
        for m in _rows(
            con,
            """
                SELECT m.case_id, m.expected_media, a.account_key,
                       min(t.ts::DATE) AS day
                FROM injected_media m JOIN accounts a USING (entity_id)
                JOIN transactions t ON t.from_key = a.account_key
                WHERE m.role = 'subject' AND t.ts >= $test_start
                GROUP BY m.case_id, m.expected_media, a.account_key
                QUALIFY row_number() OVER (PARTITION BY m.case_id ORDER BY day, a.account_key) = 1
                ORDER BY m.case_id""",
            window,
        ):
            case = account_day_case(m["account_key"], m["day"], f"media:{m['case_id']}")
            case.media_case, case.expected_media = m["case_id"], m["expected_media"]
    return sorted(cases.values(), key=lambda c: (c.source, c.alert_id))


def _insert_forced(
    con: duckdb.DuckDBPyConnection,
    audit: AuditLog,
    champion: Champion,
    calibrator: IsotonicCalibrator,
    alert_id: str,
    account: str,
    day: date,
) -> None:
    s = _score_account_day(con, champion, calibrator, account, day)
    threshold = con.execute(
        "SELECT any_value(threshold) FROM alerts WHERE split = 'test'"
    ).fetchone()
    con.execute(
        "INSERT INTO alerts VALUES (?, ?, ?, 'eval', ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
        [
            alert_id,
            account,
            day,
            s["score"],
            s["score_raw"],
            threshold[0] if threshold else None,
            champion.version,
            s["top_txn_id"],
            s["contributing"],
            s["n_txn"],
            json.dumps(s["features"]),
            json.dumps(s["shap"]),
        ],
    )
    gt = con.execute(
        "SELECT count(*) FILTER (WHERE is_laundering) FROM transactions "
        "WHERE from_key = ? AND ts::DATE = ?",
        [account, day],
    ).fetchone()
    n_laundering = int(gt[0]) if gt else 0
    con.execute(
        "INSERT INTO alert_labels VALUES (?, ?, ?)", [alert_id, n_laundering > 0, n_laundering]
    )
    audit.append(
        NewEvent(
            "ALERT_CREATED",
            f"ml:{champion.version}",
            {
                "account_key": account,
                "alert_date": day.isoformat(),
                "score": round(s["score"], 6),
                "model_version": champion.version,
                "forced_for_evaluation": True,
                "shap_top5": s["shap"],
            },
            alert_id=alert_id,
            run_id=f"alerts-eval-{datetime.now():%Y%m%d}",
        )
    )


def write_eval_set(cases: list[EvalCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    return path


def load_eval_set(path: Path) -> list[EvalCase]:
    return [
        EvalCase.model_validate_json(line) for line in path.read_text("utf-8").splitlines() if line
    ]


def describe(cases: list[EvalCase]) -> pd.DataFrame:
    rows = []
    for c in cases:
        rows.append(
            {
                "alert_id": c.alert_id,
                "source": c.source,
                "laundering": c.is_laundering,
                "typologies": ",".join(c.expected_typologies),
                "reasons": ";".join(c.reasons),
            }
        )
    return pd.DataFrame(rows)
