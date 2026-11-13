"""Alert generation end to end on synthetic data (tasks 2.8-2.10 · R2.6-R2.8)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pytest

from finint.core.audit import AuditLog
from finint.data.ingest import ingest
from finint.data.normalize import normalize
from finint.data.sources import AmlWorldFiles
from finint.data.split import SplitFractions, split_boundaries
from finint.ml.alerts import generate_alerts, load_champion
from finint.ml.dataset import load_frame
from finint.ml.features import FEATURES, build_tabular_features
from finint.ml.registry import ArtifactInfo, choose_champion
from finint.ml.train import evaluate, save_artifact, train_lightgbm
from synthetic_amlworld import write_dataset


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, object]:
    root = tmp_path_factory.mktemp("alerts")
    write_dataset(root / "raw")
    db = root / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(root / "raw"), db)
    normalize(db, date(2026, 11, 13), {"US Dollar": 1.0})
    build_tabular_features(db)
    with duckdb.connect(str(db), read_only=True) as con:
        bounds = split_boundaries(con, SplitFractions(train=0.8, validation=0.1, test=0.1))
    frame = load_frame(db, bounds)
    model = train_lightgbm(
        frame, FEATURES, "m0", 42, weights=(1.0,), rounds=30, log=lambda _m: None
    )
    path = save_artifact(model, evaluate(model, frame), root / "artifacts", "m0-test-20261113")
    audit = AuditLog(root / "audit.sqlite")
    result = generate_alerts(db, frame, bounds, load_champion(path), 5, audit)
    return db, path, root / "audit.sqlite", result


def test_alerts_table_has_required_fields(run: tuple[Path, Path, Path, object]) -> None:
    db, _, _, _ = run
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            "SELECT alert_id, score, threshold, model_version, features_json, shap_top5_json, "
            "contributing_txn_ids FROM alerts"
        ).fetchall()
        labels = con.execute("SELECT count(*) FROM alert_labels").fetchone()
    assert rows
    assert labels == (len(rows),)
    for alert_id, score, threshold, version, features, shap, txns in rows:
        assert alert_id.startswith("AL-")
        assert score >= threshold
        assert version == "m0-test-20261113"
        assert set(json.loads(features)) == set(FEATURES)
        assert len(json.loads(shap)) == 5
        assert txns


def test_every_alert_is_audited_and_chain_verifies(run: tuple[Path, Path, Path, object]) -> None:
    db, _, audit_path, _ = run
    log = AuditLog(audit_path)
    events = log.events()
    with duckdb.connect(str(db), read_only=True) as con:
        alert_ids = {r[0] for r in con.execute("SELECT alert_id FROM alerts").fetchall()}
    assert {e.alert_id for e in events} == alert_ids
    assert all(e.event_type == "ALERT_CREATED" for e in events)
    assert len({e.run_id for e in events}) == 1
    assert events[0].run_id is not None
    assert all(
        {"score", "threshold", "model_version", "shap_top5"} <= set(e.payload) for e in events
    )
    assert log.verify() == len(events)


def test_calibrator_saved_as_json(run: tuple[Path, Path, Path, object]) -> None:
    _, path, _, _ = run
    raw = json.loads((path / "calibrator.json").read_text("utf-8"))
    assert raw["kind"] == "isotonic"
    assert len(raw["x"]) == len(raw["y"])


def _info(name: str, val: float, kind: str) -> ArtifactInfo:
    return ArtifactInfo(name, f"{name}-x-1", Path(name), val, kind, {})


def test_champion_is_best_explainable_model() -> None:
    champ, why = choose_champion(
        {"m0": _info("m0", 0.49, "lightgbm"), "m1": _info("m1", 0.55, "lightgbm")}
    )
    assert champ.name == "m1"
    assert "highest" in why
    champ, why = choose_champion(
        {"m1": _info("m1", 0.55, "lightgbm"), "m2": _info("m2", 0.60, "gnn")}
    )
    assert champ.name == "m1"
    assert "m2 scores higher" in why
