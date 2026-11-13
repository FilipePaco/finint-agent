"""Training pipeline on a small synthetic dataset (tasks 2.3 · R2.2, R2.8)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from finint.data.ingest import ingest
from finint.data.normalize import normalize
from finint.data.sources import AmlWorldFiles
from finint.data.split import SplitFractions, split_boundaries
from finint.ml.dataset import load_frame
from finint.ml.features import CATEGORICAL, FEATURES, build_tabular_features
from finint.ml.train import evaluate, model_version, save_artifact, train_lightgbm
from synthetic_amlworld import write_dataset


@pytest.fixture(scope="module")
def frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    root = tmp_path_factory.mktemp("train")
    write_dataset(root / "raw")
    db = root / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(root / "raw"), db)
    normalize(db, date(2026, 11, 13), {"US Dollar": 1.0})
    build_tabular_features(db)
    # The synthetic set puts laundering in the last days: use a split that has positives
    # in validation and test.
    with duckdb.connect(str(db), read_only=True) as con:
        bounds = split_boundaries(con, SplitFractions(train=0.8, validation=0.1, test=0.1))
    return load_frame(db, bounds)


def test_frame_has_typed_columns_and_all_splits(frame: pd.DataFrame) -> None:
    assert set(frame["split"].unique()) == {"train", "validation", "test"}
    assert all(str(frame[c].dtype) == "category" for c in CATEGORICAL)
    assert frame["label"].dtype == bool


def test_train_evaluate_and_save(frame: pd.DataFrame, tmp_path: Path) -> None:
    model = train_lightgbm(frame, FEATURES, "m0", seed=42, weights=(1.0, 10.0), rounds=30,
                           log=lambda _m: None)  # fmt: skip
    assert set(model.candidates) == {"scale_pos_weight=1", "scale_pos_weight=10"}
    assert model.validation_pr_auc == max(model.candidates.values())
    metrics = evaluate(model, frame)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert set(metrics["budgets"]) == {"0.10%", "0.50%", "1.00%"}
    path = save_artifact(model, metrics, tmp_path, "m0-test-20261113")
    for name in ("model.txt", "features.json", "params.json", "metrics.json"):
        assert (path / name).is_file()
    saved = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["model_version"] == "m0-test-20261113"


def test_model_version_format() -> None:
    version = model_version("m1")
    name, sha, stamp = version.split("-")
    assert name == "m1"
    assert sha
    assert len(stamp) == 8
