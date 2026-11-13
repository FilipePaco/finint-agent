"""LightGBM training with validation-based selection (tasks 2.3, 2.5 · R2.2, R2.3).

Imbalance is handled with class weights (``scale_pos_weight``), chosen on validation PR-AUC
(ADR-02.5: no resampling). Early stopping also monitors validation average precision.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from finint.ml.evaluation import evaluate_scores
from finint.ml.metrics import Array, pr_auc

BASE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}


def predict(booster: lgb.Booster, features: pd.DataFrame) -> Array:
    """Scores as a float64 array (LightGBM's return type is a loose union)."""
    return np.asarray(booster.predict(features), dtype=np.float64)


@dataclass(frozen=True)
class TrainedModel:
    name: str
    booster: lgb.Booster
    features: list[str]
    params: dict[str, Any]
    validation_pr_auc: float
    candidates: dict[str, float]


def train_lightgbm(
    frame: pd.DataFrame,
    features: Sequence[str],
    name: str,
    seed: int,
    weights: Sequence[float] = (1.0, 10.0, 50.0),
    rounds: int = 2000,
    log: Callable[[str], None] = print,
) -> TrainedModel:
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "validation"]
    cols = list(features)
    best: TrainedModel | None = None
    candidates: dict[str, float] = {}
    for weight in weights:
        params = {**BASE_PARAMS, "scale_pos_weight": weight, "seed": seed}
        dtrain = lgb.Dataset(train[cols], label=train["label"].astype(int))
        dval = lgb.Dataset(val[cols], label=val["label"].astype(int), reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        score = pr_auc(val["label"].to_numpy(), predict(booster, val[cols]))
        candidates[f"scale_pos_weight={weight:g}"] = score
        log(
            f"{name}: scale_pos_weight={weight:g} "
            f"rounds={booster.best_iteration} val PR-AUC={score:.4f}"
        )
        if best is None or score > best.validation_pr_auc:
            best = TrainedModel(name, booster, cols, params, score, candidates)
    if best is None:
        raise ValueError("weights must not be empty")
    return TrainedModel(
        best.name, best.booster, cols, best.params, best.validation_pr_auc, candidates
    )


def evaluate(model: TrainedModel, frame: pd.DataFrame, split: str = "test") -> dict[str, Any]:
    """Primary metrics on the split without the low-activity tail, plus full-split metrics."""
    part = frame[frame["split"] == split]
    scored = pd.DataFrame(
        {
            "score": predict(model.booster, part[model.features]),
            "label": part["label"].to_numpy(),
            "in_tail": part["in_tail"].to_numpy(),
            "typology": part["typology"].tolist(),
        }
    )
    return evaluate_scores(scored, split)


def save_artifact(
    model: TrainedModel, metrics: dict[str, Any], artifacts_dir: Path, version: str
) -> Path:
    out = artifacts_dir / "models" / version
    out.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(out / "model.txt"), num_iteration=model.booster.best_iteration)
    (out / "features.json").write_text(json.dumps(model.features, indent=2), encoding="utf-8")
    (out / "params.json").write_text(json.dumps(model.params, indent=2), encoding="utf-8")
    payload = {
        "model_version": version,
        "validation_pr_auc": model.validation_pr_auc,
        "weight_search": model.candidates,
        "best_iteration": model.booster.best_iteration,
        "test": metrics,
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return out


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
