"""Score calibration (task 2.8 · R2.6).

Isotonic regression fitted on validation scores. The calibrator is stored as JSON (the
breakpoints of the monotone step function) instead of a pickle: it is portable, readable
in an audit, and loading it cannot execute code.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from finint.ml.metrics import Array, Labels


@dataclass(frozen=True)
class IsotonicCalibrator:
    x: list[float]
    y: list[float]

    @classmethod
    def fit(cls, scores: Array, labels: Labels) -> IsotonicCalibrator:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores, labels.astype(float))
        return cls(x=model.X_thresholds_.tolist(), y=model.y_thresholds_.tolist())

    def transform(self, scores: Array) -> Array:
        return np.interp(scores, np.asarray(self.x), np.asarray(self.y))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"kind": "isotonic", "x": self.x, "y": self.y}), "utf-8")

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator:
        raw = json.loads(path.read_text("utf-8"))
        if raw.get("kind") != "isotonic":
            raise ValueError(f"unexpected calibrator kind {raw.get('kind')!r}")
        return cls(x=list(raw["x"]), y=list(raw["y"]))


def brier(labels: Labels, probs: Array) -> float:
    return float(np.mean((probs - labels.astype(float)) ** 2))


RELIABILITY_EDGES = (0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def reliability(labels: Labels, probs: Array) -> list[dict[str, Any]]:
    """Mean predicted vs observed rate per probability bin.

    Bins are log-spaced at the low end: with a 0.1% base rate, equal-width or quantile
    bins would put almost every transaction in the first bin and hide the top of the
    ranking, where alerts live.
    """
    rows = []
    edges = RELIABILITY_EDGES
    for lo, hi in itertools.pairwise(edges):
        mask = (probs >= lo) & ((probs < hi) if hi < 1.0 else (probs <= hi))
        if mask.any():
            rows.append(
                {
                    "bin": f"[{lo:g}, {hi:g})",
                    "n": int(mask.sum()),
                    "mean_predicted": float(probs[mask].mean()),
                    "observed_rate": float(labels[mask].mean()),
                }
            )
    return rows
