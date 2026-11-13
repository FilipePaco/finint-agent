"""Model registry over ``artifacts/models`` and champion selection (ADR-02.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactInfo:
    name: str
    version: str
    path: Path
    validation_pr_auc: float
    kind: str  # "lightgbm" or "gnn"
    metrics: dict[str, Any]


def latest_artifacts(artifacts_dir: Path) -> dict[str, ArtifactInfo]:
    """Most recent artifact per model name (m0, m1, m2), by directory modification time."""
    found: dict[str, ArtifactInfo] = {}
    root = artifacts_dir / "models"
    if not root.is_dir():
        return found
    for path in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime):
        metrics_file = path / "metrics.json"
        if not metrics_file.is_file():
            continue
        metrics = json.loads(metrics_file.read_text("utf-8"))
        name = path.name.split("-", 1)[0]
        kind = "lightgbm" if (path / "model.txt").is_file() else "gnn"
        found[name] = ArtifactInfo(
            name, path.name, path, float(metrics["validation_pr_auc"]), kind, metrics
        )
    return found


def choose_champion(artifacts: dict[str, ArtifactInfo]) -> tuple[ArtifactInfo, str]:
    """Best validation PR-AUC. Returns the champion and a one-line justification.

    Alerts need per-feature explanations (SHAP) for the investigation agent, so the
    champion is the best LightGBM model; if a GNN scores higher on validation, the
    justification says so and the trade-off is reported.
    """
    if not artifacts:
        raise ValueError("no trained models found; run 'make train' first")
    best_overall = max(artifacts.values(), key=lambda a: a.validation_pr_auc)
    explainable = [a for a in artifacts.values() if a.kind == "lightgbm"]
    if not explainable:
        raise ValueError("no LightGBM model found; alerts need SHAP explanations")
    champion = max(explainable, key=lambda a: a.validation_pr_auc)
    ranking = ", ".join(
        f"{a.name} {a.validation_pr_auc:.3f}"
        for a in sorted(artifacts.values(), key=lambda a: -a.validation_pr_auc)
    )
    if best_overall is champion:
        why = f"highest validation PR-AUC ({ranking})"
    else:
        why = (
            f"best explainable model; {best_overall.name} scores higher on validation but "
            f"has no per-feature explanations ({ranking})"
        )
    return champion, why
