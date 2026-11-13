"""GraphSAGE edge classifier M2 (task 2.7 · R2.4).

Marked ``gnn`` and run in their own process (``make test-gnn``): on macOS, PyTorch and
LightGBM ship different OpenMP runtimes, and loading both in one process segfaults
LightGBM. Training M2 is already a separate CLI process, so production is unaffected.

PyTorch is imported **inside** the tests, never at module level: pytest imports every test
module while collecting, even deselected ones, which would load torch into the main run.
Skipped where the ``gnn`` group is not installed (e.g. CI: PyPI's Linux torch wheels pull
CUDA, ~2.5 GB).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import ModuleType

import duckdb
import numpy as np
import pytest

from finint.data.ingest import ingest
from finint.data.normalize import normalize
from finint.data.sources import AmlWorldFiles
from finint.data.split import SplitBoundaries, SplitFractions, split_boundaries
from finint.ml.evaluation import evaluate_scores
from finint.ml.features import NUMERIC, build_tabular_features
from finint.ml.graph_features import build_graph_features
from synthetic_amlworld import write_dataset

pytestmark = pytest.mark.gnn


@pytest.fixture(scope="module")
def gnn() -> ModuleType:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from finint.ml import gnn as module

    return module


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, SplitBoundaries]:
    root = tmp_path_factory.mktemp("gnn")
    write_dataset(root / "raw")
    path = root / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(root / "raw"), path)
    normalize(path, date(2026, 11, 13), {"US Dollar": 1.0})
    build_tabular_features(path)
    build_graph_features(path, log=lambda _m: None)
    with duckdb.connect(str(path), read_only=True) as con:
        bounds = split_boundaries(con, SplitFractions(train=0.8, validation=0.1, test=0.1))
    return path, bounds


def test_batches_follow_shared_split_and_shapes(
    gnn: ModuleType, db: tuple[Path, SplitBoundaries]
) -> None:
    batches = gnn.load_day_batches(*db, log=lambda _m: None)
    assert batches
    for batch in batches:
        n = len(batch.txn_id)
        assert batch.edge_x.shape == (n, len(NUMERIC))
        assert batch.src.shape == batch.dst.shape == (n,)
        assert int(batch.src.max()) <= batch.x.shape[0]  # index x.shape[0] = unseen node
        assert set(np.unique(batch.split)) <= {"train", "validation", "test"}
        assert np.isfinite(batch.edge_x.numpy()).all()


def test_train_and_evaluate_end_to_end(gnn: ModuleType, db: tuple[Path, SplitBoundaries]) -> None:
    batches = gnn.load_day_batches(*db, log=lambda _m: None)
    result = gnn.train_gnn(batches, seed=42, max_epochs=2, minibatch=500, log=lambda _m: None)
    assert result.best_epoch in (1, 2)
    assert len(result.history) == 2
    metrics = evaluate_scores(gnn.collect_scores(result.model, batches, "test"))
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["n"] > 0


def test_null_ratio_features_do_not_produce_nan(gnn: ModuleType, tmp_path: Path) -> None:
    path, bounds = _fresh_db(tmp_path)
    with duckdb.connect(str(path)) as con:
        con.execute(
            "UPDATE features_txn SET amount_vs_out_mean_7d = NULL, flow_through_ratio_24h = NULL "
            "WHERE txn_id % 3 = 0"
        )
    batches = gnn.load_day_batches(path, bounds, log=lambda _m: None)
    assert all(np.isfinite(b.edge_x.numpy()).all() for b in batches)
    result = gnn.train_gnn(batches, seed=42, max_epochs=1, minibatch=500, log=lambda _m: None)
    assert np.isfinite(result.history[0]["train_loss"])


def _fresh_db(root: Path) -> tuple[Path, SplitBoundaries]:
    write_dataset(root / "raw")
    path = root / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(root / "raw"), path)
    normalize(path, date(2026, 11, 13), {"US Dollar": 1.0})
    build_tabular_features(path)
    build_graph_features(path, log=lambda _m: None)
    with duckdb.connect(str(path), read_only=True) as con:
        bounds = split_boundaries(con, SplitFractions(train=0.8, validation=0.1, test=0.1))
    return path, bounds
