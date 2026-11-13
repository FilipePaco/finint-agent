"""GraphSAGE edge classifier, model M2 (task 2.7 · R2.4).

Same leakage rule as the graph features: transactions of day *d* are classified using node
embeddings computed on the graph of days ``[d - W, d)`` only. Per day:

1. build the window graph (distinct directed edges, amount-weighted) and node inputs
   (``graph_nodes`` statistics, log-scaled);
2. a 2-layer GraphSAGE (mean aggregation) produces node embeddings ``h``;
3. an MLP scores each transaction from ``[h_src || h_dst || tabular features]``.

Accounts absent from the window get a learned "unseen" embedding. Training runs end to end
over the train days; the epoch with the best validation PR-AUC is kept. Time-boxed by
``max_epochs`` (ADR-02.2: the champion is chosen by metrics, not sophistication).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from finint.data.split import SplitBoundaries
from finint.ml.features import NUMERIC
from finint.ml.graph_features import NODE_FEATURES
from finint.ml.metrics import pr_auc


class EdgeSAGE(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = SAGEConv(node_in, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.unseen = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        return torch.cat([h, self.unseen.unsqueeze(0)], dim=0)  # last row = unseen node

    def forward(
        self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_x: torch.Tensor
    ) -> torch.Tensor:
        logits: torch.Tensor = self.head(torch.cat([h[src], h[dst], edge_x], dim=1))
        return logits.squeeze(1)


@dataclass
class DayBatch:
    day: date
    x: torch.Tensor
    edge_index: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    edge_x: torch.Tensor
    label: torch.Tensor
    txn_id: np.ndarray[Any, np.dtype[np.int64]]
    split: np.ndarray[Any, np.dtype[np.str_]]  # per transaction, from the shared boundaries
    in_tail: np.ndarray[Any, np.dtype[np.bool_]]
    typology: list[object]


def _standardize(
    frame: pd.DataFrame, columns: list[str], stats: dict[str, tuple[float, float]]
) -> np.ndarray[Any, np.dtype[np.float32]]:
    values = np.log1p(np.clip(frame[columns].to_numpy(np.float64), 0, None))
    means = np.array([stats[c][0] for c in columns])
    stds = np.array([stats[c][1] for c in columns])
    scaled = (values - means) / stds
    # Ratio features are NULL when a denominator is 0 (~4.7k rows in HI-Small). LightGBM
    # handles NaN natively; a neural net does not, so impute the training mean (0 here).
    clean: np.ndarray[Any, np.dtype[np.float32]] = np.nan_to_num(
        scaled, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    return clean


_DAY_TXNS = f"""
SELECT f.txn_id, t.from_key, t.to_key, t.is_laundering AS label, t.typology,
       CASE WHEN t.ts < $v THEN 'train' WHEN t.ts < $t THEN 'validation' ELSE 'test' END AS split,
       coalesce(t.ts >= $tail, false) AS in_tail,
       {", ".join(f"f.{c}" for c in NUMERIC)}
FROM features_txn f JOIN transactions t USING (txn_id)
WHERE t.ts >= $day AND t.ts < $next
ORDER BY f.txn_id
"""  # noqa: S608 - module constants only

_WINDOW_EDGES = """
SELECT DISTINCT from_key AS s, to_key AS d FROM transactions
WHERE ts >= $start AND ts < $day AND from_key <> to_key
"""

_NODES = f"""
SELECT account_key, {", ".join(NODE_FEATURES)} FROM graph_nodes WHERE day = $day
"""  # noqa: S608 - module constants only


def _feature_stats(
    con: duckdb.DuckDBPyConnection, until: datetime
) -> dict[str, tuple[float, float]]:
    """Mean/std of log1p-clipped features, from training rows only (before ``until``)."""
    stats: dict[str, tuple[float, float]] = {}
    sources = (
        ("features_txn f JOIN transactions t USING (txn_id)", "f.", NUMERIC, "t.ts"),
        ("graph_nodes g", "g.", NODE_FEATURES, "g.day"),
    )
    for table, prefix, columns, day_col in sources:
        exprs = ", ".join(
            f"avg(ln(1 + greatest({prefix}{c}, 0))), stddev(ln(1 + greatest({prefix}{c}, 0)))"
            for c in columns
        )
        row = con.execute(
            f"SELECT {exprs} FROM {table} WHERE {day_col} < $until",  # noqa: S608 - constants
            {"until": until},
        ).fetchone()
        values = list(row) if row else [None] * (2 * len(columns))
        for i, column in enumerate(columns):
            mean, std = values[2 * i], values[2 * i + 1]
            stats[column] = (float(mean or 0.0), float(std or 0.0) or 1.0)
    return stats


def load_day_batches(
    db_path: Path,
    bounds: SplitBoundaries,
    window_days: int = 3,
    log: Callable[[str], None] = print,
) -> list[DayBatch]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        stats = _feature_stats(con, bounds.validation_start)
        row = con.execute("SELECT min(ts)::DATE, max(ts)::DATE FROM transactions").fetchone()
        if row is None or row[0] is None:
            raise ValueError("transactions table is empty")
        first, last = row
        batches = []
        day = first + timedelta(days=1)  # day 1 has no history
        while day <= last:
            nxt = day + timedelta(days=1)
            nodes = con.execute(_NODES, {"day": day}).df()
            index = {k: i for i, k in enumerate(nodes["account_key"])}
            unseen = len(index)
            edges = con.execute(
                _WINDOW_EDGES, {"start": day - timedelta(days=window_days), "day": day}
            ).df()
            edge_index = torch.tensor(
                np.array([[index[s] for s in edges["s"]], [index[d] for d in edges["d"]]]),
                dtype=torch.long,
            ).reshape(2, -1)
            txns = con.execute(
                _DAY_TXNS,
                {
                    "day": day,
                    "next": nxt,
                    "v": bounds.validation_start,
                    "t": bounds.test_start,
                    "tail": bounds.tail_start,
                },
            ).df()
            if txns.empty:  # no transactions that day: nothing to classify
                day = nxt
                continue
            batches.append(
                DayBatch(
                    day=day,
                    x=torch.from_numpy(_standardize(nodes, list(NODE_FEATURES), stats)),
                    edge_index=edge_index,
                    src=torch.tensor(
                        [index.get(k, unseen) for k in txns["from_key"]], dtype=torch.long
                    ),
                    dst=torch.tensor(
                        [index.get(k, unseen) for k in txns["to_key"]], dtype=torch.long
                    ),
                    edge_x=torch.from_numpy(_standardize(txns, list(NUMERIC), stats)),
                    label=torch.tensor(txns["label"].to_numpy(bool), dtype=torch.float32),
                    txn_id=txns["txn_id"].to_numpy(np.int64),
                    split=txns["split"].to_numpy(str),
                    in_tail=txns["in_tail"].to_numpy(bool),
                    typology=txns["typology"].tolist(),
                )
            )
            log(f"gnn: loaded {day} ({len(nodes):,} nodes, {len(txns):,} txns)")
            day = nxt
    return batches


@dataclass
class GnnResult:
    model: EdgeSAGE
    validation_pr_auc: float
    best_epoch: int
    history: list[dict[str, float]]


def _scores(
    model: EdgeSAGE, batch: DayBatch, chunk: int = 200_000
) -> np.ndarray[Any, np.dtype[np.float64]]:
    model.eval()
    with torch.no_grad():
        h = model.embed(batch.x, batch.edge_index)
        parts = [
            torch.sigmoid(
                model(
                    h,
                    batch.src[i : i + chunk],
                    batch.dst[i : i + chunk],
                    batch.edge_x[i : i + chunk],
                )
            )
            for i in range(0, len(batch.src), chunk)
        ]
    return torch.cat(parts).numpy().astype(np.float64) if parts else np.zeros(0)


def collect_scores(model: EdgeSAGE, batches: list[DayBatch], split: str) -> pd.DataFrame:
    frames = []
    for batch in batches:
        mask = batch.split == split
        if not mask.any():
            continue
        frames.append(
            pd.DataFrame(
                {
                    "txn_id": batch.txn_id[mask],
                    "score": _scores(model, batch)[mask],
                    "label": batch.label.numpy()[mask].astype(bool),
                    "in_tail": batch.in_tail[mask],
                    "typology": [t for t, m in zip(batch.typology, mask, strict=True) if m],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def train_gnn(
    batches: list[DayBatch],
    seed: int,
    max_epochs: int = 8,
    minibatch: int = 50_000,
    pos_weight: float = 10.0,
    log: Callable[[str], None] = print,
) -> GnnResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EdgeSAGE(node_in=len(NODE_FEATURES), edge_in=len(NUMERIC))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    best_state, best_ap, best_epoch, history = None, -1.0, -1, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for batch in rng.permutation(len(batches)):
            day = batches[int(batch)]
            train_idx = np.flatnonzero(day.split == "train")
            for start in range(0, len(train_idx), minibatch):
                idx = torch.from_numpy(rng.permutation(train_idx)[start : start + minibatch])
                h = model.embed(day.x, day.edge_index)
                logits = model(h, day.src[idx], day.dst[idx], day.edge_x[idx])
                loss = loss_fn(logits, day.label[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss on {day.day} (epoch {epoch})")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
        val = collect_scores(model, batches, "validation")
        ap = pr_auc(val["label"].to_numpy(), val["score"].to_numpy())
        history.append({"epoch": epoch, "train_loss": total, "val_pr_auc": ap})
        log(f"gnn: epoch {epoch} train loss {total:,.1f} val PR-AUC {ap:.4f}")
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return GnnResult(model, best_ap, best_epoch, history)


def save_gnn_artifact(
    result: GnnResult, metrics: dict[str, Any], artifacts_dir: Path, version: str
) -> Path:
    out = artifacts_dir / "models" / version
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out / "model.pt")
    payload = {
        "model_version": version,
        "validation_pr_auc": result.validation_pr_auc,
        "best_epoch": result.best_epoch,
        "history": result.history,
        "node_features": list(NODE_FEATURES),
        "edge_features": list(NUMERIC),
        "test": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
