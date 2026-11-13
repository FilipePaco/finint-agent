"""Model-ready frames joined from ``features_txn`` and labels, with split assignment."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from finint.data.split import SplitBoundaries
from finint.ml.features import CATEGORICAL, NUMERIC
from finint.ml.graph_features import GRAPH_FEATURES

_SQL = f"""
SELECT f.txn_id,
       {", ".join(f"f.{c}" for c in NUMERIC)},
       {{graph_columns}}
       {", ".join(f"f.{c}" for c in CATEGORICAL)},
       t.is_laundering AS label,
       t.typology,
       CASE WHEN t.ts < $v THEN 'train' WHEN t.ts < $t THEN 'validation' ELSE 'test' END AS split,
       coalesce(t.ts >= $tail, false) AS in_tail
FROM features_txn f
JOIN transactions t USING (txn_id)
{{graph_join}}
ORDER BY f.txn_id
"""  # noqa: S608 - module constants only


def load_frame(db_path: Path, boundaries: SplitBoundaries, *, graph: bool = False) -> pd.DataFrame:
    """Features + label + split. ``graph=True`` adds ``features_graph`` columns (model M1)."""
    sql = _SQL.format(
        graph_columns="".join(f"g.{c}, " for c in GRAPH_FEATURES) if graph else "",
        graph_join="JOIN features_graph g USING (txn_id)" if graph else "",
    )
    numeric = (*NUMERIC, *GRAPH_FEATURES) if graph else NUMERIC
    with duckdb.connect(str(db_path), read_only=True) as con:
        frame = con.execute(
            sql,
            {
                "v": boundaries.validation_start,
                "t": boundaries.test_start,
                "tail": boundaries.tail_start,
            },
        ).df()
    for column in numeric:
        frame[column] = frame[column].astype(np.float32)
    for column in CATEGORICAL:
        # Categories fixed from the whole column so train/val/test share the same codes.
        frame[column] = frame[column].astype("category")
    frame["label"] = frame["label"].astype(bool)
    frame["split"] = frame["split"].astype("category")
    return frame
