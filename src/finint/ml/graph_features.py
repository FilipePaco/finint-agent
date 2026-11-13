"""Graph features on sliding daily windows (task 2.4 · R2.3).

For transactions on day *d*, the graph is built from transactions in ``[d - W, d)``: the W
previous **closed** days (default 3). No edge from day *d* or later is ever used, so these
features cannot leak. What happened earlier the same day is covered by the tabular windows
(trade-off documented in design.md).

Per account and day:

* in/out degree (distinct counterparties) and fan-in/fan-out ratios;
* PageRank on the directed, amount-weighted graph (igraph);
* membership in directed cycles of length 2, 3 and 4 (SQL self-joins on distinct edges);
* flow-through: share of inflow paid out again within the window;
* entity-level aggregates across all accounts of the same owner (multi-bank entities).

Each transaction gets the sender's and the receiver's features for its day.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import duckdb
import igraph as ig
import pandas as pd

NODE_FEATURES = (
    "out_degree",
    "in_degree",
    "fan_out_ratio",
    "fan_in_ratio",
    "pagerank",
    "in_cycle_2",
    "in_cycle_3",
    "in_cycle_4",
    "flow_through",
    "entity_out_degree",
    "entity_in_degree",
    "entity_n_accounts_active",
)

GRAPH_FEATURES = tuple(f"{side}_{name}" for side in ("src", "dst") for name in NODE_FEATURES)

_EDGES = """
CREATE OR REPLACE TEMP TABLE w_edges AS
SELECT from_key AS s, to_key AS d, sum(CAST(amount_usd AS DOUBLE)) AS amt, count(*) AS n
FROM transactions
WHERE ts >= $start AND ts < $end AND from_key <> to_key
GROUP BY ALL
"""

_NODE_STATS = """
WITH outs AS (SELECT s AS k, count(*) AS out_degree, sum(amt) AS out_amt FROM w_edges GROUP BY s),
ins AS (SELECT d AS k, count(*) AS in_degree, sum(amt) AS in_amt FROM w_edges GROUP BY d),
c2 AS (SELECT DISTINCT a.s AS k FROM w_edges a JOIN w_edges b ON a.d = b.s AND b.d = a.s),
c3 AS (
    SELECT DISTINCT a.s AS k FROM w_edges a
    JOIN w_edges b ON a.d = b.s
    JOIN w_edges c ON b.d = c.s AND c.d = a.s
    WHERE a.s <> b.d
),
c4 AS (
    SELECT DISTINCT a.s AS k FROM w_edges a
    JOIN w_edges b ON a.d = b.s
    JOIN w_edges c ON b.d = c.s
    JOIN w_edges e ON c.d = e.s AND e.d = a.s
    WHERE a.s <> b.d AND a.d <> c.d AND a.s <> c.s
),
nodes AS (SELECT k FROM outs UNION SELECT k FROM ins)
SELECT
    n.k AS account_key,
    coalesce(o.out_degree, 0) AS out_degree,
    coalesce(i.in_degree, 0) AS in_degree,
    coalesce(o.out_degree, 0) / (1.0 + coalesce(i.in_degree, 0)) AS fan_out_ratio,
    coalesce(i.in_degree, 0) / (1.0 + coalesce(o.out_degree, 0)) AS fan_in_ratio,
    (n.k IN (SELECT k FROM c2))::INT AS in_cycle_2,
    (n.k IN (SELECT k FROM c3))::INT AS in_cycle_3,
    (n.k IN (SELECT k FROM c4))::INT AS in_cycle_4,
    CASE WHEN coalesce(i.in_amt, 0) > 0
         THEN least(coalesce(o.out_amt, 0) / i.in_amt, 10.0) ELSE 0 END AS flow_through
FROM nodes n
LEFT JOIN outs o ON o.k = n.k
LEFT JOIN ins i ON i.k = n.k
"""


def _pagerank(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    edges = con.execute("SELECT s, d, amt FROM w_edges").df()
    if edges.empty:
        return pd.DataFrame({"account_key": pd.Series(dtype=str), "pagerank": []})
    graph = ig.Graph.DataFrame(edges[["s", "d"]], directed=True, use_vids=False)
    ranks = graph.pagerank(weights=edges["amt"].clip(lower=0).tolist(), directed=True)
    # Scale by node count so values are comparable across windows of different size.
    scale = graph.vcount()
    return pd.DataFrame({"account_key": graph.vs["name"], "pagerank": [r * scale for r in ranks]})


def _day_features(con: duckdb.DuckDBPyConnection, day: date, window_days: int) -> pd.DataFrame:
    start = day - timedelta(days=window_days)
    con.execute(_EDGES, {"start": start, "end": day})
    stats = con.execute(_NODE_STATS).df()
    stats = stats.merge(_pagerank(con), on="account_key", how="left")
    stats["day"] = day
    return stats


_ENTITY = """
SELECT g.*,
    sum(g.out_degree) OVER (PARTITION BY g.day, a.entity_id) AS entity_out_degree,
    sum(g.in_degree) OVER (PARTITION BY g.day, a.entity_id) AS entity_in_degree,
    count(*) OVER (PARTITION BY g.day, a.entity_id) AS entity_n_accounts_active
FROM graph_nodes_raw g
JOIN accounts a ON a.account_key = g.account_key
"""

_JOIN = f"""
CREATE OR REPLACE TABLE features_graph AS
SELECT t.txn_id,
    {", ".join(f"coalesce(s.{c}, 0) AS src_{c}" for c in NODE_FEATURES)},
    {", ".join(f"coalesce(r.{c}, 0) AS dst_{c}" for c in NODE_FEATURES)}
FROM transactions t
LEFT JOIN graph_nodes s ON s.account_key = t.from_key AND s.day = t.ts::DATE
LEFT JOIN graph_nodes r ON r.account_key = t.to_key AND r.day = t.ts::DATE
ORDER BY t.txn_id
"""  # noqa: S608 - module constants only


def build_graph_features(
    db_path: Path, window_days: int = 3, log: Callable[[str], None] = print
) -> int:
    """(Re)create ``graph_nodes`` (per account-day) and ``features_graph`` (per transaction)."""
    with duckdb.connect(str(db_path)) as con:
        row = con.execute("SELECT min(ts)::DATE, max(ts)::DATE FROM transactions").fetchone()
        if row is None or row[0] is None:
            raise ValueError("transactions table is empty")
        first, last = row
        frames = []
        day = first
        while day <= last:
            frames.append(_day_features(con, day, window_days))
            log(f"graph: {day} ({len(frames[-1]):,} active accounts in prior {window_days} days)")
            day += timedelta(days=1)
        nodes = pd.concat(frames, ignore_index=True)
        con.register("graph_nodes_df", nodes)
        con.execute("CREATE OR REPLACE TABLE graph_nodes_raw AS SELECT * FROM graph_nodes_df")
        con.execute(f"CREATE OR REPLACE TABLE graph_nodes AS {_ENTITY}")
        con.execute("DROP TABLE graph_nodes_raw")
        con.execute(_JOIN)
        result = con.execute("SELECT count(*) FROM features_graph").fetchone()
    return int(result[0]) if result else 0
