# 02 · ML Detection: Design

## Flow
```
transactions (DuckDB)
  ├─► features_tabular.sql  (window functions: velocity, rolling sums)
  ├─► graph_features.py     (per window: networkx/igraph → degree, PageRank, cycles, flow-through)
  └─► feature_store table   (txn_id → feature vector, as-of correct)
         │
         ├─► M0: LightGBM tabular
         ├─► M1: LightGBM tabular + graph        ──► champion selection (val PR-AUC)
         └─► M2: GraphSAGE edge classifier (PyG)
                                                      │
                                    isotonic calibration (val)
                                                      │
                          account-day aggregation → capacity threshold
                                                      │
                                   alerts table + ALERT_CREATED audit events
```

## Leakage control
- Tabular features use SQL window functions with `RANGE BETWEEN INTERVAL ... PRECEDING AND CURRENT ROW`, partitioned by account.
- Graph features use **sliding windows**: for transactions in day *d*, the graph is built from [d − W, d) plus earlier same-day edges where cheap. Otherwise the previous closed window is used (documented trade-off). Default W = 3 days.
- Cycles: bounded DFS (length ≤ 4) on the window graph, limited to nodes with in and out degree ≥ 1, to keep it tractable on 5M edges.
- A property test samples transactions and recomputes their features from a truncated table (ts ≤ t) to assert equality.

## Split (HI-Small ≈ 18 days, label rate ≈ 0.1%)
Default: train = first 60% of days, val = next 20%, test = last 20%. Boundaries are printed with label counts per split. If any typology has < 10 positives in test, the report says so next to its recall.

## Imbalance
LightGBM `scale_pos_weight` tuned on val PR-AUC. GNN uses weighted BCE. No SMOTE: synthetic graph edges would corrupt the graph features.

## GNN
GraphSAGE (2 layers, mean aggregator). Node features = account aggregates. Edge features = transaction features. Edge classifier = MLP on [h_u ‖ h_v ‖ e_uv]. Mini-batch via `LinkNeighborLoader` and temporal masking (edges only from the past). CPU-friendly sizes. Time-boxed.

## Threshold justification
Plot precision@k and alerts/day vs threshold on validation. Pick the threshold whose expected daily alert count ≈ capacity. Report expected precision, expected missed laundering (recall gap) and marginal precision of the next 50 alerts. This gives the ops lead a dial: "+50 capacity buys +x recall".

## Model versioning
`model_version = "{model}-{git_sha[:7]}-{train_date}"`. Artifacts go in `artifacts/models/{model_version}/` (model, calibrator, threshold.json, feature_list.json, metrics.json).

## Outputs
`reports/ml_report.md`, `reports/figures/pr_curves.png`, `reliability.png`, `typology_recall.png`, `threshold_tradeoff.png`.

## Model card
Each champion artifact ships `model_card.md` (pattern from `pig_finserv/modeling/model_card.py`): intended use, training window, features, metrics on test (PR-AUC, precision@budget, per-typology recall), calibration, threshold and capacity assumption, known limitations (synthetic data, short time span, typologies with few positives), and `model_version`.

## Architecture decision records

### ADR-02.1 Account-day alerts (vs transaction-level alerts)
**Decision:** score transactions, alert on account-days.
**Why:** laundering patterns span many transactions. Transaction alerts would flood analysts with duplicates of the same case, and capacity is measured in cases, not rows.

### ADR-02.2 LightGBM + graph features as expected champion, GNN as challenger (vs GNN-first)
**Decision:** train all three, pick by validation PR-AUC, report all.
**Why:** gradient boosting is the strong, explainable baseline (SHAP feeds the agent's `explain_alert`). The GNN tests whether learned neighborhood structure beats engineered graph features. Either answer is a finding.

### ADR-02.3 Capacity-derived threshold (vs fixed probability cut-off / F1-optimal)
**Decision:** the threshold yields ≈ analyst capacity alerts/day on validation.
**Why:** the binding constraint in FinCrime ops is analyst time, not a probability. F1-optimal ignores capacity. The trade-off plot shows the ops lead what extra capacity would buy.

### ADR-02.4 Isotonic calibration (vs Platt / none)
**Decision:** isotonic on validation.
**Why:** tree-ensemble scores are poorly calibrated and the validation set has enough positives for a non-parametric fit. Calibrated scores make "risk 0.8" meaningful to analysts and the agent.

### ADR-02.5 No resampling (vs SMOTE / undersampling)
**Decision:** class weights only.
**Why:** synthetic minority samples would create transactions with no coherent graph neighborhood, corrupting graph features.
