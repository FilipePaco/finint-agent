# 02 · ML Detection: Requirements

## Introduction
Train a model that scores transactions for laundering risk, show that graph features add lift over a tabular baseline, calibrate the scores, derive the alert threshold from analyst capacity, and log every alert.

## Requirements

### R2.1 Time-based split
- THE SYSTEM SHALL split data chronologically into train / validation / test, with boundaries in `config/ml.yaml`. No random split anywhere.
- THE SYSTEM SHALL compute every feature for a transaction only from data with timestamp ≤ that transaction (no future leakage). A test SHALL assert this on a fixture.

### R2.2 Tabular baseline
- THE SYSTEM SHALL train a LightGBM model on per-transaction and account-history features: amount_usd, log-amount, payment format, currency pair, cross-bank, hour/weekday, round-amount and just-under-threshold flags, and rolling account velocity (count/sum over 1h, 24h, 7d).
- THE SYSTEM SHALL handle class imbalance explicitly (class weights or focal-style weighting) and document the choice.

### R2.3 Graph features
- THE SYSTEM SHALL add graph features computed on a trailing time window: in/out degree, distinct counterparties, fan-in/fan-out ratios, PageRank, membership in short directed cycles (length ≤ 4 within window), 2-hop flow-through ratio (in ≈ out within Δt), and entity-level aggregation across banks.
- THE SYSTEM SHALL report the lift of baseline + graph features over the baseline on the same test split.

### R2.4 GNN (PyTorch)
- THE SYSTEM SHALL train a GraphSAGE edge-classification model (PyTorch Geometric) on the same split, and report it alongside the other two models.
- IF the GNN does not beat the graph-feature LightGBM THEN THE SYSTEM SHALL still report it honestly, and the champion is chosen by metrics, not by sophistication.

### R2.5 Metrics
- THE SYSTEM SHALL report on the test split: PR-AUC, precision@k at the alert budget, recall@k, and **recall per typology** at the chosen threshold. Accuracy and ROC-AUC may appear only as secondary columns.
- THE SYSTEM SHALL write `reports/ml_report.md` with a comparison table, PR curves and per-typology recall chart.

### R2.6 Calibration
- THE SYSTEM SHALL calibrate the champion's scores on the validation split (isotonic) and report reliability diagrams and Brier score before/after.

### R2.7 Capacity-based threshold
- GIVEN `analyst_capacity_per_day` (default 200) in config, THE SYSTEM SHALL aggregate transaction scores to **account-day alerts** (max score, plus the contributing txn_ids) and choose the score threshold that yields ≈ capacity alerts per day on validation.
- THE SYSTEM SHALL justify the threshold in the report: expected precision at that volume, recall left on the table, and the marginal precision of the next 50 alerts.

### R2.8 Alert logging
- WHEN `make alerts` runs THE SYSTEM SHALL write every alert to `alerts` with: alert_id, account_key, alert_date, score (raw + calibrated), threshold, model_version, feature snapshot (JSON), top contributing features (SHAP top-5), contributing txn_ids, priority rank.
- Each alert creation SHALL also be written as an `ALERT_CREATED` audit event.

### R2.9 Explanation tool
- THE SYSTEM SHALL expose `ml_score` / `explain_alert` as read-only functions that return the score, threshold, model_version and top features. The agent in spec 03 consumes them.

## Open questions [NEEDS CLARIFICATION]
- Q1: Alert unit is account-day (proposal) rather than single transaction. This reduces duplicate alerts for multi-transaction patterns. OK?
- Q2: GNN scope: one GraphSAGE variant with a 2-hour time budget for tuning. Stop there?
