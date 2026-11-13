# 02 · ML Detection: Tasks

- [x] 2.1 Split config + split report (label counts per split and typology) · _R2.1_
- [x] 2.2 Tabular features in SQL (window functions) + leakage property test · _R2.1, R2.2_
- [x] 2.3 M0 LightGBM baseline, imbalance handling, val tuning · _R2.2_
- [x] 2.4 Graph features on sliding windows (degree, fan-in/out, PageRank, bounded cycles, flow-through, entity aggregation) · _R2.3_
- [x] 2.5 M1 LightGBM + graph; lift table vs M0 · _R2.3_
- [x] 2.6 Metrics module: PR-AUC, precision@k, recall@k, per-typology recall (unit-tested on toy arrays) · _R2.5_
- [x] 2.7 M2 GraphSAGE edge classifier (PyG), time-boxed · _R2.4_
- [x] 2.8 Isotonic calibration + reliability diagrams + Brier · _R2.6_
- [x] 2.9 Account-day aggregation + capacity threshold + trade-off plot · _R2.7_
- [x] 2.10 `make alerts`: alerts table with SHAP top-5, model_version, audit events · _R2.8_
- [ ] 2.11 `tools/ml.py`: `ml_score`, `explain_alert` · _R2.9_
- [ ] 2.12 `reports/ml_report.md` generator · _R2.5_
- [x] 2.13 Model card per champion artifact (**port from `pig_finserv/modeling/model_card.py`**) · _R2.8_
