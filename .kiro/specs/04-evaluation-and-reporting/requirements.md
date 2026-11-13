# 04 · Evaluation & Reporting: Requirements

## Introduction
Prove the agent's quality, safety and cost with a reproducible evaluation. Produce the weekly operations report and the delivery artifacts: README results, architecture PDF and hosted demo.

## Requirements

### R4.1 Evaluation set
- THE SYSTEM SHALL build an eval set of ≥ 50 alerts from the **test split** alerts produced by spec 02, stratified as follows: ≥ 30 true laundering alerts covering every typology present (≥ 3 each where available), ≥ 20 high-score false positives, ≥ 8 alerts linked to injected sanctions cases (incl. decoys), and ≥ 10 alerts carrying prompt-injection payloads (cases may overlap).
- Ground truth per alert: expected disposition (laundering → `ESCALATE`, not laundering → `CLOSE_FALSE_POSITIVE`), expected typologies (from `attempts`), expected sanctions outcome and injection flag. Stored in `eval/eval_set.jsonl` with the seed and selection query.

### R4.2 Metrics (computed in code)
- **Disposition accuracy** vs ground truth, confusion matrix, and the `NEEDS_MORE_INFO` rate reported separately (counted as not-correct in accuracy).
- **Faithfulness:** (a) % of first drafts passing the grounding validator; (b) unsupported values per released case file (target 0); (c) % of cases `BLOCKED_UNGROUNDED`. An LLM judge may add a narrative-support score as a secondary metric only.
- **Pattern identification recall:** expected typologies marked `CONFIRMED` / expected typologies. Precision is also reported.
- **Sanctions screening:** precision/recall of the deterministic matcher on injected cases, and decoy false-positive rate.
- **Injection resistance:** % of payload cases where the payload changed disposition, triggered an off-allowlist tool, or appeared as an instruction-following output (target 0), plus detector recall.
- **Latency** (p50/p95 per case) and **cost** (tokens, searches, USD per case).

### R4.3 Model comparison and ablations
- THE SYSTEM SHALL run the eval for ≥ 2 model configs (Gemini default; Ollama open-weight) and report a quality × cost table.
- THE SYSTEM SHALL run the ablations `no-skills` and `no-reviewer` on the default model and report their deltas.

### R4.4 One command, CI
- `make eval` SHALL run the full eval and write `reports/eval_report.md` + `reports/eval_results.jsonl`.
- CI SHALL run unit tests plus a **replay eval** on a small subset using recorded LLM responses (no API keys needed). Recorded responses are refreshed with `make eval-record`.

### R4.5 Weekly operations report
- `make report --week <iso-week>` SHALL write `reports/ops_report_<week>.md` (+ HTML) with:
  - alert volume and % change vs previous week;
  - precision of reviewed past alerts (share confirmed as laundering);
  - average evidence-gathering time per alert, before (documented manual baseline) and after (measured agent time);
  - agent–analyst disposition agreement;
  - cost per investigated alert and its trend;
  - chart 1: daily alerts, last 30 days; chart 2: monthly alerts, last 12 months.
- Where history is simulated (dataset spans ~18 days; analyst decisions derived from labels), THE report SHALL label it **SIMULATED** and document the method.

### R4.6 Delivery artifacts
- `docs/architecture.pdf`: orchestrating agent, the three LLM agents, tools (incl. ML scorer and sanctions screener), LLM provider, DuckDB store, OFAC list, news sources, analyst console and audit log, with trust boundaries marked. Generated from source in `docs/`.
- README: problem, architecture, how to run, ML results, eval results (pasted from reports), governance/guardrails summary, limitations, roadmap.
- Bonus: hosted demo of the analyst console on a precomputed synthetic sample, runnable without installing anything.

## Open questions [NEEDS CLARIFICATION]
- Q1: Manual evidence-gathering baseline is not in the data. Proposal: a configurable assumption (default 40 min/alert), labeled as such, with a sensitivity line in the report.
- Q2: Hosted demo target: Streamlit Community Cloud (free, simplest) vs Cloud Run. Proposal: Streamlit Cloud with precomputed runs, and live runs only if a key is configured.
