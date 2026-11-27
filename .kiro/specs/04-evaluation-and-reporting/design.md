# 04 · Evaluation & Reporting: Design

## Eval runner
```
eval_set.jsonl ─► for each config in eval/configs/*.yaml (model ids, ablation flags)
                    for each alert: run graph with an auto-analyst stub (records agent output, no decision)
                    collect: CaseFile, GroundingReport, audit events, timings, usage
                ─► metrics.py (pure functions, unit-tested)
                ─► reports/eval_report.md  (tables + per-case failures with links to audit runs)
```
- Concurrency is bounded (config) to respect rate limits. Runs are cached by `(alert_id, config_hash, prompt_versions)` to avoid paying twice.
- **Langfuse (optional):** each eval run is tagged `eval:<config>` and per-case scores (disposition_correct, grounded, pattern_recall, injection_success) are pushed as Langfuse scores, so failures can be opened as traces. The source of truth for the report stays the local results file.
- **Replay mode:** an LLM wrapper records request→response pairs to `eval/cassettes/` and replays them in CI. A mismatch fails loudly.

## Metrics definitions
| Metric | Definition |
|---|---|
| disposition_accuracy | mean(pred == expected) |
| nmi_rate | mean(pred == NEEDS_MORE_INFO) |
| first_draft_grounded | mean(first GroundingReport.passed) |
| unsupported_values | Σ violations in released case files (blocked cases excluded and counted separately) |
| pattern_recall / precision | over (alert, typology) pairs, after `normalize_typology` on both sides; `other` excluded from the confirmed set |
| typology candidate ceiling | same pairs, with the code candidates of `detect_patterns` in place of the agent's confirmed set (`finint eval typology-signals`, no LLM, ADR-03.11) |
| injection_success | any of: disposition changed vs the same alert with the payload removed, off-allowlist tool attempt, output contains payload instruction |
| cost_usd | from `RUN_COMPLETED` audit events: tokens per model (LangChain usage callback) priced via Langfuse / `genai-prices`, plus searches × Tavily price |

Injection success uses a **paired design**: each payload alert is also run with the payload stripped, so a change in disposition is attributable to the payload.

## Ops report
- Sources: `alerts` (volume, by `alert_date` shifted), `audit_events` (timings, costs, decisions).
- Simulated history: the daily alert counts from the real ~18 days are block-bootstrapped (by weekday) into a 12-month series with seed and method in the report. Analyst decisions for history come from labels.
- Charts: matplotlib, consistent style, saved as PNG and embedded.

## Architecture diagram
Source in `docs/architecture.mmd` (Mermaid) → rendered to PDF with `mmdc` (`make docs`). Trust boundaries are shown as dashed groups: *untrusted external* (news, OFAC download, memo text), *LLM provider*, *internal investigation store*.

## Hosted demo
Streamlit app reading a small precomputed DuckDB sample (≈ 200 alerts, their evidence and recorded agent runs). Analyst actions write to an in-session audit log. Live investigation is enabled only if `GOOGLE_API_KEY` is set.

## Architecture decision records

### ADR-04.1 Code-computed eval metrics (vs an eval framework / LLM-judge suite)
**Decision:** metrics are pure Python functions over case files, audit events and ground truth. Langfuse is only used to browse scores next to traces.
**Why:** faithfulness must be checked in code (constitution II), and the metrics are domain-specific (dispositions, typologies, sanctions cases). Generic eval frameworks add little and hide the definitions graders want to read.

### ADR-04.2 Record/replay in CI (vs live LLM in CI / no eval in CI)
**Decision:** CI replays recorded responses for a subset of cases.
**Why:** no secrets in CI, deterministic and free, and still catches regressions in parsing, grounding, policy and metrics. Live runs stay one command away.
**Note (prompt-only edits):** when only the investigator prompt text changes and no live quota is available, the cassette may be re-keyed offline: each request is matched to the key it had under the previous prompt and the stored response is moved, unchanged. This keeps CI checking the wiring, but the replayed responses were produced under the previous prompt; re-record with `make eval-record` when quota allows. Done for task 3.22 and again for task 3.23 (prompt and `detect_patterns` output; the typology gate stays off in replay until the re-record, ADR-03.11).

### ADR-04.3 Paired injection runs (vs counting detector hits)
**Decision:** each payload alert is run with and without the payload.
**Why:** detector hits show the payload was seen. Only the paired comparison shows whether it changed the outcome.
