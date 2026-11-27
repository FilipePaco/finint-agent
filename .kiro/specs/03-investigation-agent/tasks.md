# 03 · Investigation Agent: Tasks

Write the tests in `test-strategy.md` before or alongside each core module.

Core (framework-free, test first)
- [x] 3.1 `core/models.py`: Alert, Evidence, CaseFile, Disposition, AuditEvent, AnalystDecision · _R3.5, R3.8_
- [x] 3.2 `core/audit.py`: hash-chained append-only log + `AuditIntegrityError` + `make audit-verify` (**adapted from an earlier personal project**, generalized to agent events) · _R3.12_
- [x] 3.3 `core/minimize.py` + test that label columns never reach a prompt · _R3.9_
- [x] 3.4 `core/screening.py` + tests on `sanctions_cases.yaml` · _R3.4_
- [x] 3.5 `core/grounding.py` + property tests · _R3.5_
- [x] 3.6 `core/policy.py` rules + tests · _R3.7_
- [x] 3.7 `core/injection.py` heuristic detector + tests on payload YAML · _R3.10_

Tools & skills
- [x] 3.8 Read-only SQL tools (parameterized, ruff `S` clean) with `content_and_artifact` + evidence registration · _R3.1_
- [x] 3.9 `detect_patterns` (code typology detectors reused from graph features) · _R3.1_
- [x] 3.10 Skills: 7 typologies + sanctions-review + case-writing · _R3.2_
- [x] 3.11 MediaProvider: FixtureProvider + TavilyProvider + permission gate (**adapted from an earlier personal project**) · _R3.3_

Agents & graph
- [x] 3.12 LLM factory per role via `init_chat_model` (**adapted from an earlier personal project**), provider-agnostic, prompt versioning · _R3.13_
- [x] 3.13 `observability.py`: Langfuse `CallbackHandler` (trace_id = run_id) or `[]`; usage via LangChain `UsageMetadataCallbackHandler`; USD via Langfuse pricing / `genai-prices` · _R3.12, R3.14_
- [x] 3.14 Media agent subgraph + verbatim-quote verification (**adapted from an earlier personal project**) · _R3.3_
- [x] 3.15 Investigator agent (tools, skills index, budget, hypotheses, LLM-unavailable fallback) · _R3.1, R3.2_
- [x] 3.16 Reviewer agent + revise loop · _R3.6_
- [x] 3.17 Graph wiring: grounding retries, policy guard, interrupt, finalize, SQLite checkpointer · _R3.5–R3.8_
- [x] 3.17b Harness counterparty sanctions screening: code selection rule, `cp-` evidence, audit, policy input, hidden from the LLMs (ADR-03.10) · _R3.4, R3.7, R3.12_
- [x] 3.18 CLI: `finint investigate <alert_id>` (prints case + audit trail)

Console & docs
- [x] 3.19 Streamlit analyst console with role check, approve/edit/reject + diff audit · _R3.8, R3.11_
- [x] 3.20 `compose.yaml` runs the console; Langfuse via its official compose or Cloud (v3 self-host needs Postgres, ClickHouse, Redis, MinIO; not duplicated here) · _R3.14_
- [x] 3.21 `docs/audit.md`: how to read the audit trail, verify the chain, and open the matching Langfuse trace (**adapted from an earlier personal project**) · _R3.12_

Evaluation follow-ups
- [x] 3.22 Controlled typology vocabulary for hypotheses (`core/typology.py`, enum in `submit_case`, prompt line, alias normaliser) · _R3.1, R3.2_ · ADR-03.9
- [x] 3.23 Multi-account typology evidence: two-hop and amount-matched chain indicators in `detect_patterns`, candidates from `config/typology.yaml` (`core/typology_signals.py`), typology gate in the investigator, skills and v2 prompt, offline ceiling `finint eval typology-signals` · _R3.15, R3.1, R3.2_ · ADR-03.11
