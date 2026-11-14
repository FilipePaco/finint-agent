# 03 · Investigation Agent: Tasks

Write the tests in `test-strategy.md` before or alongside each core module.

Core (framework-free, test first)
- [x] 3.1 `core/models.py`: Alert, Evidence, CaseFile, Disposition, AuditEvent, AnalystDecision · _R3.5, R3.8_
- [x] 3.2 `core/audit.py`: hash-chained append-only log + `AuditIntegrityError` + `make audit-verify` (**port from `pig_finserv/audit.py`**, swap triage fields for agent events) · _R3.12_
- [x] 3.3 `core/minimize.py` + test that label columns never reach a prompt · _R3.9_
- [x] 3.4 `core/screening.py` + tests on `sanctions_cases.yaml` · _R3.4_
- [x] 3.5 `core/grounding.py` + property tests · _R3.5_
- [x] 3.6 `core/policy.py` rules + tests · _R3.7_
- [x] 3.7 `core/injection.py` heuristic detector + tests on payload YAML · _R3.10_

Tools & skills
- [ ] 3.8 Read-only SQL tools (parameterized, ruff `S` clean) with `content_and_artifact` + evidence registration · _R3.1_
- [ ] 3.9 `detect_patterns` (code typology detectors reused from graph features) · _R3.1_
- [ ] 3.10 Skills: 7 typologies + sanctions-review + case-writing · _R3.2_
- [ ] 3.11 MediaProvider: FixtureProvider + TavilyProvider + permission gate (**contract adapted from `pig_finserv` `EvidenceSource`**) · _R3.3_

Agents & graph
- [ ] 3.12 LLM factory per role via `init_chat_model` (**port from HealthCareAI `llm.py`**), provider-agnostic, prompt versioning · _R3.13_
- [ ] 3.13 `observability.py`: Langfuse `CallbackHandler` (trace_id = run_id) or `[]`; usage via LangChain `UsageMetadataCallbackHandler`; USD via Langfuse pricing / `genai-prices` · _R3.12, R3.14_
- [ ] 3.14 Media agent subgraph + verbatim-quote verification (**loop pattern from HealthCareAI `news_agent.py`**) · _R3.3_
- [ ] 3.15 Investigator agent (tools, skills index, budget, hypotheses, LLM-unavailable fallback) · _R3.1, R3.2_
- [ ] 3.16 Reviewer agent + revise loop · _R3.6_
- [ ] 3.17 Graph wiring: grounding retries, policy guard, interrupt, finalize, SQLite checkpointer · _R3.5–R3.8_
- [ ] 3.18 CLI: `finint investigate <alert_id>` (prints case + audit trail)

Console & docs
- [ ] 3.19 Streamlit analyst console with role check, approve/edit/reject + diff audit · _R3.8, R3.11_
- [ ] 3.20 `compose.yaml` profile `observability` running self-hosted Langfuse · _R3.14_
- [ ] 3.21 `docs/audit.md`: how to read the audit trail, verify the chain, and open the matching Langfuse trace (**pattern from HealthCareAI `docs/auditoria.md`**) · _R3.12_
