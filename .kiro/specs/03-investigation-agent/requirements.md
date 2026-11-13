# 03 · Investigation Agent: Requirements

## Introduction
For a selected alert, a multi-agent system gathers evidence with read-only tools, identifies laundering typologies, screens counterparties against sanctions, checks adverse media and drafts a grounded case file with a recommended disposition. An analyst makes the final decision. Three LLM agents make real decisions (what to investigate, what the news supports, whether the draft holds up). Deterministic code owns every compliance-critical rule.

## Glossary
- **Evidence**: a tool output registered with an `evidence_id` in the run's evidence store.
- **Case file**: structured output (`CaseFile` Pydantic model) containing a narrative, findings, evidence references, disposition and open questions.
- **Skill**: a typology playbook (`skills/<name>/SKILL.md`) that the investigator loads on demand.

## Requirements

### R3.1 Investigator agent (autonomous evidence gathering)
**User story:** As an analyst, I want the groundwork done for me so that I spend my time judging, not collecting.
- WHEN an alert is opened THE SYSTEM SHALL run an investigator agent that **chooses** which tools to call, in what order, and when it has enough evidence. The tool sequence is not hard-coded.
- THE investigator SHALL have only read-only tools: account profile, transactions in a window, counterparties, multi-hop flow tracing, pattern detection, ML explanation, sanctions screening, skill loading and media-check delegation.
- THE investigator SHALL respect a budget (max tool calls, max tokens, max wall time from config). IF the budget is exhausted THEN it SHALL submit its best draft with disposition `NEEDS_MORE_INFO` and list what is missing.
- THE investigator SHALL state hypotheses (typologies suspected) before testing them. Hypotheses and outcomes are recorded in the case file.

### R3.2 Skills (progressive disclosure)
- THE SYSTEM SHALL provide skills for: fan-in, fan-out, cycle, scatter-gather / gather-scatter, stack (layering through intermediaries), bipartite, structuring, sanctions-review and case-writing.
- THE investigator's prompt SHALL contain only each skill's name and one-line description. The body is loaded via `load_skill` when the agent decides it is relevant. Each load is audited.
- Each typology skill SHALL define: indicators, the tool sequence that tests it, quantitative thresholds, common benign explanations, and the evidence required to support each disposition.

### R3.3 Adverse-media agent (quarantined)
- WHEN the investigator requests a media check for an entity THE SYSTEM SHALL run a separate media agent that **decides** queries (≤ 3), judges whether each result is about the same entity (using disambiguators: country, entity type), and extracts claims.
- THE media agent SHALL receive only the entity's synthetic name, country and type. It has no access to transaction tools or the case.
- Every claim SHALL carry an exact quote, source URL and retrieval date. Code SHALL verify that each quote is a verbatim substring of the retrieved document; unverifiable claims are dropped and logged.
- THE media agent SHALL classify each finding as `ALLEGATION`, `CHARGE`, `CONVICTION`, `SANCTION`, `IRRELEVANT` or `SAME_NAME_DIFFERENT_ENTITY`, and SHALL NOT escalate the wording beyond the source (e.g., an allegation must not become "involved in laundering").
- THE SYSTEM SHALL run real web search only for entities flagged `is_injected` or listed public enforcement examples. Otherwise it returns `NOT_PERMITTED`.

### R3.4 Sanctions screening (deterministic)
- THE `screen_sanctions` tool SHALL return, for each candidate: SDN entry, matched name/alias, algorithm scores, combined score, matched and mismatched fields (name, country, DOB if any), and a code-decided `match_status` (`MATCH`, `POTENTIAL_MATCH`, `NO_MATCH`) from config thresholds.
- THE LLM SHALL NOT be able to override `match_status`. It may only describe it.

### R3.5 Grounding validator (deterministic)
- BEFORE a case file reaches the reviewer or the analyst THE SYSTEM SHALL verify that:
  (a) every `evidence_id` cited exists in this run;
  (b) every amount, date, account id and entity name in findings matches a value in the cited evidence (amounts within 0.01; dates normalized);
  (c) every such value appearing anywhere in the narrative text exists in the run's evidence.
- IF validation fails THEN THE SYSTEM SHALL return the violations to the investigator for at most 2 repair attempts. IF it still fails, the case is marked `BLOCKED_UNGROUNDED` and is never shown as a recommendation.

### R3.6 Reviewer agent (adversarial critique)
- AFTER grounding passes THE SYSTEM SHALL run a reviewer agent that **decides** `ACCEPT` or `REVISE`. It checks whether the evidence supports the disposition, whether benign explanations were considered, whether wording is proportionate, and whether any text could tip off a customer.
- WHEN the reviewer returns `REVISE` it SHALL list concrete evidence requests or edits. The investigator then resumes with that feedback (max 2 review rounds). After the last round the reviewer's open objections are attached to the case for the analyst.
- THE reviewer SHALL use a different prompt and may use a different model than the investigator (config).

### R3.7 Policy guard (deterministic)
- THE SYSTEM SHALL apply disposition rules in code after review. Examples: `MATCH` sanctions status → at least `ESCALATE`; `POTENTIAL_MATCH` unresolved → not `CLOSE_FALSE_POSITIVE`; fewer than N evidence items → `NEEDS_MORE_INFO`; a detected injection attempt in untrusted data → flag for analyst.
- Every override SHALL record the rule id, the original and the new disposition.

### R3.8 Human in the loop
- THE SYSTEM SHALL pause the graph (LangGraph `interrupt`) and present the case file to the analyst, with evidence drill-down, sanctions scores and matched fields, media citations, reviewer objections and policy overrides.
- THE analyst SHALL `APPROVE`, `EDIT` (disposition and/or text) or `REJECT` with a mandatory reason. The decision, analyst id and diff SHALL be written to the audit trail. Nothing is final without this step.

### R3.9 Data minimization
- THE SYSTEM SHALL build all LLM context through an allowlist driven by `data_classification.yaml`. Label columns (`is_laundering`, `attempt_id`, `typology`) are `llm_forbidden` and a test SHALL assert they never appear in any prompt.
- Tool outputs SHALL return only the fields the investigation needs (e.g., no bank internal ids beyond `account_key`).

### R3.10 Prompt-injection resistance
- Untrusted fields (`reference`, counterparty names, media text) SHALL be wrapped in delimited data blocks, with instructions stating they are data.
- THE SYSTEM SHALL run a heuristic detector on untrusted text. Hits are registered as evidence (`INJECTION_ATTEMPT`) and surfaced to the analyst.
- Structural defenses SHALL hold even if the model is fooled: read-only tools, per-agent tool allowlists, deterministic policy guard, grounding validator.

### R3.11 Tipping-off and access
- Case files and agent outputs SHALL be stored only in the investigation store and shown only in the analyst console behind an analyst role check. No customer-facing endpoint can read them, and a test asserts this.

### R3.12 Audit trail (compliance system of record)
- THE SYSTEM SHALL append audit events for: `RUN_STARTED`, `LLM_CALL` (agent, model id, prompt version, input/output tokens, latency), `TOOL_CALL` (name, args, evidence_id), `SKILL_LOADED`, `MEDIA_CLAIM_DROPPED`, `GROUNDING_RESULT`, `REVIEW_VERDICT`, `POLICY_OVERRIDE`, `CASE_DRAFTED`, `ANALYST_DECISION`, `RUN_COMPLETED` (evidence-gathering time, total tokens per model, estimated USD).
- The audit log SHALL be hash-chained and `make audit-verify` SHALL detect tampering.
- Token counts SHALL come from LangChain's built-in usage metadata callback, not from custom counting code.

### R3.14 Observability (existing tool, optional)
- WHEN `LANGFUSE_*` credentials are configured THE SYSTEM SHALL send traces of every run to Langfuse through its LangChain callback, using `run_id` as `trace_id` and tagging each span with agent role, prompt version and alert id (no raw transaction data in metadata beyond what the LLM already saw).
- Latency breakdown, token usage and cost per trace/agent/model SHALL be read from Langfuse. USD estimates in the audit trail come from Langfuse model pricing or the `genai-prices` package. No hand-maintained price tracker.
- IF Langfuse is not configured or unreachable THEN THE SYSTEM SHALL run unchanged, and the audit trail stays complete (constitution VI).
- `observability.py` SHALL be the only module importing Langfuse.

### R3.13 Swappable LLM
- THE SYSTEM SHALL build all chat models from config through one factory, supporting Gemini (default) and an Ollama open-weight model. Each agent's model is set independently.

## Open questions [NEEDS CLARIFICATION]
- Q1: The reviewer uses the same model family as the investigator, but at a different tier (e.g., investigator on a fast/cheap Gemini model, reviewer on a stronger one). Proposal: yes. Eval reports the cost impact.
- Q2: Investigator budget defaults: 20 tool calls, 60k input tokens, 120 s. OK as a start?
