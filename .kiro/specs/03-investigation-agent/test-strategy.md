# 03 · Investigation Agent: Test Strategy

Convention: every test references a requirement. **Every guardrail has at least one rejection test**, i.e. a test that proves the bad thing is blocked, not just that the good path works.

## Layers
| Layer | What | LLM? | Runs in CI |
|---|---|---|---|
| Unit: `core/` | grounding, screening, policy, minimize, injection detector, audit chain | no | yes |
| Unit: `tools/` | SQL tools on a tiny fixture DuckDB; evidence registration | no | yes |
| Graph: scripted model | full graph with a fake chat model emitting scripted tool calls | fake | yes |
| Replay eval | real recorded LLM responses on a subset (spec 04) | replay | yes |
| Full eval | live models on ≥ 50 alerts (spec 04) | live | `make eval` |

## Guardrail rejection tests
| Test | Proves | Ref |
|---|---|---|
| `test_grounding_rejects_invented_amount` | draft citing an amount absent from evidence fails | R3.5 |
| `test_grounding_rejects_amount_in_narrative_only` | a value only in free text, not in evidence, fails | R3.5 |
| `test_grounding_rejects_unknown_evidence_id` | citing a non-existent evidence id fails | R3.5 |
| `test_ungrounded_after_retries_is_blocked` | 3rd failure → `BLOCKED_UNGROUNDED`, never shown as recommendation | R3.5 |
| `test_llm_cannot_override_match_status` | case file claiming NO_MATCH vs a code MATCH → policy forces ESCALATE | R3.4, R3.7 |
| `test_harness_screens_counterparties_without_the_llm` | counterparties are screened, audited and enforced even when the LLM never runs | R3.4, R3.7 |
| `test_counterparty_match_overrides_an_agent_closure` | agent closes a case with a counterparty MATCH → policy forces ESCALATE | R3.4, R3.7 |
| `test_harness_screens_stay_out_of_agent_context` | `cp-` evidence never reaches investigator or reviewer context; agent ids unchanged | R3.9 |
| `test_policy_blocks_close_on_potential_match` | CLOSE_FALSE_POSITIVE with unresolved POTENTIAL_MATCH is overridden | R3.7 |
| `test_label_columns_never_in_prompt` | `is_laundering`, `typology`, `attempt_id` never appear in any rendered prompt | R3.9 |
| `test_media_agent_has_no_transaction_tools` | media agent tool allowlist is exactly search/fetch | R3.3 |
| `test_media_claim_without_verbatim_quote_dropped` | paraphrased "quote" is dropped and audited | R3.3 |
| `test_web_search_not_permitted_for_non_injected_entity` | real search refused unless `is_injected` / public example | R3.3 |
| `test_off_allowlist_tool_call_rejected` | scripted model calling an unknown tool → rejected + audited | R3.10 |
| `test_injection_payload_flagged_as_evidence` | memo with payload → `INJECTION_ATTEMPT` evidence | R3.10 |
| `test_no_final_state_without_analyst` | graph cannot reach finalize without `ANALYST_DECISION` | R3.8 |
| `test_reject_requires_reason` | REJECT/EDIT without reason is refused | R3.8 |
| `test_customer_surface_has_no_case_access` | no non-analyst route/role can read case files | R3.11 |
| `test_audit_tamper_detected` | editing one stored event breaks `verify()` | R3.12 |
| `test_runs_without_langfuse` | no credentials → callbacks `[]`, run completes, audit complete | R3.14 |
| `test_undecided_candidate_is_refused_once_then_decided` | a case silent on a code-flagged typology candidate is sent back once and audited | R3.15 |
| `test_gate_refuses_at_most_once` | the typology gate never loops: a second undecided submission is accepted | R3.15 |
| `test_only_confirmed_or_rejected_with_evidence_counts_as_decided` | INCONCLUSIVE or evidence-free decisions leave a candidate open | R3.15 |
| `test_missing_or_null_indicator_never_fires` | no number, no candidate | R3.15 |
| `test_budget_exhaustion_yields_nmi` | tool budget hit → `NEEDS_MORE_INFO` with missing items | R3.1 |
| `test_llm_unavailable_degrades` | model raises → deterministic evidence pack, `NEEDS_MORE_INFO` | R3.1 |

## Fixtures
Tiny DuckDB with ~200 handcrafted transactions covering each typology, 3 injected sanctions cases (1 decoy), 2 injection memos and 3 media fixture docs (1 with payload).
