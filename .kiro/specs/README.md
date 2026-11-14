# Specs: index, plan and consistency check

Method: Kiro-style specs (`requirements.md` in EARS → `design.md` with ADRs → `tasks.md`, plus `test-strategy.md` where guardrails live) governed by the Spec Kit constitution (`.specify/memory/constitution.md`). Open questions are marked `[NEEDS CLARIFICATION]` (Spec Kit `/clarify`) and must be resolved before implementing that spec.

| Spec | Delivers | Depends on |
|---|---|---|
| [01-data-foundation](01-data-foundation/) | DuckDB store, synthetic identities, OFAC, injections | none |
| [02-ml-detection](02-ml-detection/) | 3 models, calibration, capacity threshold, alerts | 01 |
| [03-investigation-agent](03-investigation-agent/) | 3-agent LangGraph + deterministic guards + HITL console | 01, 02 |
| [04-evaluation-and-reporting](04-evaluation-and-reporting/) | eval, ops report, architecture PDF, README, demo | 02, 03 |

## 5-day plan (Nov 12 → Nov 16, 2026)
| Day | Focus | Exit criterion |
|---|---|---|
| 1 | 01 complete; 02 split + tabular features + M0 | `make data` works; M0 PR-AUC on val |
| 2 | 02 graph features, M1, calibration, threshold, alerts; GNN time-boxed | `ml_report.md` with M0 vs M1 (vs M2) |
| 3 | 03 core (grounding, screening, policy, audit, minimize) + tools + investigator | `finint investigate <id>` produces a grounded case |
| 4 | 03 media agent, reviewer, graph wiring, console; 04 eval set + runner | `make eval` on the default model |
| 5 | 04 second model + ablations, ops report, architecture PDF, README, demo | repo public, README numbers filled |

**Cut line if late (in order):** demo → GNN → ablations → second model. Never cut: grounding, HITL audit, sanctions determinism, eval.

## Reuse from earlier projects
See `.kiro/steering/tech.md` → *Reused from previous projects*. Tasks marked **port from** start from that code instead of from scratch.

## Consistency check (Spec Kit `/analyze`)
| Constitution principle | Enforced by |
|---|---|
| I. The human decides | R3.8, R3.11, read-only tools (R3.1) |
| II. Grounding in code | R3.5, R4.2 faithfulness |
| III. Deterministic where it matters | R2.7, R3.4, R3.7 |
| IV. Untrusted input stays data | R1.6, R3.3, R3.10, R4.2 injection |
| V. Minimum necessary data | R1.7, R3.9, R3.11 |
| VI. Auditable and replayable | R2.8, R3.12 (audit), R3.14 (Langfuse observability), R4.4 replay |
| VII. Measured, not claimed | R2.5, R4.2–R4.4 |
| VIII. Cost first-class | R3.12, R3.14, R4.3 |
| IX. Reputational care | R1.6, R3.3 |
| X. Simple, typed, tested | `core/` framework-free; tests in every task list |

| Challenge requirement | Covered by |
|---|---|
| Model you train; tabular vs graph; time split; precision@budget, PR-AUC, recall per pattern | R2.1–R2.5 |
| Calibrate + capacity-based threshold, justified | R2.6, R2.7 |
| Alerts logged with features, score, threshold, model version | R2.8 |
| Agent with tools, sanctions screening, adverse media, case file + disposition | R3.1–R3.4 |
| ≥ 50-alert eval: accuracy, faithfulness in code, pattern recall, latency, cost; single command/CI | R4.1–R4.4 |
| Weekly ops report with 5 metrics + 2 charts | R4.5 |
| HITL, faithfulness, sanctions FP, tipping-off, minimization, injection, reputational risk | R3.3–R3.11 |
| Architecture PDF; README; bonus hosted demo | R4.6 |

Known gaps / risks: HI-Small covers only ~18 days (history for the charts is simulated and labeled); analyst decisions are simulated from labels; the manual-time baseline is an assumption; the PoC uses the Gemini free tier (synthetic data only) and compares two Gemini tiers instead of a local open-weight model (no disk for local weights).
