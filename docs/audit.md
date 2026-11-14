# Reading the audit trail

Every decision in finint-agent leaves an event in an **append-only, hash-chained** audit log
(`data/audit.sqlite`). It is the compliance system of record (constitution VI); optional
Langfuse traces are for debugging and never the only witness.

## Event types

| Event | Actor | What it records |
|---|---|---|
| `ALERT_CREATED` | `ml:<model_version>` | score, threshold, analyst capacity, model version, SHAP top-5, alert run id |
| `RUN_STARTED` | `system:graph` | models per role, prompt versions (file hashes), tool budget |
| `LLM_CALL` | `agent:<role>` | model, prompt version, input/output tokens, latency, estimated cost, errors |
| `TOOL_CALL` | `agent:investigator` | tool, arguments, evidence id, injection hits, refusals (off-allowlist) |
| `SKILL_LOADED` | `agent:investigator` | which playbook the agent loaded |
| `MEDIA_CLAIM_DROPPED` | `system:media-verifier` | a news claim whose quote was not verbatim in the document |
| `GROUNDING_RESULT` | `system:grounding` | pass/fail, attempt, values checked, each violation |
| `REVIEW_VERDICT` | `agent:reviewer` | ACCEPT/REVISE, objections, evidence requests |
| `POLICY_OVERRIDE` | `system:policy` | rule id, disposition before and after, reason |
| `CASE_DRAFTED` | `system:graph` | the full case file, recommended and policy disposition, flags |
| `ANALYST_DECISION` | `analyst:<id>` | action, final disposition, reason, edited summary, whether it changed the recommendation |
| `RUN_COMPLETED` | `system:graph` | tokens, cost, tool calls, evidence items, gathering time, final disposition |

## Verify integrity

```bash
make audit-verify          # recomputes every hash; exits 1 if any event was edited or removed
```

Each event stores `hash = sha256(canonical(event fields + prev_hash))`. Changing a payload,
deleting a row or reordering events breaks the chain at that point.

## Follow one case

```python
from finint.config import get_settings
from finint.core.audit import AuditLog

log = AuditLog(get_settings().audit_db_path)
for e in log.events(alert_id="AL-20220909-0001"):
    print(e.seq, e.occurred_at, e.event_type, e.actor, e.payload)
```

The console shows the same trail per case (**Audit trail** tab), and the **Verify audit
chain** button runs the integrity check.

## Link to traces

The investigation `run_id` (32 hex chars) is also the Langfuse `trace_id`, so a compliance
record can be opened as a trace when tracing is enabled.
