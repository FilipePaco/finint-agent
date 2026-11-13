# finint-agent Constitution

Non-negotiable principles. Every spec, plan, task and pull request must comply. A design that needs to break a principle must amend this file first, with a written rationale.

Version: 1.0.0 · Ratified: 2026-11-12

---

## I. The human decides

The agent **recommends**; it never closes an alert, files a report, blocks an account or contacts a customer. No tool with side effects outside the investigation store is ever given to an LLM. Every case reaches a final state only through an analyst action (`APPROVE`, `EDIT`, `REJECT`), and that action is written to the audit trail with analyst id, timestamp, before/after diff and reason.

## II. Grounding is enforced in code

Every amount, date, account id and entity name in a case file must trace back to a specific tool output from the same investigation. A deterministic validator checks this before a case file reaches the analyst. An LLM may *also* critique a draft, but an LLM never replaces the deterministic check. A case file that fails validation is blocked, not "flagged".

## III. Deterministic where it matters

Decisions with legal or compliance weight are made by code with explicit thresholds, never by the LLM alone:
- sanctions match / no-match (fuzzy score + matched fields, threshold in config);
- alert threshold (derived from analyst capacity);
- disposition policy overrides (e.g., a sanctions hit above threshold can never be `CLOSE_FALSE_POSITIVE`).

The LLM may explain these results; it cannot change them.

## IV. Untrusted input stays data

Transaction references, counterparty names, news articles and search results are untrusted. They are passed to models only inside clearly delimited data blocks, never concatenated into instructions, and processed by the least-privileged component possible (the adverse-media agent has no access to transaction tools). A successful prompt injection must not be able to change a disposition, call a tool outside the agent's allowlist, or remove evidence.

## V. Minimum necessary data

The LLM receives only the fields an investigation step needs, through an explicit allowlist. Synthetic names are the only names in the system. Case files live in the investigation store, are never exposed through any customer-facing surface, and never contain language that could tip off a customer.

## VI. Everything is auditable and replayable

Every alert is logged with features, score, threshold and model version. Every agent run logs each LLM call (model id, prompt version, tokens, cost), each tool call (name, arguments, output reference) and each decision. The audit log is append-only and hash-chained. Given an audit record, we can explain *why* the system did what it did.

Two layers, never confused: the **audit log** is ours and is the system of record for compliance. **Observability** (traces, latency, cost dashboards) uses an existing tool (Langfuse) and is optional. Turning observability off never degrades the audit log, and observability is never the only witness of a decision. We don't hand-build what a maintained tool already does well.

## VII. Measured, not claimed

Every capability claimed in the README has a metric produced by a single command (`make eval`, `make train`), with numbers reported as they came out. ML is judged by PR-AUC, precision at alert budget and recall per laundering pattern, never accuracy or ROC-AUC alone. The agent is judged by disposition accuracy, faithfulness, pattern recall, latency and cost per case.

## VIII. Cost is a first-class metric

Tokens, searches and latency are tracked per case. The LLM provider is swappable behind one interface, and the evaluation reports quality × cost for at least two models (a hosted model and an open-weight local model).

## IX. Reputational care

Adverse-media results are cited with source URL and retrieval date, are only run against injected synthetic test entities or public enforcement examples, and are never summarized into claims the source does not support.

## X. Simple, typed, tested

Python 3.12, type hints, Pydantic models at every boundary, small modules. Critical rules (grounding, screening, policy, minimization) live in plain Python, independent of the agent framework, and are unit-tested. Prefer a smaller thing that works and is measured over a larger thing that is half done.
