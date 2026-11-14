---
name: sanctions-review
description: How to read sanctions screening results. Load before writing anything about a sanctions hit or when screen_sanctions returns MATCH or POTENTIAL_MATCH.
---
# Sanctions review

**You never decide a sanctions match.** `screen_sanctions` returns a `status` decided by code.
Your job is to report it accurately and to say what an analyst must check.

## Procedure
1. Screen the alerted account's owner and its largest counterparties (`screen_sanctions`).
2. For each candidate, report the `matched_name`, whether it matched a primary name or an alias,
   the `combined_score`, `matched_fields` and `mismatched_fields`, and the `status`.
3. Cite the screening evidence id in `sanctions_evidence_ids`.

## How to describe each status
- **MATCH**: "the screener reports a match on <fields>". A MATCH always leads to escalation; the
  policy guard enforces this even if you recommend otherwise.
- **POTENTIAL_MATCH**: "possible match; the analyst must verify <mismatched or missing fields>".
  Do not recommend closing the case while it is unresolved.
- **NO_MATCH with candidates**: same or similar name but contradicted by DOB/country; say so in
  one sentence. It is a false hit, not evidence.

## Never
- Never upgrade or downgrade a status, invent a score, or call someone "sanctioned" when the
  status is not MATCH.
