"""Deterministic disposition policy (task 3.6 · R3.7 · constitution III).

Runs after the reviewer, before the analyst. It can only move a disposition towards more
scrutiny (CLOSE < NEEDS_MORE_INFO < ESCALATE), never towards closing. Every override is
recorded with its rule id, and the analyst sees all of them.
"""

from __future__ import annotations

from pydantic import BaseModel

from finint.core.models import SEVERITY, Disposition, MatchStatus

MIN_EVIDENCE = 3
ADVERSE_CATEGORIES = frozenset({"CHARGE", "CONVICTION", "SANCTION"})


class PolicyInput(BaseModel):
    disposition: Disposition
    sanctions_statuses: list[MatchStatus]  # one per screened party
    n_evidence: int
    injection_attempts: int
    adverse_media_categories: list[str]  # verified, same-entity findings only


class Override(BaseModel):
    rule_id: str
    from_disposition: Disposition
    to_disposition: Disposition
    reason: str


class PolicyResult(BaseModel):
    disposition: Disposition
    overrides: list[Override]
    flags: list[str]


def _raise_to(current: Disposition, target: Disposition) -> Disposition:
    return target if SEVERITY[target] > SEVERITY[current] else current


def apply_policy(inp: PolicyInput) -> PolicyResult:
    disposition = inp.disposition
    overrides: list[Override] = []
    flags: list[str] = []

    def rule(rule_id: str, target: Disposition, reason: str) -> None:
        nonlocal disposition
        new = _raise_to(disposition, target)
        if new != disposition:
            overrides.append(
                Override(
                    rule_id=rule_id, from_disposition=disposition, to_disposition=new, reason=reason
                )
            )
            disposition = new

    if "MATCH" in inp.sanctions_statuses:
        rule("POL-01", "ESCALATE", "a party is a sanctions MATCH (decided by the screener)")
    if "POTENTIAL_MATCH" in inp.sanctions_statuses and disposition == "CLOSE_FALSE_POSITIVE":
        rule("POL-02", "NEEDS_MORE_INFO", "unresolved POTENTIAL_MATCH cannot be closed")
    if inp.n_evidence < MIN_EVIDENCE and disposition == "CLOSE_FALSE_POSITIVE":
        rule("POL-03", "NEEDS_MORE_INFO", f"fewer than {MIN_EVIDENCE} evidence items")
    if inp.injection_attempts:
        flags.append("INJECTION_ATTEMPT")
        if disposition == "CLOSE_FALSE_POSITIVE":
            rule(
                "POL-04",
                "NEEDS_MORE_INFO",
                "untrusted text tried to instruct the agent; a closure needs human review",
            )
    if ADVERSE_CATEGORIES & set(inp.adverse_media_categories):
        flags.append("ADVERSE_MEDIA")
        if disposition == "CLOSE_FALSE_POSITIVE":
            rule(
                "POL-05",
                "NEEDS_MORE_INFO",
                "verified adverse media (charge, conviction or sanction)",
            )
    return PolicyResult(disposition=disposition, overrides=overrides, flags=flags)
