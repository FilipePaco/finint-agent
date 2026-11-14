"""Deterministic grounding validator (task 3.5 · R3.5 · constitution II).

A case file may only state amounts, dates, accounts and entity names that appear in the
tool outputs of the same investigation:

1. every cited ``evidence_id`` must exist in this run;
2. every ``cited_value`` of a finding must be in the evidence that finding cites;
3. every amount, date or account written anywhere in the free text (summary, claims,
   rationales, open questions) must be in the run's evidence.

Rule 3 is what catches a hallucinated number that the model did not declare as a citation.
An LLM may also critique the draft, but this check is code (ADR-03.5).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel

from finint.core.models import CaseFile, Evidence, EvidenceValues

ACCOUNT_RE = re.compile(r"\b\d{1,7}:[0-9A-F]{8,10}\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ][0-9:.]+)?\b")
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s+(\d{4})\b"
    r"|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
# Amounts in free text: a currency marker, a thousands separator or decimals. Plain integers
# ("3 transfers", "2 hops") are counts, not amounts, and are not checked.
_AMOUNT_RE = re.compile(
    r"(?:(?:USD|US\$|EUR|GBP|\$|€|£)\s?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"|\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b"
    r"|\b(\d+\.\d{2})\b(?!\s*%)"
)
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
AMOUNT_KEYS = ("amount", "sum", "usd", "total", "value", "inflow", "outflow")
NAME_KEYS = ("name", "synthetic_name", "matched_name", "alt_name", "counterparty_name")


def normalize_amount(raw: str | float | Decimal) -> str | None:
    text = str(raw).replace(",", "").replace("$", "").strip()
    try:
        return str(Decimal(text).quantize(Decimal("0.01")))
    except (InvalidOperation, ValueError):
        return None


def normalize_date(raw: str | date | datetime) -> str | None:
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    if m := _ISO_DATE_RE.search(raw):
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if m := _TEXT_DATE_RE.search(raw):
        day, mon, year = (
            (m.group(1), m.group(2), m.group(3))
            if m.group(1)
            else (m.group(5), m.group(4), m.group(6))
        )
        return f"{int(year):04d}-{_MONTHS[mon[:3].lower()]:02d}-{int(day):02d}"
    return None


def normalize_name(raw: str) -> str:
    return " ".join(raw.casefold().split())


def extract_values(data: Any) -> EvidenceValues:
    """Walk a tool output and collect normalized amounts, dates, accounts and names."""
    found = EvidenceValues()
    _walk(data, None, found)
    return found


def _walk(node: Any, key: str | None, found: EvidenceValues) -> None:
    if isinstance(node, BaseModel):
        node = node.model_dump()
    if isinstance(node, Mapping):
        for k, v in node.items():
            _walk(v, str(k).lower(), found)
        return
    if isinstance(node, list | tuple | set):
        for item in node:
            _walk(item, key, found)
        return
    if isinstance(node, datetime | date):
        if (d := normalize_date(node)) is not None:
            found.dates.add(d)
        return
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, int | float | Decimal):
        if key and any(k in key for k in AMOUNT_KEYS) and (a := normalize_amount(node)):
            found.amounts.add(a)
        return
    text = str(node)
    found.accounts.update(ACCOUNT_RE.findall(text))
    if key and any(k in key for k in NAME_KEYS):
        found.entities.add(normalize_name(text))
    if key and any(k in key for k in AMOUNT_KEYS) and (a := normalize_amount(text)):
        found.amounts.add(a)
    if (d := normalize_date(text)) is not None and (
        _ISO_DATE_RE.fullmatch(text.strip()) or (key and ("date" in key or key == "ts"))
    ):
        found.dates.add(d)


def text_values(text: str) -> EvidenceValues:
    """Amounts, dates and accounts written in free text."""
    found = EvidenceValues(accounts=set(ACCOUNT_RE.findall(text)))
    for m in _AMOUNT_RE.finditer(text):
        raw = next(g for g in m.groups() if g)
        if (a := normalize_amount(raw)) is not None:
            found.amounts.add(a)
    for m in _ISO_DATE_RE.finditer(text):
        found.dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    for m in _TEXT_DATE_RE.finditer(text):
        if (d := normalize_date(m.group(0))) is not None:
            found.dates.add(d)
    return found


class Violation(BaseModel):
    location: str
    kind: str
    value: str
    reason: str


class GroundingReport(BaseModel):
    passed: bool
    violations: list[Violation]
    n_values_checked: int


def _union(evidence: Iterable[Evidence]) -> EvidenceValues:
    out = EvidenceValues()
    for ev in evidence:
        out.amounts |= ev.values.amounts
        out.dates |= ev.values.dates
        out.accounts |= ev.values.accounts
        out.entities |= ev.values.entities
    return out


def _amount_in(value: str, amounts: set[str]) -> bool:
    target = Decimal(value)
    return any(abs(Decimal(a) - target) <= Decimal("0.01") for a in amounts)


def _contains(pool: EvidenceValues, kind: str, value: str) -> bool:
    if kind == "amount":
        norm = normalize_amount(value)
        return norm is not None and _amount_in(norm, pool.amounts)
    if kind == "date":
        norm_d = normalize_date(value)
        return norm_d is not None and norm_d in pool.dates
    if kind == "account":
        return value.strip() in pool.accounts
    return normalize_name(value) in pool.entities


def validate_grounding(case: CaseFile, evidence: Mapping[str, Evidence]) -> GroundingReport:
    violations: list[Violation] = []
    checked = 0
    for ev_id in sorted(case.cited_evidence_ids() - set(evidence)):
        violations.append(
            Violation(
                location="evidence_ids",
                kind="evidence_id",
                value=ev_id,
                reason="cited evidence does not exist in this run",
            )
        )
    for i, finding in enumerate(case.findings):
        cited = [evidence[e] for e in finding.evidence_ids if e in evidence]
        pool = _union(cited)
        for cv in finding.cited_values:
            checked += 1
            if not _contains(pool, cv.kind, cv.value):
                violations.append(
                    Violation(
                        location=f"findings[{i}]",
                        kind=cv.kind,
                        value=cv.value,
                        reason="not found in the evidence this finding cites",
                    )
                )
    run_pool = _union(evidence.values())
    texts = {"summary": case.summary}
    for i, finding in enumerate(case.findings):
        texts[f"findings[{i}].claim"] = finding.claim
    for i, hyp in enumerate(case.hypotheses):
        texts[f"hypotheses[{i}].rationale"] = hyp.rationale
    for i, question in enumerate(case.open_questions):
        texts[f"open_questions[{i}]"] = question
    for location, text in texts.items():
        found = text_values(text)
        for kind, values in (
            ("amount", found.amounts),
            ("date", found.dates),
            ("account", found.accounts),
        ):
            for value in sorted(values):
                checked += 1
                if not _contains(run_pool, kind, value):
                    violations.append(
                        Violation(
                            location=location,
                            kind=kind,
                            value=value,
                            reason="written in text but absent from all evidence",
                        )
                    )
    return GroundingReport(passed=not violations, violations=violations, n_values_checked=checked)
