"""Domain models shared by tools, agents, the console and the evaluation (task 3.1).

Framework-free (constitution X). Every LLM output is validated against these models.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Disposition = Literal["CLOSE_FALSE_POSITIVE", "ESCALATE", "NEEDS_MORE_INFO"]
Confidence = Literal["LOW", "MEDIUM", "HIGH"]
ValueKind = Literal["amount", "date", "account", "entity"]
MatchStatus = Literal["MATCH", "POTENTIAL_MATCH", "NO_MATCH"]

SEVERITY: dict[str, int] = {"CLOSE_FALSE_POSITIVE": 0, "NEEDS_MORE_INFO": 1, "ESCALATE": 2}
SUMMARY_MAX_WORDS = 150


class EvidenceValues(BaseModel):
    """Normalized values found in a tool output; the grounding validator checks against these."""

    amounts: set[str] = Field(default_factory=set)  # Decimal strings, 2 places
    dates: set[str] = Field(default_factory=set)  # ISO dates
    accounts: set[str] = Field(default_factory=set)  # bank:account keys
    entities: set[str] = Field(default_factory=set)  # casefolded names


class Evidence(BaseModel):
    """One read-only tool output, registered under an id the case file can cite."""

    evidence_id: str
    tool: str
    args: dict[str, Any]
    summary: str  # what the LLM sees (bounded)
    data: Any  # raw output (what the validator and the analyst see)
    values: EvidenceValues
    untrusted: bool = False  # contains attacker-controllable text (memos, names, news)


class CitedValue(BaseModel):
    kind: ValueKind
    value: str


class Finding(BaseModel):
    claim: str
    typology: str | None = None
    evidence_ids: list[str] = Field(min_length=1)
    cited_values: list[CitedValue] = Field(default_factory=list)


class Hypothesis(BaseModel):
    typology: str
    status: Literal["CONFIRMED", "REJECTED", "INCONCLUSIVE"]
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class CaseFile(BaseModel):
    """The investigator's structured output."""

    alert_id: str
    summary: str
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    sanctions_evidence_ids: list[str] = Field(default_factory=list)
    adverse_media_evidence_ids: list[str] = Field(default_factory=list)
    benign_explanations_considered: list[str] = Field(default_factory=list)
    disposition: Disposition
    confidence: Confidence
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _bounded_summary(cls, summary: str) -> str:
        if len(summary.split()) > SUMMARY_MAX_WORDS:
            raise ValueError(f"summary must be at most {SUMMARY_MAX_WORDS} words")
        return summary

    def cited_evidence_ids(self) -> set[str]:
        ids = set(self.sanctions_evidence_ids) | set(self.adverse_media_evidence_ids)
        for finding in self.findings:
            ids.update(finding.evidence_ids)
        for hypothesis in self.hypotheses:
            ids.update(hypothesis.evidence_ids)
        return ids


class AnalystDecision(BaseModel):
    """The human decision that closes every case (R3.8)."""

    action: Literal["APPROVE", "EDIT", "REJECT"]
    analyst_id: str = Field(min_length=1, max_length=64)
    reason: str = ""
    final_disposition: Disposition
    edited_summary: str | None = None

    @model_validator(mode="after")
    def _reason_required(self) -> AnalystDecision:
        if self.action in ("EDIT", "REJECT") and not self.reason.strip():
            raise ValueError(f"{self.action} requires a reason")
        return self
