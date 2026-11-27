"""Code-computed typology candidates from multi-account indicators (R3.15 · ADR-03.11).

``detect_patterns`` computes numbers (two-hop convergence, hubs, amount-matched chains); this
module turns them into the canonical typologies those numbers support, with explicit
thresholds from ``config/typology.yaml``. A candidate is evidence, not a decision: the agent
must state each one as a hypothesis and confirm or reject it with cited evidence, and
:func:`undecided_candidates` lets the harness check that it did. Framework-free
(constitution X).

A rule fires when the indicator at its dotted ``indicator`` path is ``>=`` (``min``) or
``<=`` (``max``) its threshold; a missing or null indicator never fires. A typology is a
candidate when any of its rules fires.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from finint.core.models import Hypothesis
from finint.core.typology import TYPOLOGIES, Typology

DECIDED = ("CONFIRMED", "REJECTED")


class Rule(BaseModel):
    typology: Typology
    indicator: str  # dotted path into the indicators mapping
    min: float | None = None
    max: float | None = None


class Candidate(BaseModel):
    typology: Typology
    indicator: str
    observed: float | int
    rule: str  # ">= 4" or "<= 12", shown to the agent as is


def load_rules(config: Mapping[str, Any]) -> list[Rule]:
    """Parse ``config/typology.yaml``. Unknown typologies or rules without a bound are errors."""
    rules = []
    for raw in config.get("rules", []):
        rule = Rule.model_validate(raw)
        if rule.typology not in TYPOLOGIES or rule.typology == "other":
            raise ValueError(f"rule for non-candidate typology {rule.typology!r}")
        if (rule.min is None) == (rule.max is None):
            raise ValueError(f"rule {rule.indicator!r} needs exactly one of min / max")
        rules.append(rule)
    return rules


def _lookup(indicators: Mapping[str, Any], path: str) -> float | None:
    node: Any = indicators
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def typology_candidates(indicators: Mapping[str, Any], rules: Iterable[Rule]) -> list[Candidate]:
    """Typologies the numbers support: one entry per typology (its first firing rule)."""
    found: dict[str, Candidate] = {}
    for rule in rules:
        if rule.typology in found:
            continue
        value = _lookup(indicators, rule.indicator)
        if value is None:
            continue
        if rule.min is not None and value >= rule.min:
            threshold = f">= {_fmt(rule.min)}"
        elif rule.max is not None and value <= rule.max:
            threshold = f"<= {_fmt(rule.max)}"
        else:
            continue
        found[rule.typology] = Candidate(
            typology=rule.typology,
            indicator=rule.indicator,
            observed=int(value) if value.is_integer() else value,
            rule=threshold,
        )
    return [found[t] for t in TYPOLOGIES if t in found]


def undecided_candidates(candidates: Iterable[str], hypotheses: Iterable[Hypothesis]) -> list[str]:
    """Candidate typologies the case file does not mark CONFIRMED or REJECTED (with evidence).

    ``INCONCLUSIVE`` or a hypothesis without evidence ids does not count as a decision.
    """
    decided = {h.typology for h in hypotheses if h.status in DECIDED and h.evidence_ids}
    return sorted({c for c in candidates if c not in decided})
