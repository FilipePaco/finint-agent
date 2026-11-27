"""Code typology candidates and the decision check (task 3.23 · R3.15 · ADR-03.11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finint.config import load_yaml
from finint.core.models import Hypothesis
from finint.core.typology_signals import (
    Rule,
    load_rules,
    typology_candidates,
    undecided_candidates,
)

ROOT = Path(__file__).parents[1]
INDICATORS = {
    "fan": {"distinct_receivers": 3},
    "multi": {
        "scatter_gather_intermediary": {"parallel_intermediaries": 6},
        "amount_matched_chain": {"cycle_length": None, "forward_depth": 1},
    },
}


def test_rule_fires_on_min_and_max_and_reports_the_number() -> None:
    """R3.15: candidates come from explicit thresholds, with the value and the rule shown."""
    rules = [
        Rule(
            typology="scatter-gather",
            indicator="multi.scatter_gather_intermediary.parallel_intermediaries",
            min=5,
        ),
        Rule(typology="fan-out", indicator="fan.distinct_receivers", max=3),
    ]
    found = typology_candidates(INDICATORS, rules)
    assert [(c.typology, c.observed, c.rule) for c in found] == [
        ("fan-out", 3, "<= 3"),
        ("scatter-gather", 6, ">= 5"),
    ]


def test_missing_or_null_indicator_never_fires() -> None:
    """R3.15: no number, no candidate (a null cycle length is not a cycle)."""
    rules = [
        Rule(typology="cycle", indicator="multi.amount_matched_chain.cycle_length", max=12),
        Rule(typology="stack", indicator="multi.nope.depth", min=0),
        Rule(typology="stack", indicator="multi.amount_matched_chain.forward_depth", min=2),
    ]
    assert typology_candidates(INDICATORS, rules) == []


def test_bad_rules_are_rejected() -> None:
    """R3.15: a rule needs exactly one bound and a real typology."""
    with pytest.raises(ValueError, match="exactly one"):
        load_rules({"rules": [{"typology": "cycle", "indicator": "x", "min": 1, "max": 2}]})
    with pytest.raises(ValueError, match="non-candidate"):
        load_rules({"rules": [{"typology": "other", "indicator": "x", "min": 1}]})


def test_shipped_config_covers_the_graph_typologies() -> None:
    """R3.15: config/typology.yaml loads; bipartite and structuring are not code-flagged."""
    rules = load_rules(load_yaml("typology", ROOT / "config"))
    assert {r.typology for r in rules} == {"scatter-gather", "cycle", "fan-in", "fan-out", "stack"}


def test_only_confirmed_or_rejected_with_evidence_counts_as_decided() -> None:
    """R3.15: INCONCLUSIVE, or a decision without evidence ids, leaves the candidate open."""
    hyps = [
        Hypothesis(typology="scatter-gather", status="INCONCLUSIVE", evidence_ids=["ev-002"]),
        Hypothesis(typology="fan-in", status="REJECTED", evidence_ids=[]),
        Hypothesis(typology="cycle", status="CONFIRMED", evidence_ids=["ev-002"]),
        Hypothesis(typology="stack", status="REJECTED", evidence_ids=["ev-002"]),
    ]
    assert undecided_candidates(["scatter-gather", "fan-in", "cycle", "stack"], hyps) == [
        "fan-in",
        "scatter-gather",
    ]
