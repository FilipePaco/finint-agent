"""Controlled typology vocabulary for hypotheses (R3.1, R3.2) and robust pattern scoring (R4.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from finint.agents.investigator import tool_specs
from finint.core.models import CaseFile, Hypothesis
from finint.core.typology import OTHER, TYPOLOGIES, normalize_typology
from finint.eval.dataset import TYPOLOGY_TO_SKILL
from finint.eval.metrics import pattern_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_vocabulary_is_the_typology_skills_plus_other() -> None:
    """R3.2: every canonical typology has a skill; R4.2: ground truth uses the same names."""
    skills = {p.name for p in (ROOT / "skills").iterdir() if (p / "SKILL.md").is_file()}
    assert set(TYPOLOGIES) - {OTHER} <= skills
    assert set(TYPOLOGY_TO_SKILL.values()) <= set(TYPOLOGIES)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("fan-in", "fan-in"),
        ("Fan-In", "fan-in"),
        ("fan in", "fan-in"),
        ("FAN_OUT", "fan-out"),
        ("Fan-Out / Dispersal", "fan-out"),
        ("Gather Scatter", "scatter-gather"),
        ("scatter/gather", "scatter-gather"),
        ("Cycle (round-tripping)", "cycle"),
        ("round_tripping", "cycle"),
        ("layering", "stack"),
        ("Stack / Layering", "stack"),
        ("RANDOM", "stack"),
        ("smurfing/structuring", "structuring"),
        ("Structuring / Threshold avoidance", "structuring"),
        # Two different typologies, or none, are not guessed.
        ("Structuring / Layering", "other"),
        ("structuring-and-layering", "other"),
        ("Sanctions Match", "other"),
        ("sanctions-evasion-/-designated-entity", "other"),
        ("Large Cash and Cheque Movements", "other"),
        ("", "other"),
    ],
)
def test_normalize_typology(raw: str, canonical: str) -> None:
    """R4.2: legacy and alias labels map deterministically to the vocabulary."""
    assert normalize_typology(raw) == canonical


def test_hypothesis_typology_is_canonical() -> None:
    """R3.1: hypotheses carry a canonical typology; aliases are normalised on validation."""
    assert Hypothesis(typology="Fan-In", status="CONFIRMED").typology == "fan-in"
    assert Hypothesis(typology="Sanctions Match", status="CONFIRMED").typology == "other"
    with pytest.raises(ValidationError):
        Hypothesis.model_validate({"typology": 3, "status": "CONFIRMED"})


def test_submit_case_schema_exposes_the_vocabulary() -> None:
    """R3.1: the LLM sees the vocabulary as an enum in the submit_case tool schema."""
    spec = tool_specs(["submit_case"])[0]["function"]["parameters"]
    hypothesis = spec["properties"]["hypotheses"]["items"]
    hypothesis = spec.get("$defs", {}).get("Hypothesis", hypothesis)
    assert hypothesis["properties"]["typology"]["enum"] == list(TYPOLOGIES)


def test_investigator_prompt_lists_the_vocabulary() -> None:
    """R3.1: the v2 investigator prompt names every canonical typology."""
    prompt = (ROOT / "prompts" / "investigator.md").read_text("utf-8")
    for typology in TYPOLOGIES:
        assert f"`{typology}`" in prompt


def test_case_file_dump_is_canonical() -> None:
    case = CaseFile.model_validate(
        {
            "alert_id": "A",
            "summary": "s",
            "hypotheses": [{"typology": "Cycle (round-tripping)", "status": "REJECTED"}],
            "disposition": "ESCALATE",
            "confidence": "LOW",
        }
    )
    assert case.model_dump()["hypotheses"][0]["typology"] == "cycle"


def _row(expected: list[str], confirmed: list[str]) -> dict[str, Any]:
    return {
        "is_laundering": True,
        "expected_typologies": expected,
        "confirmed_typologies": confirmed,
    }


def test_pattern_metrics_rescore_legacy_labels() -> None:
    """R4.2: recorded free-text labels are normalised, so old results re-score offline."""
    rows = [
        _row(["cycle", "fan-in"], ["Cycle (round-tripping)", "Fan-In"]),
        _row(["stack"], ["layering"]),
    ]
    p = pattern_metrics(rows)
    assert p["recall"] == 1.0
    assert p["precision"] == 1.0


def test_pattern_precision_ignores_non_typology_hypotheses() -> None:
    """R4.2: 'other' (e.g. a sanctions concern) is not a typology claim."""
    p = pattern_metrics([_row(["fan-out"], ["fan-out", "sanctions-match", "other"])])
    assert p["recall"] == 1.0
    assert p["precision"] == 1.0
