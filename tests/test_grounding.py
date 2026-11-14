"""Grounding validator (task 3.5 · R3.5). Rejection tests from test-strategy.md."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from finint.core.grounding import extract_values, text_values, validate_grounding
from finint.core.models import CaseFile, CitedValue, Evidence, Finding


def _evidence(evidence_id: str, data: Any) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        tool="t",
        args={},
        summary="",
        data=data,
        values=extract_values(data),
    )


EVIDENCE = {
    "ev-1": _evidence(
        "ev-1",
        {
            "transactions": [
                {
                    "txn_id": 11,
                    "ts": datetime(2022, 9, 8, 10, 15),
                    "from_key": "952:8139F54E0",
                    "to_key": "111632:8062C56E0",
                    "amount_usd": Decimal("5331.44"),
                },
                {
                    "txn_id": 12,
                    "ts": datetime(2022, 9, 8, 10, 20),
                    "from_key": "952:8139F54E0",
                    "to_key": "118693:823D4E390",
                    "amount_usd": Decimal("4120.08"),
                },
            ],
            "total_out_usd": Decimal("9451.52"),
        },
    ),
    "ev-2": _evidence("ev-2", {"synthetic_name": "Velmora Freight Ltd", "n_accounts": 2}),
}


def _case(**overrides: Any) -> CaseFile:
    base: dict[str, Any] = {
        "alert_id": "AL-20220908-0001",
        "summary": "Account 952:8139F54E0 sent 5,331.44 USD and 4,120.08 USD on 2022-09-08.",
        "findings": [
            Finding(
                claim="Fan-out to two new counterparties totalling $9,451.52.",
                typology="FAN-OUT",
                evidence_ids=["ev-1"],
                cited_values=[
                    CitedValue(kind="amount", value="9451.52"),
                    CitedValue(kind="account", value="111632:8062C56E0"),
                    CitedValue(kind="date", value="8 Sep 2022"),
                ],
            )
        ],
        "disposition": "ESCALATE",
        "confidence": "MEDIUM",
    }
    base.update(overrides)
    return CaseFile(**base)


def test_grounded_case_passes() -> None:
    report = validate_grounding(_case(), EVIDENCE)
    assert report.passed, report.violations
    assert report.n_values_checked >= 6


def test_grounding_rejects_invented_amount() -> None:
    finding = (
        _case()
        .findings[0]
        .model_copy(update={"cited_values": [CitedValue(kind="amount", value="9999.99")]})
    )
    report = validate_grounding(_case(findings=[finding]), EVIDENCE)
    assert not report.passed
    assert report.violations[0].value == "9999.99"


def test_grounding_rejects_amount_in_narrative_only() -> None:
    case = _case(summary="The account moved USD 12,500.00 in one day.")
    report = validate_grounding(case, EVIDENCE)
    assert [(v.location, v.value) for v in report.violations] == [("summary", "12500.00")]


def test_grounding_rejects_unknown_evidence_id() -> None:
    finding = _case().findings[0].model_copy(update={"evidence_ids": ["ev-404"]})
    report = validate_grounding(_case(findings=[finding]), EVIDENCE)
    assert any(v.kind == "evidence_id" and v.value == "ev-404" for v in report.violations)


def test_grounding_rejects_invented_account_and_date_in_text() -> None:
    case = _case(summary="Funds went to 999:ABCDEF123 on 2022-09-30.")
    kinds = {v.kind for v in validate_grounding(case, EVIDENCE).violations}
    assert kinds == {"account", "date"}


def test_value_cited_from_the_wrong_evidence_is_rejected() -> None:
    finding = Finding(
        claim="Owned by Velmora Freight Ltd.",
        evidence_ids=["ev-1"],
        cited_values=[CitedValue(kind="entity", value="Velmora Freight Ltd")],
    )
    assert not validate_grounding(_case(findings=[finding]), EVIDENCE).passed
    ok = finding.model_copy(update={"evidence_ids": ["ev-2"]})
    assert validate_grounding(_case(findings=[ok]), EVIDENCE).passed


@pytest.mark.parametrize(
    "text",
    ["The sender made 3 transfers in 2 hops.", "Flow-through ratio was 0.98 (98%).", "Score 87%."],
)
def test_counts_ratios_and_percentages_are_not_amounts(text: str) -> None:
    assert not text_values(text).amounts - {"0.98"}


@pytest.mark.parametrize(
    ("text", "iso"),
    [
        ("on 2022-09-08", "2022-09-08"),
        ("on 8 Sep 2022", "2022-09-08"),
        ("on September 8, 2022", "2022-09-08"),
        ("at 2022-09-08T10:15:00", "2022-09-08"),
    ],
)
def test_date_formats_normalize(text: str, iso: str) -> None:
    assert text_values(text).dates == {iso}


@given(
    st.lists(
        st.decimals(min_value=1, max_value=10_000_000, places=2, allow_nan=False),
        min_size=1,
        max_size=10,
        unique=True,
    ),
    st.data(),
)
def test_property_cited_amounts_pass_and_perturbed_fail(
    amounts: list[Decimal], data: st.DataObject
) -> None:
    evidence = {"e": _evidence("e", {"rows": [{"amount": a} for a in amounts]})}
    chosen = data.draw(st.sampled_from(amounts))
    good = Finding(
        claim="x", evidence_ids=["e"], cited_values=[CitedValue(kind="amount", value=str(chosen))]
    )
    case = _case(summary="Summary without numbers.", findings=[good])
    assert validate_grounding(case, evidence).passed
    bad_value = chosen + Decimal("1.00")
    if all(abs(bad_value - a) > Decimal("0.01") for a in amounts):
        bad = good.model_copy(
            update={"cited_values": [CitedValue(kind="amount", value=str(bad_value))]}
        )
        assert not validate_grounding(_case(summary="No numbers.", findings=[bad]), evidence).passed
