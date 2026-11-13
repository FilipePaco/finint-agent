"""Patterns parser (task 1.3 · R1.1)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finint.data.patterns import PatternsFormatError, normalize_bank, parse_patterns

_TXN = (
    "2022/09/01 05:14,00952,8139F54E0,0111632,8062C56E0,5331.44,US Dollar,5331.44,US Dollar,ACH,1"
)


def test_parses_attempts_with_typology_and_detail() -> None:
    lines = [
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out",
        _TXN,
        "END LAUNDERING ATTEMPT - FAN-OUT",
        "",
        "BEGIN LAUNDERING ATTEMPT - STACK",
        _TXN,
        "END LAUNDERING ATTEMPT - STACK",
    ]
    attempts, txns = parse_patterns(lines)
    assert [(a.attempt_id, a.typology, a.detail) for a in attempts] == [
        (1, "FAN-OUT", "Max 16-degree Fan-Out"),
        (2, "STACK", ""),
    ]
    assert [t.attempt_id for t in txns] == [1, 2]
    first = txns[0]
    assert first.from_bank == "952"
    assert first.to_bank == "111632"
    assert first.amount_paid == Decimal("5331.44")
    assert first.payment_format == "ACH"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([_TXN], "outside an attempt"),
        (["END LAUNDERING ATTEMPT - CYCLE"], "END without BEGIN"),
        (["BEGIN LAUNDERING ATTEMPT - CYCLE", "BEGIN LAUNDERING ATTEMPT - CYCLE"], "BEGIN inside"),
        (["BEGIN LAUNDERING ATTEMPT - CYCLE", _TXN], "ended inside"),
        (["BEGIN LAUNDERING ATTEMPT - CYCLE", "a,b,c"], "expected 11 fields"),
    ],
)
def test_rejects_malformed_files(lines: list[str], message: str) -> None:
    with pytest.raises(PatternsFormatError, match=message):
        parse_patterns(lines)


@pytest.mark.parametrize(
    ("raw", "expected"), [("010", "10"), ("0111632", "111632"), ("000", "0"), (" 7 ", "7")]
)
def test_normalize_bank_strips_zero_padding(raw: str, expected: str) -> None:
    assert normalize_bank(raw) == expected
