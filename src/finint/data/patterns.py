"""Parser for AMLworld ``*_Patterns.txt`` (laundering attempts and their typology).

File shape::

    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    2022/09/01 05:14,00952,8139F54E0,0111632,8062C56E0,5331.44,US Dollar,5331.44,US Dollar,ACH,1
    ...
    END LAUNDERING ATTEMPT - FAN-OUT
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - (?P<typology>[A-Z-]+)(?::\s*(?P<detail>.*))?$")
_END = re.compile(r"^END LAUNDERING ATTEMPT")
_TS_FORMAT = "%Y/%m/%d %H:%M"


class PatternsFormatError(ValueError):
    """Raised when the patterns file does not follow the BEGIN/rows/END structure."""


@dataclass(frozen=True, slots=True)
class PatternTxn:
    """One transaction line inside a laundering attempt."""

    attempt_id: int
    typology: str
    ts: datetime
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: Decimal
    currency_received: str
    amount_paid: Decimal
    currency_paid: str
    payment_format: str


@dataclass(frozen=True, slots=True)
class Attempt:
    attempt_id: int
    typology: str
    detail: str


def normalize_bank(bank: str) -> str:
    """Bank ids appear zero-padded inconsistently across files; compare them unpadded."""
    stripped = bank.strip().lstrip("0")
    return stripped or "0"


def parse_patterns(lines: Iterable[str]) -> tuple[list[Attempt], list[PatternTxn]]:
    """Parse attempts and their transaction lines. Attempt ids are 1-based, in file order."""
    attempts: list[Attempt] = []
    txns: list[PatternTxn] = []
    current: Attempt | None = None
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if begin := _BEGIN.match(line):
            if current is not None:
                raise PatternsFormatError(f"line {line_no}: BEGIN inside an open attempt")
            current = Attempt(
                attempt_id=len(attempts) + 1,
                typology=begin["typology"],
                detail=(begin["detail"] or "").strip(),
            )
            attempts.append(current)
        elif _END.match(line):
            if current is None:
                raise PatternsFormatError(f"line {line_no}: END without BEGIN")
            current = None
        else:
            if current is None:
                raise PatternsFormatError(f"line {line_no}: transaction outside an attempt")
            txns.append(_parse_txn(line, current, line_no))
    if current is not None:
        raise PatternsFormatError("file ended inside an open attempt")
    return attempts, txns


def iter_pattern_file(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        yield from handle


def _parse_txn(line: str, attempt: Attempt, line_no: int) -> PatternTxn:
    fields = line.split(",")
    if len(fields) != 11:
        raise PatternsFormatError(f"line {line_no}: expected 11 fields, got {len(fields)}")
    ts, fbank, facc, tbank, tacc, amt_rec, cur_rec, amt_paid, cur_paid, fmt, _label = fields
    return PatternTxn(
        attempt_id=attempt.attempt_id,
        typology=attempt.typology,
        ts=datetime.strptime(ts, _TS_FORMAT),
        from_bank=normalize_bank(fbank),
        from_account=facc.strip(),
        to_bank=normalize_bank(tbank),
        to_account=tacc.strip(),
        amount_received=Decimal(amt_rec),
        currency_received=cur_rec.strip(),
        amount_paid=Decimal(amt_paid),
        currency_paid=cur_paid.strip(),
        payment_format=fmt.strip(),
    )
