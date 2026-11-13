"""Normalization: FX, flags, shifted timestamps (task 1.4 · R1.2)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from finint.config import load_yaml
from finint.data.ingest import ingest
from finint.data.normalize import NormalizeReport, normalize, week_aligned_shift
from finint.data.sources import AmlWorldFiles

FIXTURES = Path(__file__).parent / "fixtures" / "amlworld"
REPO_CONFIG = Path(__file__).parents[1] / "config"
AS_OF = date(2026, 11, 13)  # a Friday


@pytest.fixture
def fallback_rates() -> dict[str, float]:
    rates: dict[str, float] = load_yaml("fx", REPO_CONFIG)["usd_rates"]
    return rates


@pytest.fixture
def normalized(tmp_path: Path, fallback_rates: dict[str, float]) -> tuple[Path, NormalizeReport]:
    path = tmp_path / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(FIXTURES), path)
    return path, normalize(path, AS_OF, fallback_rates)


def test_week_aligned_shift_preserves_weekday() -> None:
    last = date(2022, 9, 18)
    shift = week_aligned_shift(last, AS_OF)
    assert shift % 7 == 0
    shifted = date.fromordinal(last.toordinal() + shift)
    assert shifted <= AS_OF
    assert (AS_OF - shifted).days < 7
    assert shifted.weekday() == last.weekday()


def test_fixture_rates_fall_back_to_config(normalized: tuple[Path, NormalizeReport]) -> None:
    _, report = normalized
    # Fixtures have no currency with >= 10 USD pairs, so non-USD rates come from config.
    assert report.fx_rates["US Dollar"] == 1.0
    assert set(report.fx_fallbacks) == {"Bitcoin", "Euro"}
    assert report.n_missing_usd == 0


def test_amount_usd_and_flags(normalized: tuple[Path, NormalizeReport]) -> None:
    path, _ = normalized
    with duckdb.connect(str(path), read_only=True) as con:
        rows = dict(
            (r[0], r[1:])
            for r in con.execute(
                "SELECT txn_id, amount_usd, cross_currency, cross_bank, self_transfer "
                "FROM transactions"
            ).fetchall()
        )
    assert rows[1] == (Decimal("3697.34"), False, False, True)
    assert rows[6] == (Decimal("9400.25"), True, True, False)  # paid in USD, received EUR
    assert rows[7][0] == Decimal("10253.25")  # 8750 EUR x 1.1718


def test_shifted_timestamps_keep_order_and_end_before_as_of(
    normalized: tuple[Path, NormalizeReport],
) -> None:
    path, report = normalized
    first, last = report.shifted_range
    assert last <= AS_OF
    with duckdb.connect(str(path), read_only=True) as con:
        deltas = con.execute(
            "SELECT DISTINCT date_diff('day', ts, ts_shifted) FROM transactions"
        ).fetchall()
    assert deltas == [(report.shift_days,)]
    assert first <= last


def test_rates_are_derived_from_pairs_when_enough_observations(
    tmp_path: Path, fallback_rates: dict[str, float]
) -> None:
    path = tmp_path / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(FIXTURES), path)
    with duckdb.connect(str(path)) as con:
        for i in range(12):
            con.execute(
                "INSERT INTO transactions (txn_id, ts, from_bank, from_account, to_bank, "
                "to_account, amount_received, currency_received, amount_paid, currency_paid, "
                "payment_format, is_laundering, from_key, to_key) VALUES "
                "(?, TIMESTAMP '2022-09-05 10:00', '1', 'A', '2', 'B', ?, 'US Dollar', ?, "
                "'Euro', 'Wire', false, '1:A', '2:B')",
                [100 + i, Decimal("120.00"), Decimal("100.00")],
            )
    report = normalize(path, AS_OF, fallback_rates)
    assert report.fx_rates["Euro"] == pytest.approx(1.2)
    assert "Euro" not in report.fx_fallbacks


def test_normalize_is_idempotent(
    normalized: tuple[Path, NormalizeReport], fallback_rates: dict[str, float]
) -> None:
    path, first = normalized
    again = normalize(path, AS_OF, fallback_rates)
    assert again == first
    with duckdb.connect(str(path), read_only=True) as con:
        columns = [r[0] for r in con.execute("DESCRIBE transactions").fetchall()]
    assert len(columns) == len(set(columns))
