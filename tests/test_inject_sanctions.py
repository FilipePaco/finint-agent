"""Sanctions near-match injection (task 1.7 · R1.5)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import duckdb
import pytest

from finint.config import load_yaml
from finint.data.identities import build_entities
from finint.data.ingest import ingest
from finint.data.inject import SanctionsCase, inject_sanctions, load_sanctions_cases
from finint.data.sanctions import _read_rows
from finint.data.sources import AmlWorldFiles
from finint.data.split import SplitFractions, split_boundaries

ROOT = Path(__file__).parents[1]
CASES = ROOT / "data" / "injections" / "sanctions_cases.yaml"
SNAPSHOT = ROOT / "data" / "fixtures" / "sdn_snapshot"
FIXTURES = Path(__file__).parent / "fixtures" / "amlworld"


@pytest.fixture(scope="module")
def cases() -> list[SanctionsCase]:
    return load_sanctions_cases(CASES)


def test_case_file_meets_spec_minimums(cases: list[SanctionsCase]) -> None:
    expected = Counter(c.expected for c in cases)
    assert len(cases) >= 20
    assert expected["FALSE_MATCH"] >= 5
    variants = {c.variant for c in cases}
    assert {
        "transliteration",
        "missing_middle_name",
        "swapped_order",
        "diacritics_spacing",
    } <= variants


def test_every_case_points_to_a_real_snapshot_entry(cases: list[SanctionsCase]) -> None:
    entries = {int(r[0]): r for r in _read_rows(SNAPSHOT / "SDN.CSV.gz")}
    aliases: dict[int, list[str]] = {}
    for r in _read_rows(SNAPSHOT / "ALT.CSV.gz"):
        aliases.setdefault(int(r[0]), []).append(str(r[3]))
    for case in cases:
        assert case.sdn_ent_num in entries, case.id
        is_individual = (entries[case.sdn_ent_num][2] or "").lower() == "individual"
        assert is_individual == (case.kind == "person"), case.id
        if case.variant == "alias":
            assert aliases.get(case.sdn_ent_num), f"{case.id}: SDN entry has no alias"


def test_person_case_requires_dob() -> None:
    with pytest.raises(ValueError, match="dob"):
        SanctionsCase(
            id="X", sdn_ent_num=1, variant="alias", expected="TRUE_MATCH", kind="person",
            role="subject", name="A B", country="Iran", country_iso2="IR",
        )  # fmt: skip


def test_injection_on_fixture_renames_a_test_window_entity(tmp_path: Path) -> None:
    db = tmp_path / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(FIXTURES), db)
    build_entities(
        db, load_yaml("identities", ROOT / "config"), load_yaml("countries", ROOT / "config"), 42
    )
    with duckdb.connect(str(db), read_only=True) as con:
        split = split_boundaries(con, SplitFractions(train=0.6, validation=0.2, test=0.2))
    case = SanctionsCase(
        id="T-1", sdn_ent_num=41896, variant="legal_form", expected="TRUE_MATCH",
        kind="company", role="subject", name="Sumilab SA de CV", country="Mexico",
        country_iso2="MX",
    )  # fmt: skip
    first = inject_sanctions(db, [case], split, seed=42)
    second = inject_sanctions(db, [case], split, seed=42)
    assert first == second
    with duckdb.connect(str(db), read_only=True) as con:
        row = con.execute(
            "SELECT synthetic_name, country, is_injected, injection_case_id FROM entities "
            "WHERE entity_id = ?",
            [first["T-1"]],
        ).fetchone()
        n_injected = con.execute("SELECT count(*) FROM injected_sanctions").fetchone()
    assert row == ("Sumilab SA de CV", "Mexico", True, "T-1")
    assert n_injected == (1,)


def test_injection_fails_loudly_without_candidates(tmp_path: Path) -> None:
    db = tmp_path / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(FIXTURES), db)
    build_entities(
        db, load_yaml("identities", ROOT / "config"), load_yaml("countries", ROOT / "config"), 42
    )
    with duckdb.connect(str(db), read_only=True) as con:
        split = split_boundaries(con, SplitFractions(train=0.6, validation=0.2, test=0.2))
    case = SanctionsCase(
        id="T-2", sdn_ent_num=1, variant="alias", expected="TRUE_MATCH", kind="person",
        role="counterparty", name="A B", country="Iran", country_iso2="IR", dob="1960-01-01",
    )  # fmt: skip
    with pytest.raises(ValueError, match="no candidate"):
        inject_sanctions(db, [case], split, seed=42)
