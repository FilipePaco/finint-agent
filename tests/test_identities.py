"""Synthetic identities (task 1.5 · R1.3)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest

from finint.config import load_yaml
from finint.data.identities import IdentityFactory, build_entities
from finint.data.ingest import ingest
from finint.data.sources import AmlWorldFiles

FIXTURES = Path(__file__).parent / "fixtures" / "amlworld"
CONFIG = Path(__file__).parents[1] / "config"
LATIN = re.compile(r"^[A-Za-z0-9À-ɏ .,'&\-()/]+$")


@pytest.fixture(scope="module")
def cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_yaml("identities", CONFIG), load_yaml("countries", CONFIG)


@pytest.fixture
def factory(cfgs: tuple[dict[str, Any], dict[str, Any]]) -> IdentityFactory:
    return IdentityFactory(*cfgs, seed=42)


def test_same_entity_same_identity(cfgs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    a = IdentityFactory(*cfgs, seed=42).build("80062E240", "Corporation", "Germany", 2)
    b = IdentityFactory(*cfgs, seed=42).build("80062E240", "Corporation", "Germany", 2)
    assert a == b


def test_seed_changes_identity(cfgs: tuple[dict[str, Any], dict[str, Any]]) -> None:
    a = IdentityFactory(*cfgs, seed=42).build("80062E240", "Individual", "Germany", 1)
    b = IdentityFactory(*cfgs, seed=7).build("80062E240", "Individual", "Germany", 1)
    assert a.synthetic_name != b.synthetic_name


@pytest.mark.parametrize("country", sorted(load_yaml("countries", CONFIG)["countries"]))
@pytest.mark.parametrize("entity_type", ["Individual", "Corporation", "Partnership"])
def test_every_country_yields_latin_script_names(
    factory: IdentityFactory, country: str, entity_type: str
) -> None:
    for entity_id in ("800000001", "8ABCDEF02", "80F00BA03"):
        identity = factory.build(entity_id, entity_type, country, 1)
        assert LATIN.match(identity.synthetic_name), (country, identity.synthetic_name)


def test_person_types_get_plausible_dob(factory: IdentityFactory) -> None:
    person = factory.build("800000001", "Sole Proprietorship", "Brazil", 1)
    company = factory.build("800000002", "Corporation", "Brazil", 1)
    assert person.date_of_birth is not None
    age = (date(2022, 9, 1) - person.date_of_birth).days / 365
    assert 21 < age < 81
    assert company.date_of_birth is None


def test_missing_bank_country_falls_back_to_default(factory: IdentityFactory) -> None:
    assert factory.build("800000003", "Individual", None, 1).country == "United States"


def test_build_entities_on_fixture(
    tmp_path: Path, cfgs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    path = tmp_path / "finint.duckdb"
    ingest(AmlWorldFiles.in_dir(FIXTURES), path)
    assert build_entities(path, *cfgs, seed=42) == 9
    with duckdb.connect(str(path), read_only=True) as con:
        rows = con.execute(
            "SELECT entity_id, entity_type, n_accounts, is_injected FROM entities ORDER BY 1"
        ).fetchall()
        multi_bank = con.execute(
            "SELECT n_accounts FROM entities WHERE entity_id = '8139F54E0'"
        ).fetchone()
    assert len(rows) == 9
    assert {r[1] for r in rows} == {"Sole Proprietorship", "Partnership", "Corporation"}
    assert all(r[3] is False for r in rows)
    assert multi_bank == (2,)
