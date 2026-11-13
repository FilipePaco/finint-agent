"""Synthetic identities for AMLworld entities (task 1.5 · R1.3).

Each entity gets a fictional Latin-script name, a country (the most frequent bank country
across its accounts) and, for person-type entities, a date of birth. Generation is
deterministic: every entity seeds Faker with ``crc32(entity_id) ^ seed``. Names are never
derived from, or linked to, real persons.
"""

from __future__ import annotations

import csv
import tempfile
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
from faker import Faker
from unidecode import unidecode

_ENTITY_BASE = """
WITH acct AS (
    SELECT
        entity_id,
        regexp_extract(entity_label, '^(.*) #', 1) AS entity_type,
        regexp_extract(bank_name, '^(.*) Bank #', 1) AS bank_country
    FROM accounts
),
ranked AS (
    SELECT
        entity_id,
        bank_country,
        row_number() OVER (
            PARTITION BY entity_id ORDER BY count(*) DESC, bank_country
        ) AS rank
    FROM acct
    WHERE NOT list_contains($no_country, bank_country)
    GROUP BY entity_id, bank_country
)
SELECT
    a.entity_id,
    min(a.entity_type) AS entity_type,
    min(r.bank_country) AS bank_country,
    count(*) AS n_accounts
FROM acct a
LEFT JOIN ranked r ON r.entity_id = a.entity_id AND r.rank = 1
GROUP BY a.entity_id
ORDER BY a.entity_id
"""


@dataclass(frozen=True, slots=True)
class Identity:
    entity_id: str
    entity_type: str
    country: str
    country_iso2: str
    synthetic_name: str
    date_of_birth: date | None
    n_accounts: int


class IdentityFactory:
    """Builds one deterministic identity per entity from config/identities.yaml."""

    def __init__(
        self, identity_cfg: dict[str, Any], country_cfg: dict[str, Any], seed: int
    ) -> None:
        self._cfg = identity_cfg
        self._countries: dict[str, dict[str, str]] = country_cfg["countries"]
        self._default_country: str = country_cfg["default"]
        self._seed = seed
        self._fakers: dict[str, Faker] = {}
        romanize = identity_cfg["romanize"]
        self._faker_romanized = set(romanize["faker_romanized"])
        self._transliterate = set(romanize["transliterate"])
        self._curated = set(romanize["curated"])
        self._person_types = set(identity_cfg["person_entity_types"])
        self._age_min, self._age_max = identity_cfg["age_range_years"]
        self._reference = date.fromisoformat(str(identity_cfg["reference_date"]))

    def build(
        self, entity_id: str, entity_type: str, bank_country: str | None, n_accounts: int
    ) -> Identity:
        country = bank_country or self._default_country
        if country not in self._countries:
            raise KeyError(f"Country {country!r} missing from config/countries.yaml")
        locale = self._countries[country]["locale"]
        fake = self._faker(locale)
        fake.seed_instance(zlib.crc32(entity_id.encode()) ^ self._seed)
        is_person = entity_type in self._person_types
        if is_person:
            name = self._person_name(fake, locale)
        elif entity_type == "Country":
            name = f"Government of {country} Treasury"
        elif entity_type == "Direct":
            name = f"Direct Settlement {entity_id[-5:]}"
        else:
            name = self._company_name(fake, locale, partnership=entity_type == "Partnership")
        return Identity(
            entity_id=entity_id,
            entity_type=entity_type,
            country=country,
            country_iso2=self._countries[country]["iso2"],
            synthetic_name=" ".join(name.split()),
            date_of_birth=self._dob(fake) if is_person else None,
            n_accounts=n_accounts,
        )

    def _faker(self, locale: str) -> Faker:
        if locale not in self._fakers:
            self._fakers[locale] = Faker(locale)
        return self._fakers[locale]

    def _first_last(self, fake: Faker, locale: str) -> tuple[str, str]:
        if locale in self._curated:
            names = self._cfg["curated_names"][locale]
            return fake.random_element(names["first"]), fake.random_element(names["last"])
        if locale in self._faker_romanized:
            return fake.first_romanized_name(), fake.last_romanized_name()
        first, last = fake.first_name(), fake.last_name()
        if locale in self._transliterate:
            return unidecode(first), unidecode(last)
        return first, last

    def _person_name(self, fake: Faker, locale: str) -> str:
        first, last = self._first_last(fake, locale)
        return f"{first} {last}"

    def _company_name(self, fake: Faker, locale: str, *, partnership: bool) -> str:
        suffix = self._cfg["company_suffix"].get(locale)
        if partnership:
            _, last_a = self._first_last(fake, locale)
            _, last_b = self._first_last(fake, locale)
            return f"{last_a} & {last_b} Partners"
        if suffix is not None:
            _, last = self._first_last(fake, locale)
            return f"{last} {suffix}"
        company: str = fake.company()
        return company

    def _dob(self, fake: Faker) -> date:
        days = fake.random_int(self._age_min * 365, self._age_max * 365)
        return self._reference - timedelta(days=days)


def entity_rows(
    con: duckdb.DuckDBPyConnection, no_country: list[str]
) -> Iterator[tuple[str, str, str | None, int]]:
    cursor = con.execute(_ENTITY_BASE, {"no_country": no_country})
    while batch := cursor.fetchmany(10_000):
        for entity_id, entity_type, bank_country, n_accounts in batch:
            yield entity_id, entity_type, bank_country or None, int(n_accounts)


def build_entities(
    db_path: Path, identity_cfg: dict[str, Any], country_cfg: dict[str, Any], seed: int
) -> int:
    """(Re)create the ``entities`` table. Returns the number of entities."""
    factory = IdentityFactory(identity_cfg, country_cfg, seed)
    with duckdb.connect(str(db_path)) as con, tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "entities.csv"
        count = 0
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in list(entity_rows(con, country_cfg["no_country_prefixes"])):
                identity = factory.build(*row)
                writer.writerow(
                    (
                        identity.entity_id, identity.entity_type, identity.country,
                        identity.country_iso2, identity.synthetic_name,
                        identity.date_of_birth.isoformat() if identity.date_of_birth else "",
                        identity.n_accounts,
                    )
                )  # fmt: skip
                count += 1
        con.execute(
            """
            CREATE OR REPLACE TABLE entities AS
            SELECT *, false AS is_injected, NULL::VARCHAR AS injection_case_id
            FROM read_csv($path, header = false, columns = {
                'entity_id': 'VARCHAR', 'entity_type': 'VARCHAR', 'country': 'VARCHAR',
                'country_iso2': 'VARCHAR', 'synthetic_name': 'VARCHAR',
                'date_of_birth': 'DATE', 'n_accounts': 'INTEGER'
            })
            ORDER BY entity_id
            """,
            {"path": str(staging)},
        )
    return count
