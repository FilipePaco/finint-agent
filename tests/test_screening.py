"""Sanctions screening against the injected cases and the pinned OFAC snapshot (task 3.4 · R3.4)."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path

import pytest

from finint.config import load_yaml
from finint.core.screening import SanctionsScreener, SdnRecord, normalize
from finint.data.inject import load_sanctions_cases
from finint.data.sanctions import _read_rows

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "data" / "fixtures" / "sdn_snapshot"


def _records() -> list[SdnRecord]:
    aliases: dict[int, list[str]] = defaultdict(list)
    for r in _read_rows(SNAPSHOT / "ALT.CSV.gz"):
        aliases[int(str(r[0]))].append(str(r[3]))
    countries: dict[int, list[str]] = defaultdict(list)
    for r in _read_rows(SNAPSHOT / "ADD.CSV.gz"):
        if r[4]:
            countries[int(str(r[0]))].append(str(r[4]))
    out = []
    for r in _read_rows(SNAPSHOT / "SDN.CSV.gz"):
        ent = int(str(r[0]))
        remarks = r[11] or ""
        dob = remarks.split("DOB ", 1)[1].split(";")[0] if "DOB " in remarks else None
        nat = (
            remarks.split("nationality ", 1)[1].split(";")[0].rstrip(".")
            if "nationality " in remarks
            else None
        )
        sdn_type = (r[2] or "entity").lower()
        out.append(
            SdnRecord(
                ent,
                str(r[1]),
                sdn_type if sdn_type in ("individual", "vessel", "aircraft") else "entity",
                r[3],
                dob,
                nat,
                tuple(countries[ent]),
                tuple(aliases[ent]),
            )
        )
    return out


@pytest.fixture(scope="module")
def screener() -> SanctionsScreener:
    return SanctionsScreener(_records(), load_yaml("screening", ROOT / "config"))


CASES = load_sanctions_cases(ROOT / "data" / "injections" / "sanctions_cases.yaml")


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_injected_case_outcome(screener: SanctionsScreener, case) -> None:  # type: ignore[no-untyped-def]
    result = screener.screen(case.name, case.kind, case.country, case.dob)
    target = [c for c in result.candidates if c.ent_num == case.sdn_ent_num]
    if case.expected == "TRUE_MATCH":
        assert target, f"{case.id}: SDN entry {case.sdn_ent_num} not among candidates"
        assert target[0].match_status in ("MATCH", "POTENTIAL_MATCH"), target[0]
    else:
        assert all(c.match_status != "MATCH" for c in target), target


def test_true_matches_mostly_reach_match(screener: SanctionsScreener) -> None:
    statuses = []
    for case in (c for c in CASES if c.expected == "TRUE_MATCH"):
        result = screener.screen(case.name, case.kind, case.country, case.dob)
        hit = [c for c in result.candidates if c.ent_num == case.sdn_ent_num]
        statuses.append(hit[0].match_status if hit else "MISSED")
    assert statuses.count("MATCH") / len(statuses) >= 0.8, statuses


def test_candidate_reports_scores_and_fields(screener: SanctionsScreener) -> None:
    result = screener.screen("Mohamed Hamcho", "person", "Syria", date(1966, 5, 20))
    top = result.candidates[0]
    assert top.ent_num == 12845
    assert top.matched_on == "alias"
    assert {"token_sort", "token_set", "jaro_winkler"} <= set(top.scores)
    assert {"name", "dob", "country"} <= set(top.matched_fields)
    assert result.status == "MATCH"


def test_same_name_different_person_is_not_a_match(screener: SanctionsScreener) -> None:
    result = screener.screen("Mohammad Mokhber", "person", "UK", date(1991, 3, 14))
    target = [c for c in result.candidates if c.ent_num == 30879]
    assert target
    assert target[0].mismatched_fields == ["dob", "country"]
    assert target[0].match_status == "NO_MATCH"


def test_ordinary_names_do_not_hit(screener: SanctionsScreener) -> None:
    for name in ("Cynthia Webb", "Ferguson & Garcia Partners"):
        kind = "company" if "Partners" in name else "person"
        assert screener.screen(name, kind, "United States").status != "MATCH"


def test_normalize_strips_diacritics_legal_forms_and_particles() -> None:
    assert (
        normalize("Arcángel de Jesús Henao-Montoya", "person") == "arcangel de jesus henao montoya"
    )
    assert normalize("SUMILAB, S.A. DE C.V.", "company") == "sumilab"
    assert normalize("AL-ASSAD, Bushra", "person") == "assad bushra"


def test_counterparty_selection_is_ranked_capped_and_explained() -> None:
    """R3.4 / ADR-03.10: the harness picks counterparties by a code rule, never the LLM."""
    from finint.core.screening import CounterpartyRule, select_counterparties

    day = [("1:B", 50.0), ("1:A", 50.0), ("1:SELF", 999.0), ("1:C", 10.0)]
    window = [("1:Z", 5000.0), ("1:A", 400.0), ("1:Y", 1.0)]
    rule = CounterpartyRule(alert_day_max=2, lookback_days=7, top_n=2)
    picked = select_counterparties("1:SELF", day, window, rule)
    # alert-day first (volume desc, ties by key), then the window's top volume; no self.
    assert [(p.account_key, p.reasons) for p in picked] == [
        ("1:A", ["alert_day", "top_volume_window"]),
        ("1:B", ["alert_day"]),
        ("1:Z", ["top_volume_window"]),
    ]


def test_counterparty_selection_defaults_match_config() -> None:
    """R3.4: the caps live in config/screening.yaml and parse into the rule."""
    from finint.core.screening import CounterpartyRule

    rule = CounterpartyRule.model_validate(
        load_yaml("screening", ROOT / "config")["counterparties"]
    )
    assert rule.alert_day_max >= 1
    assert rule.top_n >= 1
    assert CounterpartyRule().model_dump() == rule.model_dump()
