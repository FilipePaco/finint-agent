"""Deterministic sanctions screening (task 3.4 · R3.4 · ADR-03.4).

The LLM never decides a sanctions match. This module returns, per candidate: the SDN entry,
the matched name or alias, each algorithm's score, the combined score, which secondary
fields (date of birth, country) agree or disagree, and a ``match_status`` set by
thresholds in ``config/screening.yaml``:

* ``MATCH``: name score >= match threshold and no secondary field contradicts, or a name
  >= the potential threshold **corroborated** by an exact date of birth and the country
  (e.g. a missing middle name);
* ``POTENTIAL_MATCH``: a strong name with one contradicting field, or a name between the
  potential and match thresholds; an analyst must resolve it;
* ``NO_MATCH``: weak name, or a strong name contradicted by both DOB and country (a
  different person with the same name).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode

from finint.core.models import MatchStatus

LEGAL_FORMS = {
    "llc",
    "ltd",
    "limited",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "sa",
    "de",
    "cv",
    "gmbh",
    "ag",
    "plc",
    "llp",
    "sirketi",
    "sti",
    "ooo",
    "kk",
    "spa",
    "srl",
    "bv",
    "nv",
    "liability",
    "the",
    "and",
}
HONORIFICS = {"mr", "mrs", "ms", "dr", "sheikh", "haji", "al", "el", "bin", "ibn"}
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_FULL_DOB = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s+(\d{4})\b")
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}
COUNTRY_ALIASES = {"uk": "united kingdom", "korea, north": "north korea", "usa": "united states"}

EntityKind = Literal["person", "company"]


def normalize(name: str, kind: EntityKind) -> str:
    """Latin script, casefolded, punctuation removed, legal forms (companies) dropped.

    Honorific particles (al-, bin...) are dropped for matching only; the matched string is
    always reported in its original form.
    """
    # Drop dots first so abbreviations stay one token ("S.A." -> "sa", "C.V." -> "cv").
    text = unidecode(name).casefold().replace(".", "").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    drop = LEGAL_FORMS if kind == "company" else HONORIFICS
    tokens = [t for t in text.split() if t not in drop]
    return " ".join(tokens) or text.strip()


def _norm_country(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.casefold().strip()
    return COUNTRY_ALIASES.get(lowered, lowered)


@dataclass(frozen=True)
class SdnRecord:
    ent_num: int
    name: str
    sdn_type: str  # individual | entity | vessel | aircraft
    programs: str | None
    dob_text: str | None
    nationality: str | None
    countries: tuple[str, ...]  # from addresses
    aliases: tuple[str, ...]


class Candidate(BaseModel):
    ent_num: int
    sdn_name: str
    matched_name: str
    matched_on: Literal["primary", "alias"]
    programs: str | None
    scores: dict[str, float]
    combined_score: float
    dob_check: Literal["match", "year_match", "mismatch", "unknown"]
    country_check: Literal["match", "mismatch", "unknown"]
    matched_fields: list[str]
    mismatched_fields: list[str]
    match_status: MatchStatus


class ScreeningResult(BaseModel):
    query_name: str
    query_kind: EntityKind
    query_country: str | None
    query_dob: date | None
    status: MatchStatus  # worst case across candidates
    candidates: list[Candidate]


def _dob_check(query: date | None, dob_text: str | None) -> str:
    if query is None or not dob_text:
        return "unknown"
    if m := _FULL_DOB.search(dob_text):
        month = _MONTHS.get(m.group(2).casefold()[:3])
        if month:
            listed = date(int(m.group(3)), month, int(m.group(1)))
            if listed == query:
                return "match"
            return "year_match" if listed.year == query.year else "mismatch"
    years = {int(y) for y in _YEAR.findall(dob_text)}
    if not years:
        return "unknown"
    return "year_match" if query.year in years else "mismatch"


def _country_check(query: str | None, record: SdnRecord) -> str:
    q = _norm_country(query)
    listed = {
        c for c in (_norm_country(record.nationality), *map(_norm_country, record.countries)) if c
    }
    if q is None or not listed:
        return "unknown"
    return "match" if q in listed else "mismatch"


class SanctionsScreener:
    def __init__(self, records: Sequence[SdnRecord], config: dict[str, Any]) -> None:
        self._cfg = config
        self._records = {r.ent_num: r for r in records}
        self._names: list[str] = []
        self._owners: list[tuple[int, str, str]] = []  # (ent_num, original, primary|alias)
        for record in records:
            if record.sdn_type in ("vessel", "aircraft"):
                continue
            kind: EntityKind = "person" if record.sdn_type == "individual" else "company"
            for original, origin in (
                (record.name, "primary"),
                *((a, "alias") for a in record.aliases),
            ):
                self._names.append(normalize(_reorder(original), kind))
                self._owners.append((record.ent_num, original, origin))

    def screen(
        self, name: str, kind: EntityKind, country: str | None = None, dob: date | None = None
    ) -> ScreeningResult:
        query = normalize(name, kind)
        hits = process.extract(
            query, self._names, scorer=fuzz.token_set_ratio, limit=self._cfg["candidates"] * 3
        )
        best: dict[int, Candidate] = {}
        for matched, _, idx in hits:
            ent_num, original, origin = self._owners[idx]
            record = self._records[ent_num]
            if (record.sdn_type == "individual") != (kind == "person"):
                continue
            candidate = self._score(query, matched, original, origin, record, country, dob)
            if ent_num not in best or candidate.combined_score > best[ent_num].combined_score:
                best[ent_num] = candidate
        ranked = sorted(best.values(), key=lambda c: -c.combined_score)[: self._cfg["candidates"]]
        relevant = [c for c in ranked if c.combined_score >= self._cfg["name_score"]["potential"]]
        order = {"MATCH": 2, "POTENTIAL_MATCH": 1, "NO_MATCH": 0}
        status: MatchStatus = max(
            (c.match_status for c in relevant), key=order.__getitem__, default="NO_MATCH"
        )
        return ScreeningResult(
            query_name=name,
            query_kind=kind,
            query_country=country,
            query_dob=dob,
            status=status,
            candidates=relevant,
        )

    def _score(
        self,
        query: str,
        matched: str,
        original: str,
        origin: str,
        record: SdnRecord,
        country: str | None,
        dob: date | None,
    ) -> Candidate:
        weights = self._cfg["weights"]
        q_sorted = " ".join(sorted(query.split()))
        m_sorted = " ".join(sorted(matched.split()))
        scores = {
            "token_sort": float(fuzz.token_sort_ratio(query, matched)),
            "token_set": float(fuzz.token_set_ratio(query, matched)),
            "jaro_winkler": round(100 * JaroWinkler.similarity(q_sorted, m_sorted), 2),
        }
        combined = round(sum(scores[k] * float(w) for k, w in weights.items()), 2)
        dob_c = _dob_check(dob, record.dob_text) if record.sdn_type == "individual" else "unknown"
        country_c = _country_check(country, record)
        matched_fields = ["name"] + [
            f for f, c in (("dob", dob_c), ("country", country_c)) if c in ("match", "year_match")
        ]
        mismatched = [f for f, c in (("dob", dob_c), ("country", country_c)) if c == "mismatch"]
        thresholds = self._cfg["name_score"]
        corroborated = dob_c == "match" and country_c == "match"
        if (combined >= thresholds["match"] and not mismatched) or (
            combined >= thresholds["potential"] and corroborated
        ):
            status: MatchStatus = "MATCH"
        elif combined >= thresholds["potential"] and len(mismatched) < 2:
            # a strong name with one contradicting field, or a borderline name
            status = "POTENTIAL_MATCH"
        else:
            status = "NO_MATCH"
        return Candidate(
            ent_num=record.ent_num,
            sdn_name=record.name,
            matched_name=original,
            matched_on="primary" if origin == "primary" else "alias",
            programs=record.programs,
            scores=scores,
            combined_score=combined,
            dob_check=dob_c,
            country_check=country_c,
            matched_fields=matched_fields,
            mismatched_fields=mismatched,
            match_status=status,
        )


def _reorder(sdn_name: str) -> str:
    """OFAC writes individuals as 'SURNAME, Given'. Token-based scores ignore order anyway."""
    if "," in sdn_name:
        last, _, first = sdn_name.partition(",")
        return f"{first.strip()} {last.strip()}"
    return sdn_name


# --- Counterparty selection for harness screening (R3.4 · ADR-03.10) -------------------------


class CounterpartyRule(BaseModel):
    """Which counterparties of an alerted account the harness screens (config/screening.yaml)."""

    alert_day_max: int = 20  # counterparties of the alert date's transactions, by USD volume
    lookback_days: int = 7  # window ending on the alert date for the volume ranking
    top_n: int = 10  # top counterparties by USD volume in that window


class SelectedCounterparty(BaseModel):
    account_key: str
    reasons: list[str]  # "alert_day" and/or "top_volume_window"


def select_counterparties(
    subject: str,
    alert_day: Sequence[tuple[str, float]],
    window: Sequence[tuple[str, float]],
    rule: CounterpartyRule,
) -> list[SelectedCounterparty]:
    """Deterministic, explainable choice of the counterparties to screen.

    ``alert_day`` and ``window`` are ``(counterparty account, USD volume)`` pairs. Each list is
    ranked by volume (ties by account key) and cut to its cap; the union is returned in that
    order, never including the alerted account itself.
    """

    def top(rows: Sequence[tuple[str, float]], n: int) -> list[str]:
        ranked = sorted(
            ((k, float(v or 0)) for k, v in rows if k != subject), key=lambda r: (-r[1], r[0])
        )
        return [k for k, _ in ranked[: max(0, n)]]

    chosen: dict[str, list[str]] = {}
    for key in top(alert_day, rule.alert_day_max):
        chosen.setdefault(key, []).append("alert_day")
    for key in top(window, rule.top_n):
        chosen.setdefault(key, []).append("top_volume_window")
    return [SelectedCounterparty(account_key=k, reasons=r) for k, r in chosen.items()]
