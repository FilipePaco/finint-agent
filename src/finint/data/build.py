"""Run the data-foundation steps in order (``make data`` · spec 01)."""

from __future__ import annotations

from collections.abc import Callable
from typing import BinaryIO

import duckdb
from pydantic import BaseModel

from finint.config import Settings, load_yaml
from finint.data.download import https_open
from finint.data.identities import build_entities
from finint.data.ingest import IngestReport, ingest
from finint.data.inject import inject_sanctions, load_sanctions_cases
from finint.data.normalize import NormalizeReport, normalize
from finint.data.sanctions import SanctionsReport, fetch_sdn, load_sdn
from finint.data.sources import AmlWorldFiles
from finint.data.split import SplitBoundaries, SplitFractions, split_boundaries


class BuildReport(BaseModel):
    ingest: IngestReport
    normalize: NormalizeReport
    n_entities: int
    split: SplitBoundaries
    sanctions: SanctionsReport
    injected_sanctions: dict[str, str]


def _offline_opener(url: str) -> BinaryIO:
    raise OSError(f"offline build: not fetching {url}")


def build(
    settings: Settings,
    variant: str = "HI-Small",
    *,
    offline: bool = False,
    log: Callable[[str], None] = print,
) -> BuildReport:
    files = AmlWorldFiles.in_dir(settings.raw_dir, variant)
    db = settings.duckdb_path

    log("ingest: loading raw files")
    ingested = ingest(files, db)

    log("normalize: FX, flags, shifted timestamps")
    normalized = normalize(
        db, settings.as_of_date, load_yaml("fx", settings.config_dir)["usd_rates"]
    )

    log("identities: synthetic names, countries, dates of birth")
    n_entities = build_entities(
        db,
        load_yaml("identities", settings.config_dir),
        load_yaml("countries", settings.config_dir),
        settings.seed,
    )

    fractions = SplitFractions.model_validate(load_yaml("ml", settings.config_dir)["split"])
    with duckdb.connect(str(db), read_only=True) as con:
        split = split_boundaries(con, fractions)

    log("sanctions: OFAC SDN list" + (" (offline snapshot)" if offline else ""))
    fetched = fetch_sdn(
        settings.raw_dir / "ofac",
        settings.fixtures_dir / "sdn_snapshot",
        _offline_opener if offline else https_open,
    )
    sanctions = load_sdn(db, fetched)
    if sanctions.fallback_reason:
        log(f"sanctions: download failed, using pinned snapshot ({sanctions.fallback_reason})")

    log("inject: sanctions near-match cases")
    cases = load_sanctions_cases(settings.injections_dir / "sanctions_cases.yaml")
    injected = inject_sanctions(db, cases, split, settings.seed)

    return BuildReport(
        ingest=ingested,
        normalize=normalized,
        n_entities=n_entities,
        split=split,
        sanctions=sanctions,
        injected_sanctions=injected,
    )
