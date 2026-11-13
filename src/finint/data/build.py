"""Run the data-foundation steps in order (``make data`` · spec 01)."""

from __future__ import annotations

from collections.abc import Callable
from typing import BinaryIO

import duckdb
from pydantic import BaseModel

from finint.config import Settings, load_yaml
from finint.data.classification import DataClassification
from finint.data.download import https_open
from finint.data.identities import build_entities
from finint.data.ingest import IngestReport, ingest
from finint.data.inject import (
    inject_media,
    inject_sanctions,
    load_media_cases,
    load_sanctions_cases,
)
from finint.data.memos import add_memos, inject_prompts, load_injection_cases
from finint.data.normalize import NormalizeReport, normalize
from finint.data.quality import write_quality_report
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
    n_memos: int
    injected_media: dict[str, str]
    injected_prompts: dict[str, int]


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

    log("memos: synthetic payment references")
    n_memos = add_memos(db, load_yaml("memos", settings.config_dir)["templates"])

    log("inject: adverse-media test entities and prompt-injection memos")
    media = inject_media(
        db,
        load_media_cases(settings.injections_dir / "media_cases.yaml"),
        split,
        settings.seed,
        exclude=set(injected.values()),
    )
    prompts = inject_prompts(
        db,
        load_injection_cases(settings.injections_dir / "prompt_injections.yaml"),
        split,
        settings.seed,
    )

    log("classify: every column must be in config/data_classification.yaml")
    DataClassification(load_yaml("data_classification", settings.config_dir)).enforce(db)

    report_path = write_quality_report(db, split, variant, settings.reports_dir / "data_quality.md")
    log(f"report: {report_path}")

    return BuildReport(
        ingest=ingested,
        normalize=normalized,
        n_entities=n_entities,
        split=split,
        sanctions=sanctions,
        injected_sanctions=injected,
        n_memos=n_memos,
        injected_media=media,
        injected_prompts=prompts,
    )
