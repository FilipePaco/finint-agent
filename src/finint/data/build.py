"""Run the data-foundation steps in order (``make data`` · spec 01)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from finint.config import Settings, load_yaml
from finint.data.identities import build_entities
from finint.data.ingest import IngestReport, ingest
from finint.data.normalize import NormalizeReport, normalize
from finint.data.sources import AmlWorldFiles


class BuildReport(BaseModel):
    ingest: IngestReport
    normalize: NormalizeReport
    n_entities: int


def build(
    settings: Settings, variant: str = "HI-Small", log: Callable[[str], None] = print
) -> BuildReport:
    files = AmlWorldFiles.in_dir(settings.raw_dir, variant)
    log("ingest: loading raw files")
    ingested = ingest(files, settings.duckdb_path)
    log("normalize: FX, flags, shifted timestamps")
    fx = load_yaml("fx", settings.config_dir)["usd_rates"]
    normalized = normalize(settings.duckdb_path, settings.as_of_date, fx)
    log("identities: synthetic names, countries, dates of birth")
    n_entities = build_entities(
        settings.duckdb_path,
        load_yaml("identities", settings.config_dir),
        load_yaml("countries", settings.config_dir),
        settings.seed,
    )
    return BuildReport(ingest=ingested, normalize=normalized, n_entities=n_entities)
