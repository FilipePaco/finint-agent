"""End-to-end ``finint data build`` on a synthetic AMLworld-shaped dataset (spec 01)."""

from __future__ import annotations

import duckdb
import pytest

from finint.config import Settings
from finint.data.build import BuildReport, build
from synthetic_amlworld import write_dataset


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, BuildReport]:
    root = tmp_path_factory.mktemp("e2e")
    settings = Settings(data_dir=root / "data", reports_dir=root / "reports")
    write_dataset(settings.raw_dir)
    report = build(settings, offline=True, log=lambda _msg: None)
    return settings, report


def test_build_runs_every_step(built: tuple[Settings, BuildReport]) -> None:
    _, report = built
    assert report.ingest.n_pattern_unmatched == 0
    assert report.ingest.n_laundering_without_attempt == 15
    assert report.normalize.n_missing_usd == 0
    assert report.sanctions.source == "snapshot"
    assert len(report.injected_sanctions) == 24
    assert len(report.injected_media) == 6
    assert len(report.injected_prompts) == 12


def test_injected_entities_are_distinct(built: tuple[Settings, BuildReport]) -> None:
    _, report = built
    entities = list(report.injected_sanctions.values()) + list(report.injected_media.values())
    assert len(entities) == len(set(entities))


def test_quality_report_is_written(built: tuple[Settings, BuildReport]) -> None:
    settings, _ = built
    text = (settings.reports_dir / "data_quality.md").read_text(encoding="utf-8")
    for heading in ("## Overview", "## Laundering by typology", "## Injected test cases"):
        assert heading in text
    assert "UNLISTED" in text


def test_every_table_has_rows(built: tuple[Settings, BuildReport]) -> None:
    settings, _ = built
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        empty = [t for t in tables if t != "pattern_unmatched"
                 and con.execute(f'SELECT count(*) FROM "{t}"').fetchone() == (0,)]  # noqa: S608  # fmt: skip
    assert not empty
