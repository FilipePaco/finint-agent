"""Command-line entry point (``finint``).

Commands are added as their specs are implemented. ``make`` targets wrap them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from finint import __version__
from finint.config import get_settings

app = typer.Typer(no_args_is_help=True, help="finint-agent: AML alert detection and investigation.")
data_app = typer.Typer(no_args_is_help=True, help="Build the data foundation (spec 01).")
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Run the finint-agent command-line application."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def info() -> None:
    """Show resolved paths and which integrations are configured (never prints secrets)."""
    settings = get_settings()
    rows = {
        "data_dir": settings.data_dir,
        "duckdb": settings.duckdb_path,
        "audit_db": settings.audit_db_path,
        "as_of_date": settings.as_of_date,
        "investigator_model": f"{settings.investigator.provider}:{settings.investigator.model}",
        "reviewer_model": f"{settings.reviewer.provider}:{settings.reviewer.model}",
        "media_model": f"{settings.media.provider}:{settings.media.model}",
        "google_api_key": "set" if settings.google_api_key else "missing",
        "tavily_api_key": "set" if settings.tavily_api_key else "missing",
        "langfuse": "enabled" if settings.langfuse_enabled else "disabled",
    }
    width = max(len(key) for key in rows)
    for key, value in rows.items():
        typer.echo(f"{key.ljust(width)}  {value}")


@data_app.command("ingest")
def data_ingest(
    raw_dir: Annotated[Path | None, typer.Option(help="Folder with the AMLworld files.")] = None,
    variant: Annotated[str, typer.Option(help="AMLworld variant prefix.")] = "HI-Small",
) -> None:
    """Load AMLworld transactions, accounts and laundering patterns into DuckDB."""
    from finint.data.ingest import ingest
    from finint.data.sources import AmlWorldFiles, MissingRawDataError

    settings = get_settings()
    files = AmlWorldFiles.in_dir(raw_dir or settings.raw_dir, variant)
    try:
        report = ingest(files, settings.duckdb_path)
    except MissingRawDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    for key, value in report.model_dump().items():
        typer.echo(f"{key}: {value}")
