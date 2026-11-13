"""Command-line entry point (``finint``).

Commands are added as their specs are implemented. ``make`` targets wrap them.
"""

from __future__ import annotations

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


@data_app.command("download")
def data_download(
    variant: Annotated[str, typer.Option(help="AMLworld variant prefix.")] = "HI-Small",
    force: Annotated[bool, typer.Option(help="Re-download files that already exist.")] = False,
) -> None:
    """Download the AMLworld files into data/raw (skips files already present)."""
    from finint.data.download import download
    from finint.data.sources import AmlWorldFiles

    fetched = download(AmlWorldFiles.in_dir(get_settings().raw_dir, variant), force=force)
    typer.echo("\n".join(f"downloaded {path}" for path in fetched) or "all files present")


@data_app.command("build")
def data_build(
    variant: Annotated[str, typer.Option(help="AMLworld variant prefix.")] = "HI-Small",
) -> None:
    """Rebuild data/finint.duckdb from the raw files (all spec-01 steps, in order)."""
    from finint.data.build import build
    from finint.data.sources import MissingRawDataError

    try:
        report = build(get_settings(), variant, log=typer.echo)
    except MissingRawDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(report.model_dump_json(indent=2))
