"""Command-line entry point (``finint``).

Commands are added as their specs are implemented. ``make`` targets wrap them.
"""

from __future__ import annotations

from typing import Annotated

import typer

from finint import __version__
from finint.config import get_settings
from finint.data.split import SplitBoundaries

app = typer.Typer(no_args_is_help=True, help="finint-agent: AML alert detection and investigation.")
data_app = typer.Typer(no_args_is_help=True, help="Build the data foundation (spec 01).")
app.add_typer(data_app, name="data")
ml_app = typer.Typer(no_args_is_help=True, help="Train and evaluate detection models (spec 02).")
app.add_typer(ml_app, name="ml")
audit_app = typer.Typer(no_args_is_help=True, help="Inspect the audit trail.")
app.add_typer(audit_app, name="audit")


@audit_app.command("verify")
def audit_verify() -> None:
    """Recompute the hash chain; exit 1 if any event was altered."""
    from finint.core.audit import AuditIntegrityError, AuditLog

    try:
        count = AuditLog(get_settings().audit_db_path).verify()
    except AuditIntegrityError as error:
        typer.echo(f"AUDIT CHAIN BROKEN: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"audit chain intact: {count:,} events")


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
    offline: Annotated[bool, typer.Option(help="Use the pinned OFAC snapshot.")] = False,
) -> None:
    """Rebuild data/finint.duckdb from the raw files (all spec-01 steps, in order)."""
    from finint.data.build import build
    from finint.data.sources import MissingRawDataError

    try:
        report = build(get_settings(), variant, offline=offline, log=typer.echo)
    except MissingRawDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(report.model_dump_json(indent=2))


def _split_boundaries() -> SplitBoundaries:
    import duckdb

    from finint.config import load_yaml
    from finint.data.split import SplitFractions, split_boundaries

    settings = get_settings()
    fractions = SplitFractions.model_validate(load_yaml("ml", settings.config_dir)["split"])
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        return split_boundaries(con, fractions)


@ml_app.command("features")
def ml_features() -> None:
    """Build per-transaction tabular features (features_txn)."""
    from finint.ml.features import build_tabular_features

    typer.echo(f"features_txn rows: {build_tabular_features(get_settings().duckdb_path):,}")


@ml_app.command("graph")
def ml_graph(
    window_days: Annotated[int, typer.Option(help="Closed days of history per graph.")] = 3,
) -> None:
    """Build graph features on sliding daily windows (graph_nodes, features_graph)."""
    from finint.ml.graph_features import build_graph_features

    rows = build_graph_features(get_settings().duckdb_path, window_days, log=typer.echo)
    typer.echo(f"features_graph rows: {rows:,}")


@ml_app.command("train")
def ml_train(
    model: Annotated[
        str, typer.Option(help="m0 = tabular, m1 = tabular + graph, m2 = GraphSAGE (GNN).")
    ] = "m0",
) -> None:
    """Train a model, select on validation, evaluate on test, save the artifact."""
    import json

    if model == "m2":
        _train_gnn()
        return

    from finint.ml.artifacts import model_version
    from finint.ml.dataset import load_frame
    from finint.ml.features import FEATURES
    from finint.ml.graph_features import GRAPH_FEATURES
    from finint.ml.train import evaluate, save_artifact, train_lightgbm

    feature_sets = {"m0": list(FEATURES), "m1": [*FEATURES, *GRAPH_FEATURES]}
    if model not in feature_sets:
        raise typer.BadParameter(f"unknown model {model!r}; choose from {sorted(feature_sets)}")
    settings = get_settings()
    frame = load_frame(settings.duckdb_path, _split_boundaries(), graph=model == "m1")
    trained = train_lightgbm(frame, feature_sets[model], model, settings.seed, log=typer.echo)
    metrics = evaluate(trained, frame)
    path = save_artifact(trained, metrics, settings.artifacts_dir, model_version(model))
    typer.echo(json.dumps({"artifact": str(path), **metrics}, indent=2, default=str))


def _train_gnn() -> None:
    import json

    # Never import finint.ml.train here: LightGBM's OpenMP next to PyTorch's segfaults.
    from finint.ml.artifacts import model_version
    from finint.ml.evaluation import evaluate_scores
    from finint.ml.gnn import collect_scores, load_day_batches, save_gnn_artifact, train_gnn

    settings = get_settings()
    batches = load_day_batches(settings.duckdb_path, _split_boundaries(), log=typer.echo)
    result = train_gnn(batches, settings.seed, log=typer.echo)
    metrics = evaluate_scores(collect_scores(result.model, batches, "test"))
    path = save_gnn_artifact(result, metrics, settings.artifacts_dir, model_version("m2"))
    typer.echo(json.dumps({"artifact": str(path), **metrics}, indent=2, default=str))


@ml_app.command("alerts")
def ml_alerts() -> None:
    """Pick the champion, calibrate, set the capacity threshold, write alerts + audit events."""
    import json
    from dataclasses import asdict

    from finint.config import load_yaml
    from finint.core.audit import AuditLog
    from finint.ml.alerts import generate_alerts, load_champion
    from finint.ml.dataset import load_frame
    from finint.ml.model_card import render_model_card, write_model_card
    from finint.ml.registry import choose_champion, latest_artifacts

    settings = get_settings()
    info, why = choose_champion(latest_artifacts(settings.artifacts_dir))
    typer.echo(f"champion: {info.version} ({why})")
    bounds = _split_boundaries()
    frame = load_frame(settings.duckdb_path, bounds, graph=info.name != "m0")
    capacity = int(load_yaml("ml", settings.config_dir)["alerts"]["capacity_per_day"])
    run = generate_alerts(
        settings.duckdb_path,
        frame,
        bounds,
        load_champion(info.path),
        capacity,
        AuditLog(settings.audit_db_path),
    )
    payload = asdict(run)
    (info.path / "alert_run.json").write_text(json.dumps(payload, indent=2, default=str), "utf-8")
    card = render_model_card(info, why, payload, bounds.model_dump(mode="json"))
    write_model_card(info.path / "model_card.md", card)
    choice = run.threshold
    typer.echo(
        f"alerts: {run.n_alerts:,} {run.alerts_by_split} | threshold {choice.threshold:.4f} "
        f"for {choice.capacity_per_day}/day | val precision {choice.precision:.1%} | "
        f"test precision {run.test_precision:.1%}, recall {run.test_recall_transactions:.1%}"
    )
