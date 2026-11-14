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
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate the investigation agent (spec 04).")
app.add_typer(eval_app, name="eval")
report_app = typer.Typer(no_args_is_help=True, help="Operational reports (spec 04).")
app.add_typer(report_app, name="report")


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
    version = model_version(model)  # at start: the code that trains, not the code at save time
    frame = load_frame(settings.duckdb_path, _split_boundaries(), graph=model == "m1")
    trained = train_lightgbm(frame, feature_sets[model], model, settings.seed, log=typer.echo)
    metrics = evaluate(trained, frame)
    path = save_artifact(trained, metrics, settings.artifacts_dir, version)
    typer.echo(json.dumps({"artifact": str(path), **metrics}, indent=2, default=str))


def _train_gnn() -> None:
    import json

    # Never import finint.ml.train here: LightGBM's OpenMP next to PyTorch's segfaults.
    from finint.ml.artifacts import model_version
    from finint.ml.evaluation import evaluate_scores
    from finint.ml.gnn import collect_scores, load_day_batches, save_gnn_artifact, train_gnn

    settings = get_settings()
    version = model_version("m2")  # at start: the code that trains, not the code at save time
    batches = load_day_batches(settings.duckdb_path, _split_boundaries(), log=typer.echo)
    result = train_gnn(batches, settings.seed, log=typer.echo)
    metrics = evaluate_scores(collect_scores(result.model, batches, "test"))
    path = save_gnn_artifact(result, metrics, settings.artifacts_dir, version)
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


@ml_app.command("report")
def ml_report() -> None:
    """Write reports/ml_report.md and figures from the saved artifacts."""
    import json

    import lightgbm as lgb

    from finint.ml.dataset import load_frame
    from finint.ml.evaluation import evaluate_scores
    from finint.ml.registry import choose_champion, latest_artifacts
    from finint.ml.report import build_ml_report

    settings = get_settings()
    models = latest_artifacts(settings.artifacts_dir)
    stale = [
        a for a in models.values() if a.kind == "lightgbm" and "pr_curve" not in a.metrics["test"]
    ]
    if stale:  # artifacts trained before curves were stored: re-score test once
        frame = load_frame(settings.duckdb_path, _split_boundaries(), graph=True)
        test = frame[frame["split"] == "test"]
        for info in stale:
            booster = lgb.Booster(model_file=str(info.path / "model.txt"))
            scored = test[["label", "in_tail", "typology"]].copy()
            scored["score"] = booster.predict(
                test[json.loads((info.path / "features.json").read_text())]
            )
            info.metrics["test"] = evaluate_scores(scored)
            (info.path / "metrics.json").write_text(json.dumps(info.metrics, indent=2), "utf-8")
            typer.echo(f"re-scored {info.version} to store its PR curve")
        models = latest_artifacts(settings.artifacts_dir)
    champion, why = choose_champion(models)
    run = json.loads((champion.path / "alert_run.json").read_text("utf-8"))
    text = build_ml_report(models, champion, why, run, settings.reports_dir / "figures")
    out = settings.reports_dir / "ml_report.md"
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {out}")


@app.command()
def investigate(
    alert_id: Annotated[str, typer.Argument(help="Alert id, e.g. AL-20220909-0001")],
    decision: Annotated[
        str | None, typer.Option(help="Record an analyst decision now: APPROVE, EDIT or REJECT")
    ] = None,
    final: Annotated[str | None, typer.Option(help="Final disposition for EDIT/REJECT")] = None,
    reason: Annotated[str, typer.Option(help="Required for EDIT and REJECT")] = "",
    analyst: Annotated[str, typer.Option(help="Analyst id")] = "cli-analyst",
    live_media: Annotated[bool, typer.Option(help="Use Tavily instead of fixtures")] = False,
) -> None:
    """Run the multi-agent investigation for one alert and stop at the analyst decision."""
    import json

    from finint.agents.graph import build_graph, resume, sqlite_checkpointer, start
    from finint.agents.runtime import build_runtime
    from finint.core.models import AnalystDecision

    settings = get_settings()
    runtime = build_runtime(settings, live_media=live_media)
    graph = build_graph(runtime, sqlite_checkpointer(settings.data_dir / "checkpoints.sqlite"))
    run_id, payload = start(graph, alert_id)
    typer.echo(json.dumps(payload, indent=2, default=str))
    typer.echo(f"\nrun_id: {run_id} (waiting for the analyst)")
    if decision:
        recommended = (payload.get("policy") or {}).get("disposition") or "NEEDS_MORE_INFO"
        choice = AnalystDecision(
            action=decision.upper(),
            analyst_id=analyst,
            reason=reason,
            final_disposition=final or recommended,
        )
        resume(graph, run_id, choice)
        typer.echo(f"decision recorded: {choice.action} -> {choice.final_disposition}")


EVAL_SET = "eval/eval_set.jsonl"


@eval_app.command("build")
def eval_build() -> None:
    """Build eval/eval_set.jsonl (natural + forced alerts) and register forced alerts."""
    from collections import Counter

    from finint.core.audit import AuditLog
    from finint.eval.dataset import build_eval_set, write_eval_set
    from finint.ml.alerts import load_champion
    from finint.ml.registry import choose_champion, latest_artifacts

    settings = get_settings()
    info, _ = choose_champion(latest_artifacts(settings.artifacts_dir))
    cases = build_eval_set(
        settings.duckdb_path,
        load_champion(info.path),
        info.path / "calibrator.json",
        AuditLog(settings.audit_db_path),
        settings.seed,
    )
    path = write_eval_set(cases, settings.config_dir.parent / EVAL_SET)
    typer.echo(f"{len(cases)} cases -> {path}: {dict(Counter(c.source for c in cases))}")


@eval_app.command("run")
def eval_run(
    config: Annotated[
        str, typer.Option(help="main, flash-investigator, no-skills, no-reviewer")
    ] = "main",
    limit: Annotated[int | None, typer.Option(help="Run at most N pending cases")] = None,
    paired: Annotated[
        bool, typer.Option(help="Also run injection cases without the payload")
    ] = True,
    only: Annotated[str | None, typer.Option(help="Comma-separated alert ids")] = None,
) -> None:
    """Run the agent on the eval set (resumable; finished cases are skipped)."""
    from finint.agents.llm import make_model
    from finint.agents.runtime import build_runtime, llm_available
    from finint.config import ModelRole
    from finint.data.memos import load_injection_cases
    from finint.eval.dataset import load_eval_set
    from finint.eval.runner import CONFIGS, configure, results_path, run_eval

    settings = get_settings()
    if not llm_available(settings):
        typer.echo(
            "No LLM key configured (GOOGLE_API_KEY); the evaluation needs a live model.", err=True
        )
        raise typer.Exit(code=2)
    cfg = CONFIGS[config]
    root = settings.config_dir.parent
    cases = load_eval_set(root / EVAL_SET)
    if only:
        wanted = set(only.split(","))
        cases = [c for c in cases if c.alert_id in wanted]
    payloads = {
        c.id: c.payload
        for c in load_injection_cases(settings.injections_dir / "prompt_injections.yaml")
    }
    rt = configure(
        build_runtime(settings), cfg, lambda model: make_model(ModelRole(model=model), settings)
    )
    n = run_eval(
        rt,
        cases,
        cfg,
        results_path(root, cfg),
        payloads=payloads,
        paired=paired,
        limit=limit,
        log=typer.echo,
    )
    typer.echo(f"ran {n} case(s) for config {cfg.name}")


@eval_app.command("report")
def eval_report() -> None:
    """Write reports/eval_report.md from eval/results plus screener metrics on injected cases."""
    from finint.config import load_yaml
    from finint.core.audit import AuditLog
    from finint.data.inject import load_sanctions_cases
    from finint.data.memos import load_injection_cases
    from finint.eval.report import build_eval_report, load_results, refresh_echo
    from finint.tools.sanctions import _freeze, load_screener

    settings = get_settings()
    root = settings.config_dir.parent
    config = load_yaml("screening", settings.config_dir)
    screener = load_screener(
        settings.duckdb_path, tuple(sorted((k, _freeze(v)) for k, v in config.items()))
    )
    cases = load_sanctions_cases(settings.injections_dir / "sanctions_cases.yaml")
    true_found = decoys_cleared = 0
    for case in cases:
        result = screener.screen(case.name, case.kind, case.country, case.dob)
        hit = [c for c in result.candidates if c.ent_num == case.sdn_ent_num]
        status = hit[0].match_status if hit else "NO_MATCH"
        if case.expected == "TRUE_MATCH":
            true_found += status == "MATCH"
        else:
            decoys_cleared += status != "MATCH"
    n_true = sum(c.expected == "TRUE_MATCH" for c in cases)
    screening = {
        "n": len(cases),
        "n_true": n_true,
        "n_decoys": len(cases) - n_true,
        "true_matches_found": true_found,
        "decoys_cleared": decoys_cleared,
    }
    results = load_results(root / "eval" / "results")
    refresh_echo(
        results,
        AuditLog(settings.audit_db_path),
        {
            c.id: c.payload
            for c in load_injection_cases(settings.injections_dir / "prompt_injections.yaml")
        },
    )
    text = build_eval_report(results, screening)
    out = settings.reports_dir / "eval_report.md"
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {out}")


@report_app.command("weekly")
def report_weekly() -> None:
    """Write reports/ops_report_<week>.md with the five metrics and the two charts."""
    from finint.config import load_yaml
    from finint.core.audit import AuditLog
    from finint.eval.ops_report import build_ops_report

    settings = get_settings()
    week, text = build_ops_report(
        settings.duckdb_path,
        AuditLog(settings.audit_db_path),
        settings.as_of_date,
        load_yaml("ops", settings.config_dir),
        settings.reports_dir / "figures",
    )
    out = settings.reports_dir / f"ops_report_{week}.md"
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {out}")


@eval_app.command("record")
def eval_record() -> None:
    """Record live Gemini responses on the synthetic world for the CI replay (needs a key)."""
    import tempfile
    from pathlib import Path

    from finint.agents.llm import make_model
    from finint.agents.runtime import ROLES, llm_available
    from finint.data.synthetic import build_world
    from finint.eval.replay import Cassette
    from finint.eval.replay_run import CASSETTE, replay_runtime, run_alerts

    settings = get_settings()
    if not llm_available(settings):
        typer.echo("Recording needs GOOGLE_API_KEY.", err=True)
        raise typer.Exit(code=2)
    with tempfile.TemporaryDirectory() as tmp:
        world = build_world(Path(tmp) / "data", Path(tmp) / "reports")
        live = {role: make_model(getattr(settings, role), settings) for role in ROLES}
        cassette = Cassette(CASSETTE)
        cassette.entries = {}
        runs = run_alerts(replay_runtime(world["settings"], cassette, live), world["alerts"])
        cassette.save()
    for run in runs:
        typer.echo(
            f"{run['alert_id']}: {run['payload'].get('status')} "
            f"{(run['payload'].get('policy') or {}).get('disposition')}"
        )
    typer.echo(f"recorded {len(cassette.entries)} responses -> {CASSETTE}")
