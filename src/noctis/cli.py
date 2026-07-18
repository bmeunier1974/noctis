"""Command-line interface (Typer).

Commands: ``setup`` (the guided first-run wizard), ``init``/``migrate`` (scaffold /
legacy-layout move), ``run``, ``status``, ``report``, ``backtest``, ``champions``,
``account`` (the continuous paper account), ``research`` (one observable agent research
session), ``strategies`` (the authored library index), and the ``data`` sub-app.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime

import typer

from noctis.config import SafetyGateError, load_settings, resolve_execution_mode


def _logging_level(verbose: int) -> int:
    """One verbosity ladder shared by ``run`` and ``research`` (P3 unifies the two, which used
    to disagree — ``run`` mapped ``-v`` to INFO, ``research`` to WARNING).

    Stdlib logging stays quiet (WARNING) until ``-vv`` drops it to DEBUG. The level-1 (``-v``)
    feed — tool calls, phase banners, per-session usage — rides the observability
    :class:`~noctis.observability.Console`, not raw INFO log lines, so ``-v`` reads clean on both
    commands. ``--show-reasoning`` is orthogonal: it opens ``think``/``say`` on the Console
    without touching this level.
    """
    return logging.WARNING if verbose < 2 else logging.DEBUG


app = typer.Typer(
    add_completion=False,
    help="Noctis — an autonomous, paper-only trading agent.",
    no_args_is_help=True,
)


def _resolve_mode_or_exit(config: str | None):
    """Load settings and resolve the execution mode, exiting non-zero on a gate error."""
    settings = load_settings(config_path=config)
    try:
        mode = resolve_execution_mode(settings)
    except SafetyGateError as exc:
        typer.secho(f"SAFETY GATE: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    return settings, mode


def _resolve_session_or_exit(config: str | None, **kwargs):
    """Resolve the session inputs (the composition root's precedence chain), mapping each
    typed startup error to red text + a non-zero exit. Errors are loud at startup by design:
    a typo'd selector or a closed safety gate must never silently un-steer a multi-day run.
    """
    from noctis.bootstrap import UsageError, resolve_session
    from noctis.research import MandateError

    try:
        return resolve_session(config, **kwargs)
    except UsageError as exc:  # --directive × --mandate, or an unknown --metric
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except MandateError as exc:
        typer.secho(f"MANDATE: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except SafetyGateError as exc:
        typer.secho(f"SAFETY GATE: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _guard_legacy_or_exit(settings, *, warn_only: bool = False) -> None:
    """Refuse to run (exit 2) beside an un-migrated legacy layout; ``status`` only warns.

    A silently-empty champion board next to abandoned data is worse than a hard stop, so
    every state/lake/report-touching command refuses until ``noctis migrate`` has moved the
    legacy artifacts (or the knobs explicitly point at them).
    """
    from noctis.bootstrap import detect_legacy_layout

    found = detect_legacy_layout(settings)
    if not found:
        return
    lines = "\n".join(
        f"  {a.legacy}  (configured location {a.configured} does not exist)" for a in found
    )
    message = (
        "Legacy (pre-workspace) layout detected — running now would abandon:\n"
        f"{lines}\n"
        "Run `noctis migrate` to move them into the workspace, or point the knobs at them "
        "explicitly in config.yaml."
    )
    if warn_only:
        typer.secho(f"WARNING: {message}", fg=typer.colors.YELLOW, err=True)
        return
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


def _echo_mandate(mandate, override_lines: list[str]) -> None:
    """Print the resolved mandate provenance + any applied config overrides at session start."""
    if mandate is not None:
        typer.echo(f"Mandate: {mandate.source}")
    for line in override_lines:
        source = mandate.source if mandate is not None else "?"
        typer.echo(f"  mandate {source} overrides {line}")


def _echo_research_engine(settings) -> None:
    """Announce, up front, which research engine the loop will run — the agent (an LLM authoring
    strategies) or the legacy proposer/Optuna fallback — and the model behind it. The fallback is
    silent otherwise: it's an INFO log the ``-v`` (WARNING) ladder swallows, so an operator can run
    the whole night on the legacy loop believing the LLM they configured is driving. When the agent
    can't run, this says why and how to fix it, in yellow, by default."""
    from noctis.research import client_status

    status = client_status(settings)
    if status.ok:
        typer.echo(f"Research engine: agent loop → {status.model}")
    else:
        typer.secho(
            f"Research engine: LEGACY loop (no LLM) — configured {status.model} "
            f"unavailable: {status.reason}",
            fg=typer.colors.YELLOW,
        )


@app.command()
def run(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    time_limit_hours: float = typer.Option(
        None, "--time-limit-hours", help="Stop after this many hours (overrides config)."
    ),
    directive: str = typer.Option(
        None,
        "--directive",
        help="Inline operator mandate for this run (overrides config), e.g. 'find a strategy "
        "on very volatile stocks; high risk appetite'. Mutually exclusive with --mandate.",
    ),
    mandate: str = typer.Option(
        None,
        "--mandate",
        help="Select a mandate under mandate_dir for this run (a profile name, MANDATE, or "
        "auto), overriding config. Mutually exclusive with --directive.",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Show progress: -v streams phase banners + the research tool feed, -vv adds "
        "think/say/per-round usage and DEBUG logs.",
    ),
    show_reasoning: bool = typer.Option(
        False,
        "--show-reasoning",
        help="Surface each research session's reasoning + narration inline (think/say) even "
        "without -vv. Only providers that return chain-of-thought over the API show reasoning; "
        "narration always shows.",
    ),
) -> None:
    """Start Noctis. Loads config + memory, resolves the safety gate, runs the loop."""
    import signal

    from noctis.bootstrap import build_console, build_lake, build_memory
    from noctis.engine import MarketClock, build_runtime, initial_phase_for
    from noctis.engine.runtime import trading_roster

    # Off by default (WARNING) so a bare run stays quiet; the -v feed rides the Console below,
    # -vv drops stdlib logging to DEBUG. One ladder shared with `noctis research`.
    logging.basicConfig(
        level=_logging_level(verbose), format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # The composition root owns the ordering: safety gate → mandate → overlay → CLI flags.
    # Under `run` a pinned mandate's overlay IS the metric selector (there is no --metric).
    inputs = _resolve_session_or_exit(
        config,
        directive=directive,
        mandate=mandate,
        time_limit_hours=time_limit_hours,
        require_gate=True,
    )
    settings, mode, active_mandate = inputs.settings, inputs.mode, inputs.mandate
    assert mode is not None  # require_gate=True always resolves it
    _guard_legacy_or_exit(settings)

    clock = MarketClock(settings.session.calendar, settings.session.timezone)
    typer.echo(f"Noctis starting in {mode.upper()} mode.")
    _echo_research_engine(settings)
    typer.echo(f"Universe: {', '.join(settings.universe)}")
    typer.echo(f"Calendar: {settings.session.calendar} ({settings.session.timezone})")
    typer.echo(f"Market is currently {'OPEN' if clock.is_open() else 'CLOSED'}.")
    typer.echo(f"Initial phase: {initial_phase_for(clock).value}")
    _echo_mandate(active_mandate, inputs.overrides)

    memory = build_memory(settings)
    lake = build_lake(settings)

    # Optional, opt-in auto-backfill: before the readiness check, fetch missing history for
    # any not-yet-ready universe symbol (budget-gated). Off by default → zero fetches.
    missing = [s for s in settings.universe if not lake.check_symbol_ready(s)]
    if missing and settings.data.auto_backfill:
        if not settings.databento_api_key:
            typer.secho(
                "auto_backfill is on but no DATABENTO_API_KEY — skipping backfill.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        else:
            _auto_backfill(settings, lake, missing)

    # Readiness spans the trading roster (the growing universe): the config seed plus every symbol
    # the research agent has fetched into the lake.
    ready = [s for s in trading_roster(settings, lake) if lake.check_symbol_ready(s)]
    if not ready:
        typer.echo(
            "No catalog data yet — ingest history first (e.g. `noctis data ingest AAPL "
            "--start 2024-01-01 --end 2024-12-31`), then run again. Exiting cleanly."
        )
        return

    # One level-aware console renders the loop's typed events (phase banners + the research feed,
    # and the trading feed once P4 lands). Wired only when asked for — a bare run passes None, so
    # the runtime stays byte-identical to today: research falls back to its logger, no banners.
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=memory,
        clock=clock,
        mandate=active_mandate,
        on_event=build_console(verbose, show_reasoning=show_reasoning),
    )

    # SIGINT/SIGTERM route through one clean shutdown path (stops between phases).
    def _shutdown(signum, _frame):
        typer.echo(f"\nReceived signal {signum}; stopping cleanly after this phase…")
        runtime.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)

    result = runtime.run()
    typer.echo(
        f"Stopped ({result.stopped_reason}): {result.cycles_completed} cycle(s), "
        f"{result.research_iterations} candidates researched, {result.trades} paper orders."
    )


@app.command()
def status(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Show the resolved execution mode, market state, and a configuration summary."""
    from noctis.champions import build_registry
    from noctis.engine import MarketClock, initial_phase_for, resolve_trading_driver
    from noctis.engine.report_assembly import gather_account_forward

    settings, mode = _resolve_mode_or_exit(config)
    # status stays usable beside a legacy layout — it's the diagnostic you'd run first.
    _guard_legacy_or_exit(settings, warn_only=True)
    clock = MarketClock(settings.session.calendar, settings.session.timezone)
    is_open = clock.is_open()
    next_transition = clock.next_close() if is_open else clock.next_open()
    registry = build_registry(settings)
    # Account + per-champion forward track record: the same one-read gather the close
    # report assembles from, rendered as status lines. Degrades gracefully, never errors.
    af = gather_account_forward(settings.state_dir, registry.list())
    if af.account_corrupt:
        account_line = "CORRUPT — trading refuses; recover with `noctis account --reset`"
    elif af.account is None:
        account_line = "none yet (the first TRADING session opens one at 100,000.00)"
    else:
        account_line = (
            f"equity {af.account.equity:,.2f} ({af.account.cumulative_pnl:+,.2f} since "
            f"{af.account.opened}, {af.account.open_positions} open position(s))"
        )

    typer.echo(f"mode (resolved):   {mode}")
    typer.echo(f"mode (config):     {settings.mode}")
    typer.echo(f"allow_live (env):  {settings.allow_live}")
    typer.echo(f"market:            {'OPEN' if is_open else 'CLOSED'}")
    typer.echo(f"phase:             {initial_phase_for(clock).value}")
    typer.echo(f"next transition:   {next_transition.isoformat()}")
    typer.echo(f"champions:         {len(registry.list())}/{settings.champion_count}")
    if af.forward.corrupt:
        typer.echo("forward record:    unreadable ledger — omitted (trading unaffected)")
    elif not af.records:
        typer.echo("forward record:    none yet (accrues as champions trade live-holdout sessions)")
    else:
        typer.echo("forward record:")
        for r in af.records:
            typer.echo(
                f"  {r.family:<22} {r.forward_pnl:+,.2f}  "
                f"({r.sessions_traded} session(s) since {r.opened_session})"
            )
    typer.echo(f"account:           {account_line}")
    typer.echo(f"universe:          {', '.join(settings.universe)}")
    typer.echo(f"research_budget:   {settings.research_time_budget_minutes} min")
    typer.echo(f"data provider:     {settings.data.provider} (budget ${settings.data.budget_usd})")
    typer.echo(
        f"trading driver:    {resolve_trading_driver(settings)} "
        f"(execution={settings.trading.execution})"
    )


@app.command()
def report(
    as_of: str = typer.Option(None, "--as-of", help="Report date (YYYY-MM-DD); default latest."),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    sweep_stale: bool = typer.Option(
        False,
        "--sweep-stale",
        help="Archive reports dated after today (stale simulated-clock runs); dry-run by default.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="With --sweep-stale: preview only (default); pass --no-dry-run to actually move.",
    ),
) -> None:
    """Generate or retrieve the close-of-day report."""
    from pathlib import Path

    from noctis.bootstrap import build_memory
    from noctis.champions import build_registry
    from noctis.engine.report_assembly import assemble_report
    from noctis.reporting import latest_report, sweep_stale_reports, today_str, write_report

    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    reports_dir = settings.reports_dir

    # Opt-in, never automatic: a legitimately simulated run writes future-dated as-ofs on
    # purpose, so an auto-sweep would delete its own outputs. Dry-run by default.
    if sweep_stale:
        stale = sweep_stale_reports(reports_dir, apply=not dry_run)
        if not stale:
            typer.echo("No stale (future-dated) reports found.")
            return
        typer.echo(f"{'Would archive' if dry_run else 'Archived'} {len(stale)} stale report(s):")
        for p in stale:
            typer.echo(f"  {p.name}")
        if dry_run:
            typer.echo(f"\n(dry run — pass --no-dry-run to move them to {reports_dir}/archive/)")
        return

    if as_of:
        path = Path(reports_dir) / f"{as_of}.md"
        if not path.is_file():
            typer.echo(f"No report for {as_of}. Generating one from current state.")
        else:
            typer.echo(path.read_text())
            return
    else:
        existing = latest_report(reports_dir)
        if existing is not None:
            typer.echo(existing.read_text())
            return

    # Generate a fresh report from persisted state — the same assembly the CLOSE phase
    # runs, minus session activity (no trades/equity/events outside a live day-cycle).
    data = assemble_report(
        as_of=as_of or today_str(),
        mode=settings.mode,
        registry=build_registry(settings),
        memory=build_memory(settings),
        state_dir=settings.state_dir,
    )
    path = write_report(data, reports_dir)
    typer.echo(path.read_text())
    typer.echo(f"\n(written to {path})")


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Strategy family to backtest."),
    symbol: str = typer.Option(None, "--symbol", "-s", help="Symbol (default: first in universe)."),
    schema: str = typer.Option("ohlcv-1m", "--schema", help="Bar schema/resolution."),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Backtest a strategy from the library (or any registered family) on catalog data.

    Runs with the file's current ``Params`` defaults — after a champion promotion those are
    the tuned values, so this replays exactly what the research loop shipped.
    """
    from noctis.backtest import Candidate, PipelineConfig, evaluate
    from noctis.bootstrap import build_families, build_lake

    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    # One hydration (seeds → spec-families → library files): the library files are the
    # canonical versions of their families (tuned defaults live in the file), so a human
    # replays exactly what the agent shipped — and a minted spec champion replays too.
    families = build_families(settings)
    if strategy not in families:
        typer.secho(
            f"Unknown strategy '{strategy}'. Known: {', '.join(families.names())}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    lake = build_lake(settings)
    sym = symbol or settings.universe[0]
    if not lake.check_symbol_ready(sym):
        typer.echo(
            f"No catalog data for {sym}. Ingest first: "
            f"noctis data ingest {sym} --start <d> --end <d>"
        )
        return

    bars = lake.get_bars(settings.data.dataset, schema, [sym], 0, 2**63 - 1)[sym]
    # Strategies declare the bar granularity their thesis needs; the lake stays 1m.
    from noctis.data.aggregate import aggregate_bars, bars_per_year

    timeframe = families.get_class(strategy).timeframe
    bars = aggregate_bars(bars, timeframe)
    if len(bars) < 200:
        typer.echo(
            f"Only {len(bars)} {timeframe} bars for {sym}; need >=200 for walk-forward splits."
        )
        return

    # A panel of one, on the same auto geometry/metric research uses for this data length.
    scorecard = evaluate(
        Candidate(strategy, {}),
        {sym: bars},
        config=PipelineConfig.auto_from_settings(
            settings,
            len(bars),
            periods_per_year=bars_per_year(timeframe),
            prefilter_min_score=None,  # a replay always shows the full scorecard
        ),
        families=families,
    )
    n_splits = sum(len(ss.splits) for ss in scorecard.symbols.values())
    typer.echo(f"Backtest {strategy} on {sym} ({len(bars)} {timeframe} bars):")
    typer.echo(f"  metric:           {scorecard.metric_name}")
    typer.echo(f"  stage:            {scorecard.stage}")
    typer.echo(f"  splits:           {n_splits}")
    typer.echo(f"  avg train metric: {scorecard.avg_train_metric:.4f}")
    typer.echo(f"  avg test metric:  {scorecard.avg_test_metric:.4f}")
    typer.echo(f"  train-test gap:   {scorecard.gap:.4f}")
    if scorecard.holdout_metric is not None:
        typer.echo(f"  holdout metric:   {scorecard.holdout_metric:.4f}  (forward-holdout gate)")


@app.command()
def champions(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Drop every champion (history is kept) so the slots re-fill under the "
        "current gates and metric.",
    ),
) -> None:
    """Show the current champion registry."""
    from noctis.champions import build_registry

    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    registry = build_registry(settings)
    if reset:
        dropped = registry.reset("operator reset via `noctis champions --reset`")
        typer.echo(f"Dropped {dropped} champion(s); the registry is empty.")
        return
    entries = registry.list()
    if not entries:
        typer.echo("No champions yet. The research loop promotes them over time.")
        return
    current_metric = settings.promotion.metric
    typer.echo(
        f"{'family':<20} {'test_metric':>12} {'gap':>10}  {'metric':<14} {'crowned_at':<26} params"
    )
    for entry in entries:
        params = ", ".join(f"{k}={v}" for k, v in sorted(entry.params.items()))
        metric_name = entry.scorecard.metric_name
        metric_label = metric_name if metric_name == current_metric else f"{metric_name}(stale)"
        typer.echo(
            f"{entry.family:<20} {entry.test_metric:>12.4f} {entry.gap:>10.4f}  "
            f"{metric_label:<14} {entry.crowned_at:<26} {params}"
        )


@app.command()
def account(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Archive the account to paper_account.<date>.json in the state dir and start "
        "fresh (100k) next session. Also the recovery path for a corrupt account file.",
    ),
) -> None:
    """Show the continuous paper account — the cumulative forward track record.

    One paper account carries equity and open positions across TRADING sessions (the state
    dir's paper_account.json). Champion turnover never resets it; only --reset does.
    """
    from pathlib import Path

    from noctis.broker.persistence import AccountStore

    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    store = AccountStore(Path(settings.state_dir) / "paper_account.json")
    if reset:
        archive = store.reset()
        if archive is None:
            typer.echo("No paper account to reset.")
        else:
            typer.echo(f"Archived to {archive}; the next session starts fresh at 100,000.00.")
        return
    try:
        summary = store.summary()
    except RuntimeError as exc:
        typer.secho(f"ACCOUNT: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if summary is None:
        typer.echo("No paper account yet — the first TRADING session opens one at 100,000.00.")
        return
    typer.echo(f"opened:           {summary.opened}")
    typer.echo(f"last session:     {summary.last_session}")
    typer.echo(f"starting cash:    {summary.starting_cash:,.2f}")
    typer.echo(f"equity:           {summary.equity:,.2f}")
    typer.echo(f"cumulative P&L:   {summary.cumulative_pnl:+,.2f}")
    typer.echo(f"open positions:   {summary.open_positions}")


@app.command()
def research(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    max_iterations: int = typer.Option(
        None, "--max-iterations", help="Tool rounds this session (default from config)."
    ),
    directive: str = typer.Option(
        None,
        "--directive",
        "-d",
        help="Inline operator mandate for this session (overrides config), e.g. 'find a "
        "strategy on very volatile stocks; high risk appetite'. Mutually exclusive with "
        "--mandate.",
    ),
    mandate: str = typer.Option(
        None,
        "--mandate",
        help="Select a mandate under mandate_dir for this session (a profile name, MANDATE, "
        "or auto), overriding config. Mutually exclusive with --directive.",
    ),
    metric: str = typer.Option(
        None,
        "--metric",
        help="Scoring metric for this session (overrides config promotion.metric AND any "
        "mandate overlay): sharpe | sortino | total_return.",
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="-v prints each tool call; -vv adds DEBUG logs."
    ),
    show_reasoning: bool = typer.Option(
        False,
        "--show-reasoning",
        help="Surface the model's reasoning + narration inline (think/say) even without -vv. "
        "Only providers that return chain-of-thought over the API show reasoning (OpenAI's "
        "reasoning models do not); narration always shows.",
    ),
) -> None:
    """Run one agent research session against the current lake, observably.

    The agent formulates (or revises) a strategy in the library, matches symbols, iterates
    backtests/sweeps until the parameter space is exhausted, and reaches a verdict. Needs the
    [llm] extra and an API key for the configured ``research.model`` provider (OPENAI_API_KEY for
    ``openai/*``, ANTHROPIC_API_KEY for ``anthropic/*``; a local backend needs none).
    """
    from noctis.bootstrap import (
        build_console,
        build_families,
        build_lake,
        build_memory,
        build_research_session,
    )
    from noctis.champions import build_registry
    from noctis.research import provider_of

    logging.basicConfig(
        level=_logging_level(verbose), format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # The composition root owns the ordering (§5): load_settings → resolve_mandate →
    # apply_overrides → explicit CLI flags, so --metric still wins over a mandate overlay.
    inputs = _resolve_session_or_exit(config, directive=directive, mandate=mandate, metric=metric)
    settings, active_mandate = inputs.settings, inputs.mandate
    _guard_legacy_or_exit(settings)

    # One level-aware console renders the loop's typed events; --show-reasoning opens the
    # think/say streams without the full -vv DEBUG noise. Quiet (no -v, no flag) ⇒ None ⇒
    # the loop's own logging default handles events.
    console = build_console(verbose, show_reasoning=show_reasoning)
    session = build_research_session(
        settings=settings,
        lake=build_lake(settings),
        registry=build_registry(settings),
        families=build_families(settings),  # champions may be minted spec-families
        memory=build_memory(settings),
        mandate=active_mandate,
        on_event=console,
    )
    if session is None:
        resolved_model = settings.research.model or settings.research.agent.model
        provider = provider_of(resolved_model)
        key_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
        need_key = f" and {key_env}" if key_env else ""
        typer.secho(
            f"Agent research needs the [llm] extra{need_key} for model {resolved_model!r}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"Agent research session: model={session.model}, "
        f"profile={session.budgets.name}, "
        f"metric={settings.promotion.metric}, "
        f"budget={settings.research_time_budget_minutes} min, "
        f"max_iterations={max_iterations or session.budgets.max_iterations}, "
        f"exhaustion gate={settings.research.min_trials} trials"
    )
    _echo_mandate(active_mandate, inputs.overrides)
    summary = session.run(max_iterations=max_iterations)
    # Graceful degradation: a reasoning view (-vv or --show-reasoning) that surfaced no `think`
    # events almost always means the provider returns no chain-of-thought over the API — the
    # default OpenAI reasoning models are exactly this case. Say so once, so silence reads as
    # "expected for this provider", not "the feature is broken". Narration (say) is unaffected.
    if console is not None and (verbose >= 2 or show_reasoning) and not console.saw_think:
        console.hint(
            f"reasoning not surfaced by {provider_of(session.model)} — its reasoning models "
            f"return no raw chain-of-thought over the API; narration still shows"
        )
    # A standalone session counts toward periodic memory distillation too (the distillation
    # itself only ever runs at the day loop's CLOSE).
    from noctis.research.distill import bump_research_session

    bump_research_session(settings.state_dir)
    typer.echo(
        f"Session over ({summary.stopped_reason}): {summary.iterations} tool rounds, "
        f"{session.toolbox.backtests_run} backtests, {summary.promotions} promotion(s), "
        f"{summary.rejections} rejection(s)."
    )
    if summary.candidates:
        typer.echo(f"Strategies worked on: {', '.join(summary.candidates)}")
        typer.echo(
            f"Inspect: noctis strategies; {settings.state_dir}/experiments/<name>.jsonl; "
            f"{settings.memory_path}"
        )


@app.command()
def strategies(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """List the strategy library: status, style, symbols, tuned date, thesis."""
    from noctis.strategies.library import LibraryPaths, list_strategies

    settings = load_settings(config_path=config)
    infos = list_strategies(LibraryPaths.from_settings(settings))
    if not infos:
        typer.echo(f"No strategies in {settings.strategies_dir!r} yet.")
        return
    typer.echo(f"{'name':<22} {'status':<10} {'style':<16} {'tuned':<12} thesis")
    for info in infos:
        if info.get("error"):
            typer.secho(f"{info['name']:<22} BROKEN: {info['error']}", fg=typer.colors.RED)
            continue
        thesis = info["thesis"][:70] + ("…" if len(info["thesis"]) > 70 else "")
        typer.echo(
            f"{info['name']:<22} {info['status']:<10} {info['style']:<16} "
            f"{(info['tuned'] or '-'):<12} {thesis}"
        )
        if info["symbols"]:
            typer.echo(f"{'':<22} symbols: {' '.join(info['symbols'])}  params: {info['params']}")


@app.command()
def setup(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    check: bool = typer.Option(
        False,
        "--check",
        help="Audit the install read-only (files, components, keys, LLM) and exit 1 on gaps.",
    ),
    databento_key: str = typer.Option(
        None, "--databento-key", help="DataBento API key to save into .env (skips the prompt)."
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="LiteLLM provider/model to configure, e.g. anthropic/claude-sonnet-5 or "
        "ollama_chat/noctis-qwen3:14b (skips the LLM menu).",
    ),
    api_key: str = typer.Option(
        None, "--api-key", help="API key for --model's hosted provider, saved into .env."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Never prompt: take every default, skip unanswerable steps."
    ),
) -> None:
    """Guided first-run setup — everything needed to research and paper-trade.

    Scaffolds the local files, installs the optional components (`uv sync --all-extras`),
    collects the DataBento key, connects an LLM (hosted API key or a local
    Ollama/noctis-ollama backend), and verifies it answers with one real completion.
    Idempotent and edit-preserving — re-run it any time.
    """
    from noctis.onboarding import run_setup

    code = run_setup(
        config_path=config,
        check_only=check,
        databento_key=databento_key,
        model=model,
        api_key=api_key,
        assume_yes=yes,
    )
    if code:
        raise typer.Exit(code=code)


@app.command()
def init(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Scaffold the local input files (config.yaml, .env, mandate/MANDATE.md) + workspace.

    The non-interactive core of `noctis setup` — idempotent and never overwrites: an
    existing local file is kept untouched, so re-running after edits is always safe. For
    the guided first-run experience (components, keys, LLM verify) use `noctis setup`.
    """
    from noctis.bootstrap import scaffold_init

    settings = load_settings(config_path=config)
    for line in scaffold_init(settings):
        typer.echo(line)


@app.command()
def migrate(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the migration plan without moving anything."
    ),
) -> None:
    """Move a legacy (pre-workspace) layout into the workspace, one artifact at a time.

    Covers state/, data_lake/, reports/, MEMORY.md, and the strategies/__tmp|champions
    tiers. Refuses (with a list) when a legacy artifact and its workspace counterpart
    both exist — resolve by hand, then re-run. config.yaml never moves.
    """
    from noctis.bootstrap import execute_migration, plan_migration

    settings = load_settings(config_path=config)
    plan = plan_migration(settings)
    if plan.conflicts:
        lines = "\n".join(f"  {a.legacy}  AND  {a.configured}  both exist" for a in plan.conflicts)
        typer.secho(
            f"Refusing to migrate — resolve these by hand first (keep one, remove the other):\n"
            f"{lines}\nNothing was moved.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    for path in plan.pinned:
        typer.secho(
            f"note     {path} stays put — a config knob explicitly points at it "
            f"(remove the override in config.yaml to adopt the workspace location)",
            fg=typer.colors.YELLOW,
        )
    if not plan.moves:
        typer.echo("Nothing to migrate — no un-migrated legacy artifacts found.")
        return
    verb = "would move" if dry_run else "moved"
    if not dry_run:
        execute_migration(plan)
    for artifact in plan.moves:
        typer.echo(f"{verb}  {artifact.legacy}  →  {artifact.configured}")
    if dry_run:
        typer.echo("\n(dry run — re-run without --dry-run to perform these moves)")
    else:
        typer.echo(f"\nMigrated {len(plan.moves)} artifact(s) into the workspace.")


# --- data sub-app -----------------------------------------------------------------------

data_app = typer.Typer(help="Market-data lake operations (fetch-once).", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _utcnow() -> datetime:
    """Current UTC time. Indirected so tests can freeze the backfill window deterministically."""
    return datetime.now(UTC)


@contextmanager
def _symbol_progress(verb: str):
    """Yield a per-symbol progress callback: an animated spinner on a TTY, plain lines otherwise.

    A multi-symbol DataBento fetch can run for minutes with nothing on screen; this is the
    operator's only still-alive signal ('ingesting AAPL (3/12)…'). Progress rides stderr so
    stdout stays clean for the per-symbol result lines.
    """
    if sys.stderr.isatty():
        from rich.console import Console

        status = Console(stderr=True).status(f"{verb}…")

        def report(symbol: str, index: int, total: int) -> None:
            status.update(f"{verb} {symbol} ({index}/{total})…")

        with status:
            yield report
    else:

        def report(symbol: str, index: int, total: int) -> None:
            typer.echo(f"{verb} {symbol} ({index}/{total})…", err=True)

        yield report


def _auto_backfill(settings, lake, missing: list[str]) -> None:
    """Fetch missing history for ``missing`` symbols over the ``history_days`` lookback window.

    Budget-gated by the cost preflight inside ``ensure_coverage`` (already-covered ranges are
    ``$0`` no-ops). A preflight refusal is surfaced as a per-symbol ``refused`` result — it does
    not raise — so the run continues cleanly on to the readiness check either way.
    """
    from noctis.data.types import NS_PER_DAY, t1_boundary_ns

    schema = "ohlcv-1m"
    # T+1 boundary (UTC midnight of the current ET trading date, i.e. through end of
    # yesterday) — the same boundary the nightly sync uses. Vendor availability ends
    # there; requesting past it is rejected outright (422 data_end_after_available_end,
    # or a 403 license error once the end crosses into the current ET session).
    end_ns = t1_boundary_ns(_utcnow())
    start_ns = end_ns - settings.data.history_days * NS_PER_DAY
    typer.echo(
        f"Auto-backfilling {len(missing)} symbol(s) over {settings.data.history_days} days "
        f"(budget ${settings.data.budget_usd})…"
    )
    try:
        with _symbol_progress("ingesting") as progress:
            results = lake.ensure_coverage(
                settings.data.dataset, schema, missing, start_ns, end_ns, on_progress=progress
            )
    except Exception as exc:  # noqa: BLE001 — never let a backfill error crash the run
        typer.secho(
            f"Auto-backfill error: {exc}; continuing without it.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return
    for symbol, res in results.items():
        cost = f"${res.padded_cost:.4f}" if res.padded_cost else "$0"
        line = f"  {symbol}: {res.status} ({res.fetch_calls} fetches, {cost}) {res.detail}"
        typer.echo(line.rstrip())


def _vendor_lake_or_exit(settings):
    """A lake that can fetch, or a red message + non-zero exit when no vendor key is set."""
    from noctis.bootstrap import MissingVendorKey, build_lake

    try:
        return build_lake(settings, require_vendor=True)
    except MissingVendorKey as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@data_app.command("status")
def data_status(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Show tracked series in the coverage registry."""
    from noctis.bootstrap import build_lake

    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    lake = build_lake(settings)
    records = lake.coverage_records()
    if not records:
        typer.echo("No tracked series. Run 'noctis data ingest' to populate the lake.")
        return
    typer.echo(f"{'dataset':<12} {'schema':<10} {'symbol':<8} {'rows':>8} {'status':<10}")
    for rec in records:
        typer.echo(
            f"{rec.dataset:<12} {rec.schema:<10} {rec.symbol:<8} "
            f"{rec.row_count:>8} {rec.status:<10}"
        )


@data_app.command("sync")
def data_sync(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Incrementally extend every tracked series to the T+1 boundary (tail only)."""
    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    lake = _vendor_lake_or_exit(settings)
    with _symbol_progress("syncing") as progress:
        results = lake.sync(on_progress=progress)
    if not results:
        typer.echo("Nothing to sync (no tracked series).")
        return
    for symbol, res in results.items():
        typer.echo(f"{symbol}: {res.status} (+{res.rows_added} rows) {res.detail}".rstrip())


@data_app.command("ingest")
def data_ingest(
    symbols: str = typer.Argument(..., help="Comma-separated symbols, e.g. AAPL,MSFT."),
    start: str = typer.Option(..., "--start", help="Start date (ISO, inclusive)."),
    end: str = typer.Option(..., "--end", help="End date (ISO, inclusive)."),
    schema: str = typer.Option("ohlcv-1m", "--schema", help="Vendor schema/resolution."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Price the ingest without spending."),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Coverage-diffed ingest of a date range (only missing slices are fetched)."""
    from noctis.data.types import to_ns, to_ns_end_inclusive

    settings = load_settings(config_path=config)
    _guard_legacy_or_exit(settings)
    lake = _vendor_lake_or_exit(settings)
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    # ``--end`` is inclusive: a date-only end covers that whole trading day (its intraday
    # bars run past midnight), so map it to end-of-day ns rather than the day's midnight.
    with _symbol_progress("pricing" if dry_run else "ingesting") as progress:
        results = lake.ensure_coverage(
            settings.data.dataset,
            schema,
            syms,
            to_ns(start),
            to_ns_end_inclusive(end),
            dry_run=dry_run,
            on_progress=progress,
        )
    for symbol, res in results.items():
        cost = f"${res.padded_cost:.4f}" if res.padded_cost else "$0"
        line = f"{symbol}: {res.status} ({res.fetch_calls} fetches, {cost}) {res.detail}"
        typer.echo(line.rstrip())


if __name__ == "__main__":
    app()
