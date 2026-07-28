"""Command-line interface (Typer).

Commands: ``setup`` (the guided first-run wizard), ``init``/``migrate`` (scaffold /
legacy-layout move), ``run``, ``runs``/``run-record`` (the run board and one run's record),
``status``, ``mandate`` (the steering preflight), ``engine`` (the
engine's identity and comparability key), ``report``, ``backtest``, ``champions``, ``account``
(the continuous paper account), ``research`` (one observable agent research session),
``strategies`` (the authored library index), and the ``data`` sub-app.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NoReturn

import typer

from noctis.config import SafetyGateError, load_settings, resolve_execution_mode

if TYPE_CHECKING:  # the composition root imports at call time (fast `--help`), types don't
    from noctis.bootstrap import OverrideChange, SessionInputs


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


def _exit_red(exc: Exception, prefix: str = "") -> NoReturn:
    """Red text on stderr + exit 1 — the one mapping every typed startup error takes."""
    typer.secho(f"{prefix}{exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


def _resolve_session_or_exit(config: str | None, **kwargs):
    """Resolve the session inputs (the composition root's precedence chain), mapping each
    typed startup error to red text + a non-zero exit. Errors are loud at startup by design:
    a typo'd selector or a closed safety gate must never silently un-steer a multi-day run.

    The four resume refusals map here too, for the same reason: an unresumable run must be said
    out loud at the start rather than turned into a *new* run nobody asked for.
    """
    from noctis.bootstrap import UsageError, resolve_session
    from noctis.config.rehydrate import RehydrationError
    from noctis.observability.engine_change import EngineChangeError
    from noctis.reporting.run_store import RunCompletedError, RunNotFoundError
    from noctis.research import MandateError

    try:
        return resolve_session(config, **kwargs)
    except UsageError as exc:  # --directive × --mandate, or an unknown --metric
        _exit_red(exc)
    except MandateError as exc:
        _exit_red(exc, prefix="MANDATE: ")
    except SafetyGateError as exc:
        _exit_red(exc, prefix="SAFETY GATE: ")
    except (RunNotFoundError, RunCompletedError) as exc:
        _exit_red(exc, prefix="RESUME: ")
    except (RehydrationError, EngineChangeError) as exc:
        _exit_red(exc, prefix="RESUME: ")


def _resolve_status_session(config: str | None) -> tuple[SessionInputs, str | None]:
    """Resolve the whole session for ``status``, degrading an unusable mandate to a report.

    ``status`` is the command an operator runs *because* something is wrong, so a mandate the
    overlay refuses must not take the rest of the diagnosis down with it: the refusal is
    returned as text for the mandate line, the remaining values fall back to the pre-overlay
    settings, and the command still exits 0 — the same warn-don't-exit contract this one
    already keeps beside an un-migrated legacy layout. ``run``/``research`` still refuse
    outright, which is where a fatal configuration error belongs.

    The safety gate is deliberately **not** degraded (a mode that will not resolve is not a
    configuration ``status`` can narrate), and it resolves before the mandate, so the fallback
    reload below is only ever reached with the gate already open.
    """
    from noctis.bootstrap import SessionInputs, resolve_session
    from noctis.research import MandateError

    try:
        return resolve_session(config, require_gate=True), None
    except MandateError as exc:
        settings, mode = _resolve_mode_or_exit(config)
        return SessionInputs(settings=settings, mode=mode, mandate=None, overrides=[]), str(exc)
    except SafetyGateError as exc:
        _exit_red(exc, prefix="SAFETY GATE: ")


def _guard_legacy_or_exit(settings, *, warn_only: bool = False) -> None:
    """One guard over both legacy generations, with one instruction: ``noctis migrate``.

    Two things can be un-migrated, and they differ in what *continuing* would cost — so they
    differ in severity, never in the remedy:

    * **Pre-workspace artifacts** beside ``config.yaml`` would be genuinely abandoned: nothing
      looks there again, so a run would start against a silently-empty champion board while the
      real data rots at the old path. That is a **refusal** (exit 2); ``status`` only warns,
      because it is the diagnostic an operator runs *because* something is wrong.
    * **Pre-run-scoped ``workspace/`` state** is not abandoned at all — it sits in the same
      workspace, and the run starting beside it is a new run with its own board, which is exactly
      what run-scoped state means. That is a **warning**, always, on every command.

    Both are reported in one message so an operator is never handed two competing instructions.
    """
    from noctis.bootstrap import detect_legacy_layout, detect_unadopted_state

    abandoned = detect_legacy_layout(settings)
    unadopted = detect_unadopted_state(settings)
    blocks: list[str] = []
    if abandoned:
        blocks.append(
            "Legacy (pre-workspace) layout detected — running now would abandon:\n"
            + "\n".join(
                f"  {a.legacy}  (configured location {a.configured} does not exist)"
                for a in abandoned
            )
        )
    if unadopted:
        blocks.append(
            "Pre-run-scoped state detected — a run's champions, account, memory and reports now "
            "live under its own run, and this state belongs to no run yet:\n"
            + "\n".join(f"  {a.legacy}  →  {a.configured}" for a in unadopted)
        )
    if not blocks:
        return
    message = "\n".join(
        [
            *blocks,
            "Run `noctis migrate` to move them where the engine now reads them (adopting the "
            "workspace state into the reserved `legacy` run), or point the knobs at them "
            "explicitly in config.yaml.",
        ]
    )
    if abandoned and not warn_only:
        typer.secho(message, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    typer.secho(f"WARNING: {message}", fg=typer.colors.YELLOW, err=True)


def _echo_mandate(mandate, override_lines: list[str]) -> None:
    """Print the resolved mandate provenance + **every** applied config override at session start.

    The overlay carries the whole run configuration now — which model thinks, what it may spend,
    how much history it fetches, what "good" is scored as — so the assembled configuration has to
    land in the log an operator already reads, not just the mandate's name. One header names the
    mandate that moved them (the provenance a multi-mandate install needs), then one ``k=v`` line
    per override, in the sorted order ``apply_patch`` read back off the validated settings — so
    the echo shows the value the run will actually use, not the value the file asked for.

    Nothing to say stays silent: no mandate, or a mandate whose overlay applied nothing, prints
    exactly what it printed before this existed.
    """
    if mandate is not None:
        typer.echo(f"Mandate: {mandate.source}")
    if not override_lines:
        return
    source = mandate.source if mandate is not None else "?"
    typer.echo(f"  mandate {source} overrides:")
    for line in override_lines:
        typer.echo(f"    {line}")


def _echo_status_mandate(mandate, override_lines: list[str], error: str | None) -> None:
    """The provenance footer of ``noctis status``: what steered this configuration, and how.

    Three states, one shape. Resolved: the mandate's source, then every applied ``k=v``
    override (or ``none``) — the same echo lines the kickoff prints, so the two surfaces can
    never disagree about what applied. No mandate: said explicitly, because "unconstrained" is
    a configuration too. Unusable: the overlay's own refusal, verbatim and in red, plus the
    warning that every value above it is pre-overlay — the diagnosis an operator opened
    ``status`` for, rather than the traceback ``run`` would have handed them.
    """
    if error is not None:
        typer.secho(
            "mandate:           UNUSABLE — the values above are pre-overlay; `run` would "
            "refuse to start:",
            fg=typer.colors.RED,
        )
        for line in error.splitlines():
            typer.echo(f"  {line}")
    elif mandate is None:
        typer.echo("mandate:           none (unconstrained)")
    else:
        typer.echo(f"mandate:           {mandate.source}")
    if not override_lines:
        typer.echo("overrides:         none")
        return
    typer.echo("overrides:")
    for line in override_lines:
        typer.echo(f"  {line}")


def _echo_mandate_preflight(mandate, changes: list[OverrideChange]) -> None:
    """The whole body of ``noctis mandate``: what this selector resolved to, and what it moves.

    Same label gutter as ``status`` (and the same ``overrides:`` block, ``none`` included), so
    the two read as one tool: an operator who has seen one has read the other. The diff is the
    part ``status`` cannot show — every bound path with the value config resolved for it *and*
    the value the run would use — because a preflight's question is "what would this change?",
    not "what is set?".
    """
    if mandate is None:  # an empty MANDATE.md — the one selector that resolves to no steering
        typer.echo("mandate:           none — this selector resolves to no steering at all")
        typer.echo("overrides:         none (a run under it would be unconstrained)")
        return
    typer.echo(f"mandate:           {mandate.source}")
    typer.echo(f"summary:           {mandate.summary or 'none'}")
    typer.echo(f"symbols:           {', '.join(mandate.symbols) if mandate.symbols else 'none'}")
    if not mandate.references:
        typer.echo("references:        none")
    else:
        typer.echo("references:")
        for ref in mandate.references:
            typer.echo(f"  {ref.path} ({len(ref.text.encode('utf-8'))} bytes)")
    if not changes:
        typer.echo("overrides:         none (this mandate binds no settings)")
        return
    typer.echo("overrides:")
    width = max(len(change.path) for change in changes)
    for change in changes:
        typer.echo(f"  {change.path:<{width}}  {change.before} → {change.after}")


def _show_config_drift(config: str | None, resume: str | None) -> None:
    """``--show-config-drift``: print what a rebase *would* adopt, and leave the run alone.

    An inspection, structurally: it resolves the current files exactly as a first start would,
    reads the addressed record, and renders the diff. It opens no segment, takes no lock and
    writes nothing — a command an operator runs to decide must not itself be a decision.

    The comparison is the freezing tiers' own (:func:`~noctis.config.rehydrate.config_drift`), so
    what is shown here is exactly what ``--rebase-config`` would write: the frozen keys and the
    resolved mandate text, never the live tier (always this process's, so never drift) and never
    the refused pair (never recorded, never restored, never rebasable).
    """
    from noctis.bootstrap import UsageError
    from noctis.config.rehydrate import config_drift
    from noctis.reporting.run_store import RunNotFoundError, read_run_record

    if resume is None:
        _exit_red(
            UsageError(
                "--show-config-drift compares a run's frozen config against the current files, so "
                "it needs the run: pass --resume <address> (`noctis runs` lists them)."
            )
        )
    current = _resolve_session_or_exit(config)
    try:
        record = read_run_record(current.settings.runs_dir, resume)
    except RunNotFoundError as exc:
        _exit_red(exc, prefix="RESUME: ")
    run = record.get("run")
    run_id = str((run.get("run_id") if isinstance(run, Mapping) else None) or resume)
    inputs = record.get("inputs")
    inputs = inputs if isinstance(inputs, Mapping) else None
    drift = config_drift(
        record, current.settings, mandate=current.mandate, overrides=current.overrides
    )
    _echo_config_drift(drift, run_id=run_id, inputs=inputs)


def _finish_run(config: str | None, resume: str | None) -> None:
    """``--finish``: seal one run as ``completed`` — terminal — and run nothing.

    The deliberate counterpart of the run-level cap (story #136). It takes the address the operator
    would have resumed, writes one stamp into that run's record, and exits: no segment is opened,
    no engine starts, and the liveness lock is read only far enough to refuse a run another process
    is working. Sealing a run that is already completed is a **no-op** that says so — terminal means
    terminal, so the moment a published result was sealed is never rewritten.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    from noctis.bootstrap import UsageError
    from noctis.reporting.run_store import RunLockedError, RunNotFoundError, finish_run

    if resume is None:
        _exit_red(
            UsageError(
                "--finish marks an existing run completed, so it needs the run: pass "
                "--resume <address> (`noctis runs` lists them). A run being minted right now has "
                "produced nothing to publish."
            )
        )
    settings = load_settings(config_path=config)
    try:
        outcome = finish_run(
            Path(settings.runs_dir),
            resume,
            clock=lambda: datetime.now(UTC),
            election_metric=settings.promotion.metric,
        )
    except RunNotFoundError as exc:
        _exit_red(exc, prefix="FINISH: ")
    except RunLockedError as exc:
        _exit_red(exc, prefix="RUN LOCKED: ")
    if not outcome.sealed:
        typer.echo(
            f"Run {outcome.run_id} was already completed (at {outcome.completed_utc}) — nothing "
            "to do. `completed` is terminal, so the stamp it was sealed with stands."
        )
        return
    typer.echo(
        f"Run {outcome.run_id} is now completed (at {outcome.completed_utc}). That is terminal: "
        "it refuses resume from here, so the numbers quoted from its record can never gain "
        "another segment. `noctis run-record` prints it."
    )


def _echo_run_limit(limit_hours: float | None, *, spent_s: float) -> None:
    """State the run's compute budget and what is left of it, when it has one.

    Printed at start beside the run id, because a cap an operator cannot see is one they will be
    surprised by: this is the line that says tonight may end early — and why."""
    if limit_hours is None:
        return
    remaining = max(0.0, limit_hours * 3600.0 - spent_s)
    typer.echo(
        f"Run limit: {limit_hours:g}h of cumulative runtime "
        f"({_runtime(spent_s)} used, {_runtime(remaining)} left; the run completes at the cap)"
    )


def _echo_config_drift(drift, *, run_id: str, inputs: Mapping | None) -> None:
    """Render one run's config drift — the only place the diff becomes text.

    Three shapes, because they are three different situations: a run that froze nothing has
    nothing to compare, a run in step with the files has nothing to adopt, and a run that has
    drifted needs every changed key with both values side by side. The closing lines always say
    what happens *next* — frozen wins unless you rebase — because the diff itself decides nothing.
    """
    if not drift.frozen:
        typer.echo(
            f"Run {run_id} froze no configuration (it predates config freezing, or it is history "
            "adopted by `noctis migrate`), so there is nothing to compare. Its next resume "
            "freezes the current config.yaml and mandate/ onto it."
        )
        return
    epoch = inputs.get("config_epoch") if inputs is not None else None
    frozen_at = inputs.get("frozen_at_utc") if inputs is not None else None
    if not drift:
        typer.echo(
            f"No config drift for run {run_id}: the current config.yaml and mandate/ still match "
            f"what it froze (config_epoch {epoch}, frozen at {frozen_at})."
        )
        return
    typer.echo(f"Config drift for run {run_id} (config_epoch {epoch}, frozen at {frozen_at}):")
    typer.echo("")
    if drift.settings:
        typer.echo("settings (frozen tier — a resume ignores the current files for these):")
        width = max(len(change.path) for change in drift.settings)
        for change in drift.settings:
            typer.echo(f"  {change.path:<{width}}  {change.frozen!r} → {change.current!r}")
    if drift.mandate is not None:
        mandate = drift.mandate
        typer.echo("mandate (frozen as resolved text, not as a selector):")
        typer.echo(f"  source       {mandate.frozen_source} → {mandate.current_source}")
        frozen_sha, current_sha = _short(mandate.frozen_sha256), _short(mandate.current_sha256)
        typer.echo(f"  text_sha256  {frozen_sha} → {current_sha}")
        typer.echo(f"  frozen text  {_excerpt(mandate.frozen_text)}")
        typer.echo(f"  current text {_excerpt(mandate.current_text)}")
    typer.echo("")
    typer.echo(
        "Frozen wins: resuming this run keeps the left-hand values, which is what makes its "
        "accumulated results mean one thing. Adopt the right-hand ones deliberately with "
        "`--rebase-config` (it bumps config_epoch and records a before/after entry)."
    )
    typer.echo(
        "The live tier — paths, secrets, and per-process budgets like --time-limit-hours — is "
        "always this process's and is never drift; `mode`/`allow_live` are never rebasable."
    )


def _short(digest: str | None) -> str:
    """A digest as a label, not a wall of hex — the full value is in the record."""
    return "none" if digest is None else digest[:12]


def _excerpt(text: str | None, limit: int = 160) -> str:
    """One mandate body as a single readable line, truncated honestly."""
    if text is None:
        return "none"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}… ({len(flat)} chars)"


def _echo_config_rebase(rebase: Mapping | None, *, rebased: bool) -> None:
    """Say what ``--rebase-config`` did — including when it deliberately did nothing.

    The no-op is worth a line of its own: an operator who asked to adopt the current files and is
    told nothing would reasonably assume an epoch moved. It did not, because nothing had changed.
    """
    if not rebased:
        return
    if rebase is None:
        typer.echo(
            "Config rebase: no drift — this run's frozen config already matches the "
            "current config.yaml and mandate/, so config_epoch is unchanged."
        )
        return
    change = rebase["config_changes"][-1]
    moved = len(change["settings"])
    parts = [f"{moved} setting(s)"] if moved else []
    if change["mandate"] is not None:
        parts.append("the mandate text")
    typer.echo(
        f"Config rebased: config_epoch {change['from_epoch']} → {change['to_epoch']} "
        f"({', '.join(parts)}), recorded in segment {change['segment']}."
    )


def _echo_engine_change(store, inputs, *, upgrading: bool) -> None:
    """Say — and **record** — that this run resumed under a different engine (story #135).

    Both, always, and in that order of importance: the record is what an experiment is judged from
    months later, so an engine change that only ever appeared in a terminal would be invisible
    exactly when it mattered. The event lands against the segment it is true of, which is why it is
    written per resume rather than once per run.

    A refused resume never reaches here (the policy raises during session assembly), and a resume
    that found no drift prints and records nothing at all — silence is the signal.
    """
    for note in inputs.engine_notes:
        typer.secho(f"Engine change: {note}", fg=typer.colors.YELLOW)
        store.note(note)
    if not upgrading:
        return
    upgrade = inputs.engine_upgrade
    if upgrade is None:
        typer.echo(
            "Engine upgrade: nothing to accept — the components that decide what passes and what "
            "a number means still match what this run was created under, so engine_epoch is "
            "unchanged."
        )
        return
    moved = ", ".join(str(component["component"]) for component in upgrade["components"])
    typer.secho(
        f"Engine upgraded: engine_epoch {upgrade['from_epoch']} → {upgrade['to_epoch']} ({moved}), "
        f"recorded in segment {upgrade['segment']}. This run is flagged mixed_engine and its "
        "comparable key follows the new engine from here on.",
        fg=typer.colors.YELLOW,
    )


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
        None,
        "--time-limit-hours",
        help="Stop THIS PROCESS after this many hours (overrides config). Bounds tonight, not the "
        "experiment: the run stays resumable and the same id keeps accumulating tomorrow.",
    ),
    run_limit_hours: float = typer.Option(
        None,
        "--run-limit-hours",
        help="Give this RUN a compute budget, in hours of cumulative runtime across every "
        "stop/resume. Accepted at run creation only and frozen into the record: once the run's "
        "accumulated runtime crosses it the loop stops cleanly between phases and the run is "
        "marked `completed` — terminal, so it refuses resume. That is how 'run this mandate for "
        "100 hours, then stop' is expressed, and it is what lets two runs be compared on equal "
        "compute. Not settable on --resume (the cap defines the experiment).",
    ),
    embed_all_sources: bool = typer.Option(
        False,
        "--embed-all-sources",
        help="Archive this RUN whole: embed every candidate's strategy source in the run record, "
        "not just the champions'. Accepted at run creation only and frozen into the record — what "
        "the artifact contains is part of what the run is, and a flag passed on some nights and "
        "not others would make the record's contents depend on how it was last invoked. Off by "
        "default: champions are embedded in full and every other candidate is named by a "
        "run-relative path plus a content hash, which keeps a fortnight's record in the hundreds "
        "of kilobytes. Turn it on when an experiment is worth keeping after its workspace is not.",
    ),
    finish: bool = typer.Option(
        False,
        "--finish",
        help="With --resume: mark that run `completed` — terminal — and exit. Runs NO segment and "
        "starts no engine; it only refuses when another process holds the run. `completed` is the "
        "published state: a result that has been cited must not be able to gain more segments "
        "afterwards. A no-op on a run that is already completed.",
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Continue an existing run instead of minting a new one: it appends a segment and "
        "keeps accumulating research hours, trials, champions and P&L into the same record, "
        "under the config that run froze at creation. Four address forms, tried in this order: "
        "a path to a run.json (or the run dir); @LABEL (the label first, then the same name as "
        "an id); the reserved word `latest` (the most recently active run that is not completed, "
        "never one merely named or labelled `latest`); a run id. A bare address is ALWAYS the "
        "id. `noctis runs` lists ids and labels.",
    ),
    label: str = typer.Option(
        None,
        "--label",
        help="Attach a human alias to this run, stored in its record and shown by `noctis runs`; "
        "address it later with `--resume @LABEL`. Convenience only — the id stays the identity, "
        "and a label may be reused, in which case @LABEL refuses rather than guess. Also accepted "
        "with --resume, where it renames the run it addressed (a nickname decides nothing).",
    ),
    show_config_drift: bool = typer.Option(
        False,
        "--show-config-drift",
        help="With --resume: print how the current config.yaml and mandate/ differ from the "
        "config that run froze, then exit. Inspection only — it opens no segment, takes no lock "
        "and writes nothing. Drift alone is normal and costs nothing: a resume keeps using the "
        "frozen values. Only frozen keys and the resolved mandate text are compared; the live "
        "tier (paths, secrets, per-process budgets) is always this process's and is never drift.",
    ),
    rebase_config: bool = typer.Option(
        False,
        "--rebase-config",
        help="With --resume: deliberately adopt the current config.yaml and mandate/ for the rest "
        "of this run. Bumps inputs.config_epoch and appends a before/after config_change entry "
        "naming the segment, so a run whose config changed mid-flight always says so. A NO-OP "
        "when there is no drift — the epoch never moves for nothing. `mode` and `allow_live` are "
        "never rebasable: the safety gate re-resolves from two independent sources every start.",
    ),
    allow_engine_upgrade: bool = typer.Option(
        False,
        "--allow-engine-upgrade",
        help="With --resume: accept that this checkout's ENGINE has changed since the run was "
        "created, in the components that decide what passes and what a number means. Without it "
        "such a resume is REFUSED — champions crowned under two sets of gates must not accumulate "
        "inside one experiment. Accepting bumps engine.engine_epoch, appends an engine_changes "
        "entry naming every component that moved, and flags the run mixed_engine for good. A "
        "no-op when only the searcher tier moved (that warns, records and proceeds anyway).",
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
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Record an hour-segmented QA report of the whole run under qa_dir "
        "(funnel counts, per-strategy fates, phase timing, events.jsonl, a stamped manifest). "
        "Records silently — it never turns on the -v console feed. Off by default.",
    ),
) -> None:
    """Start Noctis. Loads config + memory, resolves the safety gate, runs the loop."""
    import signal
    import sys

    from noctis.bootstrap import (
        UsageError,
        build_event_sink,
        build_lake,
        build_memory,
        build_recorder,
        open_run_store,
        segment_counters,
        segment_phase_seconds,
    )
    from noctis.engine import MarketClock, build_runtime, initial_phase_for
    from noctis.engine.runtime import trading_roster
    from noctis.reporting.run_store import RunLockedError
    from noctis.research import client_status

    # Off by default (WARNING) so a bare run stays quiet; the -v feed rides the Console below,
    # -vv drops stdlib logging to DEBUG. One ladder shared with `noctis research`.
    logging.basicConfig(
        level=_logging_level(verbose), format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # Seeing what you would adopt is not adopting it (story #134): --show-config-drift prints the
    # diff and exits *before* anything opens a run, so an inspection never becomes a decision.
    if show_config_drift and rebase_config:
        _exit_red(
            UsageError(
                "--show-config-drift inspects and exits; --rebase-config adopts. Pass one: look "
                "first, then decide."
            )
        )
    # Sealing a run is not working one (story #136): --finish writes one stamp into the record and
    # exits before any session is assembled, so it can never be the thing that starts a segment.
    # Every flag that shapes the segment it would have run is therefore refused beside it rather
    # than silently ignored — a flag an operator typed must always mean something.
    shaping = [
        name
        for name, given in (
            ("--show-config-drift", show_config_drift),
            ("--rebase-config", rebase_config),
            ("--allow-engine-upgrade", allow_engine_upgrade),
            ("--run-limit-hours", run_limit_hours is not None),
            ("--time-limit-hours", time_limit_hours is not None),
            ("--embed-all-sources", embed_all_sources),
        )
        if given
    ]
    if finish and shaping:
        _exit_red(
            UsageError(
                f"--finish ends a run; {', '.join(shaping)} shapes the segment it would run next. "
                "A completed run has no next segment, so pass one: run the segment, or finish "
                "the run."
            )
        )
    if show_config_drift:
        _show_config_drift(config, resume)
        return
    if finish:
        _finish_run(config, resume)
        return

    # The composition root owns the ordering: safety gate → mandate → overlay → CLI flags.
    # Under `run` a pinned mandate's overlay IS the metric selector (there is no --metric).
    inputs = _resolve_session_or_exit(
        config,
        directive=directive,
        mandate=mandate,
        time_limit_hours=time_limit_hours,
        run_limit_hours=run_limit_hours,
        embed_all_sources=embed_all_sources,
        require_gate=True,
        resume=resume,
        rebase_config=rebase_config,
        allow_engine_upgrade=allow_engine_upgrade,
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

    # Always-on run identity (story #129): every invocation mints a run, gets its own tree under
    # workspace/runs/<run_id>/, takes the liveness lock, and writes one self-describing run.json.
    # A live lock is the one fatal failure in this subsystem — two engines writing one run would
    # corrupt it — so it exits non-zero here instead of degrading to a shared write.
    # …and `--resume <run_id>` continues one instead: the same tree, the same record, one more
    # segment, under the configuration that run froze at creation (story #132). The run — not this
    # process — is the unit progress accumulates on.
    try:
        store = open_run_store(
            settings,
            argv=sys.argv[1:],
            command="run",
            run_id=resume,
            resume=resume is not None,
            label=label,
            mandate=active_mandate,
            mode=mode,
            overrides=inputs.overrides,
            rebase=inputs.rebase,
            engine_upgrade=inputs.engine_upgrade,
        )
    except RunLockedError as exc:
        _exit_red(exc, prefix="RUN LOCKED: ")
    typer.echo(f"{'Resumed run' if resume else 'Run'}: {store.run_id}")
    typer.echo(f"Run record: {store.record_path}")
    _echo_run_limit(settings.run_limit_hours, spent_s=store.prior_runtime_s)
    _echo_config_rebase(inputs.rebase, rebased=rebase_config)
    _echo_engine_change(store, inputs, upgrading=allow_engine_upgrade)

    # --debug assembles the QA recorder in the composition root (prune-on-start → run tree →
    # stamped manifest), echoes the run id + report path here at start, and — when the legacy
    # loop will drive research — marks the funnel uninstrumented so a zero-fill can't read as
    # "nothing happened". Off by default: recorder stays None and the run is byte-identical.
    # It rides the run's OWN id, so the QA tree and the run record name the same run.
    recorder = None
    if debug:
        recorder = build_recorder(settings, argv=sys.argv[1:], mode=mode, run_id=store.run_id)
        typer.echo(f"QA run: {recorder.run_id}")
        typer.echo(f"QA report: {recorder.run_dir}")
        if not client_status(settings).ok:
            recorder.mark_legacy_research()

    # The run store and the recorder wrap the whole run: a between-phases stop
    # (SIGINT/SIGTERM/time limit returns normally through runtime.run()) AND a hard error both
    # reach close() via the finally, so an interrupted run still lands a readable final segment,
    # a stamped manifest, and a closed run record with its lock released.
    stopped_reason = "startup"
    result = None
    try:
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

        # Readiness spans the trading roster (the growing universe): the config seed plus every
        # symbol the research agent has fetched into the lake.
        ready = [s for s in trading_roster(settings, lake) if lake.check_symbol_ready(s)]
        if not ready:
            typer.echo(
                "No catalog data yet — ingest history first (e.g. `noctis data ingest AAPL "
                "--start 2024-01-01 --end 2024-12-31`), then run again. Exiting cleanly."
            )
            stopped_reason = "no_data"
            return

        # One level-aware sink renders the loop's typed events (phase banners + the research feed,
        # and the trading feed once P4 lands) and, under --debug, tees them into the recorder even
        # when the console is absent (quiet --debug records silently). A bare run passes the plain
        # console (or None), so the runtime stays byte-identical to today.
        runtime = build_runtime(
            settings,
            market_lake=lake,
            memory=memory,
            clock=clock,
            mandate=active_mandate,
            on_event=build_event_sink(verbose, show_reasoning=show_reasoning, secondary=recorder),
            # Incremental durability: the record is rewritten at each CLOSE, so a multi-week run's
            # evidence is current on disk long before the process stops.
            on_cycle_close=lambda outcome: store.checkpoint(
                counters=segment_counters(outcome),
                phase_seconds=segment_phase_seconds(outcome),
            ),
            # The run-level cap is measured against the whole run, so this segment starts counting
            # from what its predecessors already spent (story #136).
            prior_runtime_s=store.prior_runtime_s,
        )

        # SIGINT/SIGTERM route through one clean shutdown path (stops between phases).
        def _shutdown(signum, _frame):
            typer.echo(f"\nReceived signal {signum}; stopping cleanly after this phase…")
            runtime.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _shutdown)

        result = runtime.run()
        stopped_reason = result.stopped_reason
        typer.echo(
            f"Stopped ({result.stopped_reason}): {result.cycles_completed} cycle(s), "
            f"{result.research_iterations} candidates researched, {result.trades} paper orders."
        )
    finally:
        if recorder is not None:
            recorder.close()
            typer.echo(f"QA report: {recorder.run_dir} (run {recorder.run_id})")
            typer.echo(f"QA funnel: {recorder.funnel_line()}")
        # Closing the segment stamps its stop reason and duration, writes the record one last
        # time and releases the lock — including on the paths that never reached the loop, so a
        # run that exits early is still a closed, resumable run rather than a dangling lock.
        store.close(
            reason=stopped_reason,
            counters=segment_counters(result) if result is not None else None,
            phase_seconds=segment_phase_seconds(result) if result is not None else None,
        )
        # Said after the segment is closed, because that write is what makes it true: the record
        # derives the seal from its own segments, so the run is completed the moment the segment
        # that spent the cap lands on disk.
        if stopped_reason == "run_limit":
            typer.echo(
                f"Run {store.run_id} reached its {settings.run_limit_hours:g}h compute cap and is "
                "now completed — a terminal state, so it refuses resume. Start a new run to "
                "continue researching."
            )


@app.command()
def status(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Show the resolved execution mode, market state, and a configuration summary.

    Reports the configuration the session actually assembled: ``status`` runs the composition
    root's whole precedence chain (gate → mandate → overlay), not the mode alone, so every line
    below shows the **post-overlay** value a run would use rather than what ``config.yaml`` said
    before the mandate touched it. The mandate, its applied overrides, and the resolved research
    model are stated alongside, so the steering and its effect are readable together.
    """
    from noctis.champions import build_registry
    from noctis.engine import MarketClock, initial_phase_for, resolve_trading_driver
    from noctis.engine.report_assembly import gather_account_forward
    from noctis.research import resolved_research_model

    inputs, mandate_error = _resolve_status_session(config)
    settings, mode = inputs.settings, inputs.mode
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
    # The same resolution the run's own "Research engine" line announces (research.model, else
    # the agent fallback) — read here without the client probe, so the quickest diagnostic in
    # the CLI never pays for (or depends on) the optional LLM stack.
    typer.echo(f"research model:    {resolved_research_model(settings)}")
    typer.echo(f"data provider:     {settings.data.provider} (budget ${settings.data.budget_usd})")
    typer.echo(
        f"trading driver:    {resolve_trading_driver(settings)} "
        f"(execution={settings.trading.execution})"
    )
    _echo_status_mandate(inputs.mandate, inputs.overrides, mandate_error)


@app.command("runs")
def runs(
    show_all: bool = typer.Option(
        False,
        "--all",
        help="Show every run, including the short ones the default listing hides "
        "(a finished run under 60s of cumulative runtime: a startup failure or a typo).",
    ),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """List this workspace's runs: id, label, status, segments and headline numbers.

    The experiment board — how a run is found and compared without opening files. Runs are
    listed newest first, and each line ends with the run's **comparable key**: the tuple two
    runs must match on before their numbers may be pooled or ranked together, so a leaderboard
    is partitioned structurally rather than from memory.

    **The default listing hides noise**: a run that finished with less than 60 seconds of
    cumulative runtime produced no evidence — it is a startup failure, a mistyped command or a
    config typo — and a board full of those buries the experiments. ``--all`` shows everything,
    and the count of what was hidden is always printed. Two kinds are never hidden whatever
    their runtime: a run that is still ``running``, and a run whose record could not be read.

    Rewrites the derived ``index.json`` roll-up from the records on disk as it goes: the index
    is derived, never authoritative, so listing is also how it is regenerated.
    """
    from pathlib import Path

    from noctis.reporting.run_store import rebuild_index, visible_runs, write_index

    settings = load_settings(config_path=config)
    runs_dir = Path(settings.runs_dir)
    index = rebuild_index(runs_dir)
    if runs_dir.is_dir():
        write_index(runs_dir, index)
    if not index["runs"]:
        typer.echo(f"No runs yet under {runs_dir}. Every `noctis run` mints one.")
        return

    listed = visible_runs(index["runs"], include_all=show_all)
    rows = [_run_row(entry) for entry in listed]
    if rows:
        ids = max(len(row[0]) for row in rows)
        labels = max(len(row[1]) for row in rows)
        # The runtime column carries "spent/cap" for a capped run, so it is sized from the rows
        # rather than fixed — a board that truncated a run's compute budget would hide the one
        # number that says whether two experiments were given the same chance.
        times = max(9, max(len(row[4]) for row in rows))
        typer.echo(
            f"{'run':<{ids}}  {'label':<{labels}}  {'status':<11} {'segments':>8} "
            f"{'runtime':>{times}}  comparable key"
        )
        for run_id, label, status, segments, runtime, key in rows:
            typer.echo(
                f"{run_id:<{ids}}  {label:<{labels}}  {status:<11} {segments:>8} "
                f"{runtime:>{times}}  {key}"
            )
    hidden = len(index["runs"]) - len(listed)
    if hidden:
        typer.echo(f"\n{hidden} short run(s) hidden; pass --all to list them.")


def _run_row(entry: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    """One index entry as the six columns the board prints.

    The last column answers "may these numbers be pooled?" with the run's comparable key — or,
    for a run whose record could not be read, says why there is no answer. Either way the line
    explains itself without a second lookup.

    A run that ran more than one engine carries ``mixed engine`` beside its key, because the key
    alone would over-promise: it names the bucket the run's *latest* engine puts it in, while some
    of its numbers were produced by another one. That is exactly the fact a leaderboard must not
    have to be told twice.
    """
    readable = bool(entry.get("readable"))
    segments = entry.get("segments")
    key = str(entry.get("comparable_key") or entry.get("note") or "-")
    return (
        str(entry.get("run_id") or "-"),
        str(entry.get("label") or "-"),
        str(entry.get("status") or "-") if readable else "unreadable",
        "-" if segments is None else str(segments),
        _runtime_against_cap(entry),
        f"{key}  (mixed engine)" if entry.get("mixed_engine") else key,
    )


def _runtime_against_cap(entry: Mapping[str, object]) -> str:
    """Runtime spent, and — for a capped run — the budget it is spending: ``2h00m/100h``.

    The board's answer to "may these two runs be compared?" is not only the engine bucket: two runs
    given different compute are different experiments, so the cap belongs beside the runtime rather
    than one lookup away in the record."""
    spent = _runtime(entry.get("cumulative_runtime_s"))
    limit = entry.get("run_limit_hours")
    if isinstance(limit, bool) or not isinstance(limit, int | float):
        return spent
    return f"{spent}/{limit:g}h"


def _runtime(seconds: object) -> str:
    """Cumulative runtime as an operator reads it — ``45s`` / ``38m`` / ``2h30m`` / ``4d02h``."""
    if not isinstance(seconds, int | float) or isinstance(seconds, bool):
        return "-"
    minutes, hours = int(seconds // 60), int(seconds // 3600)
    if minutes < 1:
        return f"{int(seconds)}s"
    if hours < 1:
        return f"{minutes}m"
    if hours < 24:
        return f"{hours}h{minutes % 60:02d}m"
    return f"{hours // 24}d{hours % 24:02d}h"


@app.command("run-record")
def run_record(
    run_id: str = typer.Argument(
        ...,
        help="Run address: an id as `noctis runs` lists it, a path to a run.json, @LABEL, or "
        "`latest`. The same four forms `run --resume` takes, resolved by the same rules.",
    ),
    validate: bool = typer.Option(
        False,
        "--validate",
        help="Schema-check the record instead of printing it; exits non-zero, naming every "
        "problem, when it is not valid.",
    ),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Print one run's record — the whole self-contained ``run.json``, straight to stdout.

    The record has no sidecars: everything about the run is in this one document, so the output
    is exactly what a website would `fetch()` and is safe to pipe into ``jq``. Exits non-zero
    when no run answers the id, or when its record cannot be read — the listing tolerates a
    broken record because it has others to show; a command asked for exactly one does not.

    ``--validate`` checks the document against the schema contract (``reporting/schema.py``) and
    prints the verdict rather than the record, so an artifact can be **verified before it is
    published**: every section present, every absent value an explicit ``null``, every stamp UTC
    with a ``Z``, every dimensioned number naming its unit. It names every problem at once — an
    operator asking "is this record readable?" wants the whole list, not the first line of it —
    and exits non-zero when there is one.
    """
    import json
    from pathlib import Path

    from noctis.reporting import schema
    from noctis.reporting.run_store import RunNotFoundError, read_record, resolve_run_dir

    settings = load_settings(config_path=config)
    try:
        run_dir = resolve_run_dir(Path(settings.runs_dir), run_id)
    except RunNotFoundError as exc:
        _exit_red(exc)
    record, note = read_record(run_dir)
    if record is None:
        _exit_red(RuntimeError(f"{run_dir / 'run.json'}: {note}"))
    if not validate:
        typer.echo(json.dumps(record, indent=2))
        return
    problems = schema.validate(record)
    path = run_dir / "run.json"
    if not problems:
        typer.echo(
            f"{path}: valid against run-record schema version "
            f"{record.get('schema_version')} ({len(schema.REQUIRED_SECTIONS)} sections checked)."
        )
        return
    typer.echo(f"{path}: {len(problems)} schema problem(s):")
    for problem in problems:
        typer.echo(f"  - {problem}")
    raise typer.Exit(code=1)


@app.command("run-prune")
def run_prune(
    run_id: str = typer.Argument(
        ...,
        help="Run address: an id as `noctis runs` lists it, a path to a run.json, or @LABEL. "
        "(`latest` is the most recent *resumable* run, so it never names a prunable one.)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report what would be removed and how many bytes it would free, and remove nothing.",
    ),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Reclaim disk: delete one **completed** run's ``state/``, ``strategies/`` and ``reports/``.

    Retention is **opt-in and one run at a time** — nothing here runs on a schedule, and no
    configuration setting can make it: a deletion nobody typed is exactly the surprise this design
    refuses. The run's ``run.json`` and the ``index.json`` roll-up are never pruned; they are
    kilobytes, and they *are* the long-term progress history the record exists for, so a pruned run
    still lists in ``noctis runs`` and still prints here in full.

    **Only a ``completed`` run may be pruned.** The pruned directories are what a resume reads back,
    so pruning a ``stopped``, ``interrupted`` or ``running`` run would silently destroy its
    resumability — the one thing the run record promises — and is refused with that reason. Seal a
    run you really are finished with first (``noctis run --resume <address> --finish``);
    ``completed`` is terminal, so nothing that could still be continued is ever at risk. A run
    another engine is live on is refused too, whatever its record says.

    Afterwards the record carries ``state_pruned: true``, so a reader knows the run's path-plus-hash
    references *into* those directories no longer resolve. Everything the record itself carries is
    untouched — pruning removes three directories and rewrites one flag.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    from noctis.reporting.run_store import (
        RunLockedError,
        RunNotFoundError,
        RunNotPrunableError,
        prune_run_state,
    )

    settings = load_settings(config_path=config)
    try:
        outcome = prune_run_state(
            Path(settings.runs_dir),
            run_id,
            clock=lambda: datetime.now(UTC),
            election_metric=settings.promotion.metric,
            dry_run=dry_run,
        )
    except RunNotFoundError as exc:
        _exit_red(exc, prefix="PRUNE: ")
    except RunNotPrunableError as exc:
        _exit_red(exc, prefix="PRUNE REFUSED: ")
    except RunLockedError as exc:
        _exit_red(exc, prefix="RUN LOCKED: ")
    _echo_prune(outcome)


def _echo_prune(outcome) -> None:
    """Say exactly what went (or would go), and what is deliberately still there."""
    listed = ", ".join(f"{name}/" for name in outcome.removed)
    size = f"{outcome.freed_bytes} bytes ({_human_bytes(outcome.freed_bytes)})"
    if not outcome.removed:
        typer.echo(
            f"Run {outcome.run_id} has no state/, strategies/ or reports/ left to prune — "
            "nothing to do. Its record is kept, as always."
        )
        return
    if outcome.dry_run:
        typer.echo(
            f"Would prune {listed} from run {outcome.run_id}, freeing {size}. Nothing was "
            "removed — drop --dry-run to do it."
        )
        return
    typer.echo(
        f"Pruned {listed} from run {outcome.run_id}, freeing {size}. Its run.json and the "
        "index.json roll-up are kept (the record now says state_pruned: true, so references into "
        "the pruned directories no longer resolve)."
    )


def _human_bytes(size_bytes: int) -> str:
    """The byte count as an operator reads it. Bytes stay bytes below a kilobyte — a run that
    freed 400 B should say so rather than round itself to a reassuring `0.0 MB`."""
    for unit, scale in (("MB", 1024**2), ("KB", 1024)):
        if size_bytes >= scale:
            return f"{size_bytes / scale:.1f} {unit}"
    return f"{size_bytes} B"


@app.command()
def engine(
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Print the engine's identity: declared version, per-component digests, comparable key.

    The one-command answer to "are these two runs comparable?". The declared
    ``engine_version`` groups runs; the per-component fingerprint says *what* moved — a gate,
    a prompt, a shipped profile, a seed — and the comparable key is the tuple two runs must
    match on before their champion and scorecard numbers may be pooled or ranked together.
    The ``gates`` and ``backtest`` digests carry that guarantee (they decide what passes and
    what a number means), which is why they are marked, and why the election metric rides in
    the key beside them: numbers under different metrics were never comparable.

    Reads source files and the resolved election metric. Starts nothing, writes nothing.
    """
    from noctis.observability.engine_id import ARBITER_COMPONENTS, comparable_key, fingerprint

    metric = _resolve_session_or_exit(config).settings.promotion.metric
    fp = fingerprint()
    width = max(len(name) for name in fp.components)
    typer.echo(f"engine version:    {fp.engine_version}")
    typer.echo("components:")
    for name, component in fp.components.items():
        arbiter = "  (arbiter — binds comparability)" if name in ARBITER_COMPONENTS else ""
        typer.echo(f"  {name:<{width}}  {component.digest or 'null'}{arbiter}")
        if component.note:
            typer.echo(f"  {'':<{width}}  {component.note}")
    typer.echo(f"election metric:   {metric}")
    typer.echo(f"comparable key:    {comparable_key(metric, fp)}")


@app.command()
def mandate(
    name: str = typer.Argument(
        ...,
        help="Mandate selector — the same vocabulary --mandate takes: a profile name, "
        "MANDATE, or auto.",
    ),
    config: str = typer.Option(None, "--config", "-c", help="Path to config YAML."),
) -> None:
    """Preflight one mandate: what it resolves to, and every setting it would change.

    The dry run before committing a machine to a multi-day loop — the mandate's source,
    summary, declared symbols and references, then the effective settings diff: every path
    the overlay binds, from the value config resolved for it to the value the run would use.
    Starts nothing: no research session, no LLM client, no orders.

    Exits non-zero on any refusal, wrong-direction clamp, invalid value, or unresolvable
    selector — listing every problem at once, exactly as startup would — so a cron job or a
    shell script can gate on it.
    """
    # Straight through the composition root, so a preflight can never drift from what a run
    # assembles: one precedence chain, two renderings. ``require_gate`` stays off for the same
    # reason ``research`` leaves it off — a preflight places no orders — which also keeps a
    # mandate diagnosable on a box whose ``mode`` will not resolve.
    inputs = _resolve_session_or_exit(config, mandate=name)
    _echo_mandate_preflight(inputs.mandate, inputs.changes)


@app.command()
def report(
    run_id: str = typer.Argument(
        None,
        help="Run address: an id as `noctis runs` lists it, a path to a run.json, @LABEL, or "
        "`latest`. The same four forms `run --resume` takes, resolved by the same rules. "
        "Omitted, this reads the reserved `legacy` run, as every unaddressed verb does.",
    ),
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
    """Generate or retrieve the close-of-day report — of the reserved run, or of one you name.

    ``noctis report latest`` is the just-finished run's report, ``noctis report @nightly-momo``
    a named experiment's; with no address this reads the reserved ``legacy`` run exactly as it
    always has, which is what an invocation that never opened a run should read. Everything else
    composes unchanged: ``--as-of`` picks the day, ``--sweep-stale`` sweeps that run's reports.

    An addressed run's own tree is authoritative — its ``reports/`` are read, its ``state/``
    assembles a missing one, and its ``reports/`` is where that one lands. A run whose state
    retention has **pruned** is refused instead: a report assembled from deleted state would
    claim an empty champion board for a run that had one, and the record retention deliberately
    kept still says everything the run accumulated (``noctis run-record <address>``).
    """
    from pathlib import Path

    from noctis.bootstrap import build_memory
    from noctis.champions import build_registry
    from noctis.engine.report_assembly import assemble_report
    from noctis.reporting import latest_report, sweep_stale_reports, today_str, write_report

    settings = load_settings(config_path=config)
    _bind_reported_run_or_exit(settings, run_id)
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


def _bind_reported_run_or_exit(settings, address: str | None) -> None:
    """Point ``report`` at the run it will read — the reserved one, or the one an operator named.

    Two branches, and they are one decision seen twice:

    * **No address** changes nothing and runs the legacy-layout guard, because an unaddressed
      ``report`` is exactly the invocation that guard exists for: it reads the reserved run, and
      un-migrated data beside ``config.yaml`` would make it read as silently empty.
    * **An address** *is* the answer to the guard's question, so the guard does not ask it. The
      operator named the tree, one resolver turns the address into a run dir (the four forms, in
      one fixed order) and the composition root binds ``reports_dir``/``state_dir``/
      ``memory_path`` onto it — after which nothing about a pre-workspace layout elsewhere can
      make a named run unreadable.

    A run whose heavy state ``run-prune`` removed is refused rather than reported on: its
    ``reports/`` are gone and its ``state/`` with them, so the only report left to give would be
    one assembled from nothing — an empty champion board attributed to a run that had one.
    """
    from noctis.bootstrap import bind_addressed_run
    from noctis.reporting.run_store import RunNotFoundError, read_record

    if address is None:
        _guard_legacy_or_exit(settings)
        return
    try:
        run_dir = bind_addressed_run(settings, address)
    except RunNotFoundError as exc:
        _exit_red(exc)
    record, _ = read_record(run_dir)
    if isinstance(record, Mapping) and (record.get("run") or {}).get("state_pruned"):
        _exit_red(
            RuntimeError(
                f"Run {run_dir.name} was pruned: `noctis run-prune` deleted its reports/ and "
                f"state/, so there is no report to print and nothing honest to assemble one "
                f"from. Everything the run accumulated is still in its record — "
                f"`noctis run-record {run_dir.name}`."
            )
        )


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
    resume: str = typer.Option(
        None,
        "--resume",
        help="Append this session to an existing run instead of minting a new one: the same "
        "record, the same frozen config, the same run-scoped state, strategy tiers and memory — "
        "so a research-only night accumulates into the run exactly as a `noctis run` night does. "
        "The same four address forms as `run --resume`, in the same order: a path to a run.json "
        "(or the run dir); @LABEL; the reserved word `latest`; a run id. A bare address is ALWAYS "
        "the id. `noctis runs` lists them. A run's mandate and metric are frozen at creation, so "
        "--directive/--mandate/--metric are refused here rather than silently ignored.",
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
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Record an hour-segmented QA report of this session under qa_dir (funnel counts, "
        "per-strategy fates, phase timing, events.jsonl, a stamped manifest). Records silently — "
        "it never turns on the -v console feed. Off by default.",
    ),
) -> None:
    """Run one agent research session against the current lake, observably.

    The agent formulates (or revises) a strategy in the library, matches symbols, iterates
    backtests/sweeps until the parameter space is exhausted, and reaches a verdict. Needs the
    [llm] extra and an API key for the configured ``research.model`` provider (OPENAI_API_KEY for
    ``openai/*``, ANTHROPIC_API_KEY for ``anthropic/*``; a local backend needs none).

    **The session belongs to a run** (story #137). Without ``--resume`` it mints one, exactly as
    ``noctis run`` does — a standalone night is an experiment, not an unrecorded write into a
    shared default. With ``--resume <address>`` it appends a **research-only segment** to an
    existing run: same lock, same frozen config, same run-scoped state, strategy tiers and memory,
    same record. A run may accumulate many such segments and never trade; it then reports
    ``traded: false`` and a ``null`` performance block, which is a first-class shape, not a
    degenerate one.
    """
    import sys
    import time

    from noctis.bootstrap import (
        build_event_sink,
        build_families,
        build_lake,
        build_memory,
        build_recorder,
        build_research_session,
        open_run_store,
        research_segment_counters,
    )
    from noctis.champions import build_registry
    from noctis.reporting.run_record import RESEARCH_PHASE
    from noctis.reporting.run_store import RunLockedError
    from noctis.research import client_status, provider_of, resolved_research_model

    logging.basicConfig(
        level=_logging_level(verbose), format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    # The composition root owns the ordering (§5): load_settings → resolve_mandate →
    # overlay_mandate → explicit CLI flags, so --metric still wins over a mandate overlay. Under
    # --resume that middle is replaced by the run's frozen config, and the flags that would
    # re-steer it are refused with a reason — the same chain `run --resume` walks, from one place.
    inputs = _resolve_session_or_exit(
        config, directive=directive, mandate=mandate, metric=metric, resume=resume
    )
    settings, active_mandate = inputs.settings, inputs.mandate
    _guard_legacy_or_exit(settings)

    # Open the run BEFORE any collaborator is assembled: opening binds the run's own tree
    # (state, memory, strategy tiers, reports), so everything built below reads the run's state
    # with no path arithmetic here. A live lock is the one fatal failure in this subsystem — two
    # engines writing one run would corrupt it — and it refuses identically whichever verb asked.
    try:
        store = open_run_store(
            settings,
            argv=sys.argv[1:],
            command="research",
            run_id=resume,
            resume=resume is not None,
            mandate=active_mandate,
            overrides=inputs.overrides,
        )
    except RunLockedError as exc:
        _exit_red(exc, prefix="RUN LOCKED: ")
    typer.echo(f"{'Resumed run' if resume else 'Run'}: {store.run_id}")
    typer.echo(f"Run record: {store.record_path}")
    _echo_engine_change(store, inputs, upgrading=False)

    # The run store and the recorder wrap the whole session: an early exit (no key/extra) and a
    # hard error both reach the one finally, so a session that never ran still lands a closed,
    # resumable segment with its lock released — the same shape `run` uses.
    stopped_reason = "startup"
    summary = None
    research_s = 0.0
    recorder = None
    try:
        # --debug opens the QA recorder only once the agent session is known to be buildable
        # (``client_status.ok`` mirrors ``build_research_session is not None``), so an early exit —
        # research never records a legacy session — leaves no orphaned half-written run tree. It
        # rides the run's OWN id, so the QA tree and the run record name the same run.
        if debug and client_status(settings).ok:
            recorder = build_recorder(settings, argv=sys.argv[1:], mode=None, run_id=store.run_id)
            typer.echo(f"QA run: {recorder.run_id}")
            typer.echo(f"QA report: {recorder.run_dir}")

        # One level-aware sink renders the loop's typed events, teed into the recorder under
        # --debug; --show-reasoning opens the think/say streams without the full -vv DEBUG noise.
        # Quiet (no -v, no --debug) ⇒ None ⇒ the loop's own logging default handles events. The
        # command duck-types console.saw_think/hint off this sink — the EventTee proxies both to
        # the primary (or a no-op stand-in when --debug runs without -v), so -v output stays
        # byte-identical.
        console = build_event_sink(verbose, show_reasoning=show_reasoning, secondary=recorder)
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
            stopped_reason = "no_session"
            resolved_model = resolved_research_model(settings)
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
        # Seconds this process spent *working* in RESEARCH — measured around the session itself,
        # monotonically, so a clock step cannot make a night look longer. Startup and waiting are
        # not research: they belong to the segment's wall-clock duration and to no phase, exactly
        # as the day loop's own phase timings work.
        started = time.monotonic()
        summary = session.run(max_iterations=max_iterations)
        research_s = round(time.monotonic() - started, 3)
        stopped_reason = summary.stopped_reason or "agent_done"
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
        coder_calls = (
            f" {session.toolbox.author_calls} coder authoring call(s),"
            if session.toolbox.author_calls
            else ""
        )
        typer.echo(
            f"Session over ({summary.stopped_reason}): {summary.iterations} tool rounds, "
            f"{session.toolbox.backtests_run} backtests,{coder_calls} "
            f"{summary.promotions} promotion(s), {summary.rejections} rejection(s)."
        )
        # What the session spent, and — deliberately — that the dollars are an *estimate* priced
        # from a versioned table rather than a charge anyone made (story #140). A model the table
        # does not carry is named as unpriced: a zero would be a confidently false bill.
        if summary.tokens_total:
            # The version named here is the session's **own** table (the one the run was priced
            # with, config override included), read off the same session that spent the tokens —
            # so the line can never credit a bill to a table that did not produce it.
            table_version = session.price_table.version
            if summary.usd_estimate is None:
                typer.echo(
                    f"Spend: {summary.tokens_total:,} tokens; no price for {session.model!r} in "
                    f"table {table_version} — cost not estimated"
                )
            else:
                typer.echo(
                    f"Spend: {summary.tokens_total:,} tokens ≈ ${summary.usd_estimate:.4f} "
                    f"(estimate, price table {table_version})"
                )
        if summary.candidates:
            typer.echo(f"Strategies worked on: {', '.join(summary.candidates)}")
            typer.echo(
                f"Inspect: noctis strategies; {settings.state_dir}/experiments/<name>.jsonl; "
                f"{settings.memory_path}"
            )
        if summary.undecided:
            typer.echo(
                f"Left undecided ({len(summary.undecided)}): {', '.join(summary.undecided)} "
                f"— archived after the TTL"
            )
    finally:
        if recorder is not None:
            recorder.close()
            typer.echo(f"QA report: {recorder.run_dir} (run {recorder.run_id})")
            typer.echo(f"QA funnel: {recorder.funnel_line()}")
        # Closing the segment stamps its stop reason and duration, writes the record one last time
        # (re-counting the run's journaled trials from disk) and releases the lock — including on
        # the paths that never reached a session, so a run that exits early is still a closed,
        # resumable run rather than a dangling lock. A session that never ran measured no phase
        # and counts nothing: ``None`` leaves both off the segment rather than claiming a zero.
        store.close(
            reason=stopped_reason,
            counters=research_segment_counters(summary) if summary is not None else None,
            phase_seconds={RESEARCH_PHASE: research_s} if summary is not None else None,
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
    """Move a legacy layout to where the engine reads it now, one artifact at a time.

    Two generations, one command. The **pre-workspace** artifacts beside `config.yaml` (state/,
    data_lake/, reports/, MEMORY.md, the strategies/__tmp|champions tiers) move into the
    workspace; the **pre-run-scoped** workspace artifacts (workspace/state/, reports/, memory/,
    qa/ and the two tiers) are adopted into the reserved `legacy` run, so an existing operator's
    champions, account and reports survive and become their first resumable run. The lake never
    moves under a run — vendor data is shared by every run.

    Refuses (with a list) when a destination is already taken or two legacy copies claim it —
    those are two different histories, so resolve by hand and re-run. config.yaml never moves,
    and running twice is a no-op.
    """
    from pathlib import Path

    from noctis.bootstrap import adopt_run_record, execute_migration, plan_migration

    settings = load_settings(config_path=config)
    plan = plan_migration(settings)
    if plan.conflicts:
        lines = "\n".join(f"  {c.legacy}  →  {c.configured}: {c.reason}" for c in plan.conflicts)
        typer.secho(
            f"Refusing to migrate — resolve these by hand first:\n{lines}\nNothing was moved.",
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
    adopted = sum(1 for a in plan.moves if a.configured.is_relative_to(Path(settings.run_dir)))
    if dry_run:
        if adopted:
            typer.echo(
                f"\nwould adopt  {adopted} artifact(s) into run {Path(settings.run_dir).name} "
                f"(a run record would be written to {Path(settings.run_dir) / 'run.json'})"
            )
        typer.echo("\n(dry run — re-run without --dry-run to perform these moves)")
        return
    typer.echo(f"\nMigrated {len(plan.moves)} artifact(s).")
    if adopted:
        record = adopt_run_record(settings, adopted=adopted)
        run_id = Path(settings.run_dir).name
        typer.echo(
            f"Adopted {adopted} of them into run {run_id}"
            + (f" ({record})" if record is not None else " (its record was already there)")
        )


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
