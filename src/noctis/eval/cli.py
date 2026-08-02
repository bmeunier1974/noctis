"""The ``bench`` verbs' bodies — the eval layer's operator tooling, on its own side of the line.

``noctis bench …`` is declared in :mod:`noctis.cli` (a Typer group cannot be registered lazily) but
everything it *does* lives here, one function per verb, so the engine's CLI module carries a name
and an argument list rather than benchmark logic. The engine reaches this module through a single
deferred import that :mod:`noctis.eval.guard` names and enforces the shape of — see that module's
docstring for why one exemption exists and why it is the only one.

Each verb is a thin body of the shape every other Noctis command has: resolve settings, resolve the
artifact, hand the work to a pure function, print. Assembly is never done here — the collaborators
a bench needs come from :mod:`noctis.eval.bootstrap`, this layer's own composition root, exactly as
the engine's commands take theirs from :mod:`noctis.bootstrap`.

``run`` is the verb that can spend, and its shape is the spend discipline made structural: it
**always** preflights, prints the plan it computed, and hands that same plan back as the
acknowledgement the runner requires — so :class:`~noctis.eval.runner.SpendUnacknowledged` is
unreachable through the CLI, and ``--dry-run`` simply stops after the printing, having built no
model client and written nothing at all. Nothing in it names a site: the corpus comes from the
workspace's cases root through the generic provider, and how a case becomes the site's renderer
input is a lookup in the ask table, so a second site runs down the same path.

``corpus`` is the verb that answers a question about the *input* rather than about a run: it loads
every case a site declares through that site's own reader — another lookup, so the coder's
bucket-partitioned corpus and a flat mined one report down one path — and prints what validated,
how the cases stratify, and how the tuning/holdout split falls across them. It spends nothing, asks
nothing, and writes nothing.

The refusals are the interesting part, and they are all **first**:

* a bench id nothing answers refuses *naming the bench root it looked in*, because "not found" is
  only useful beside "here is where I looked";
* a record that cannot be read as a JSON object refuses naming the file, rather than rendering the
  half of it that parsed;
* a record the schema validator has problems with refuses with **every** problem at once — the
  posture ``noctis run-record --validate`` already takes, for the same reason: an operator asking
  "is this artifact readable?" wants the whole list.

Reading a bench **writes nothing**. There is no lock, no index update and no repair of a broken
record: a report is a read, and a read that mutates its subject is how an artifact stops being
evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

import typer

from noctis.config import load_settings
from noctis.eval.bench_report import render_bench_report
from noctis.eval.bootstrap import (
    BenchSeams,
    LiveModelUnavailable,
    build_bench_runner,
    cases_root,
    configs_for,
    corpus_provider,
    live_attempt,
    load_corpus,
    select_split,
    site_vocabulary,
)
from noctis.eval.case_provider import MissingCorpus
from noctis.eval.corpus_report import read_corpus, render_corpus_report
from noctis.eval.record import validate
from noctis.eval.registry import UnknownSite
from noctis.eval.runner import (
    BENCH_RECORD_NAME,
    Attempt,
    AttemptFn,
    AttemptRequest,
    InvalidBenchRecord,
    bench_root,
)

__all__ = ["report_bench", "report_corpus", "run_bench"]

# Every refusal a bench assembly can raise, rendered as one clean line rather than a traceback: an
# undeclared site, an absent or malformed corpus, a knob the site does not accept, an id that would
# not stay inside its own directory, a count below one, and "no model client can be built".
_REFUSALS = (UnknownSite, MissingCorpus, InvalidBenchRecord, LiveModelUnavailable, ValueError)


def run_bench(
    site_id: str,
    *,
    split: str = "all",
    reps: int = 1,
    workers: int = 1,
    dry_run: bool = False,
    model: str | None = None,
    label: str | None = None,
    config: str | None = None,
    seams: BenchSeams | None = None,
) -> None:
    """Preflight one site's corpus, print the plan, and — unless this is a dry run — execute it.

    The order is the story: the plan is computed **first**, always, from the same code path the run
    itself re-derives it from, and it is printed before anything is asked. A dry run stops there
    (no client is built, no directory is made); a live run hands that plan straight back as the
    acknowledgement :meth:`~noctis.eval.runner.BenchRunner.run` refuses to start without.

    ``seams`` is the injection point the suite drives both halves through — a stub attempt
    callable, a scratch registry, a resolved identity. The command body passes none of it.
    """
    settings = load_settings(config_path=config)
    injected = seams if seams is not None else BenchSeams()
    root = cases_root(settings.workspace_dir)
    try:
        selected = select_split(split)
        corpus = load_corpus(site_id, cases_root=root, registry=injected.registry)
        runner = build_bench_runner(
            settings,
            site_id=site_id,
            attempt=_attempt_for(
                settings, site_id=site_id, model=model, dry_run=dry_run, seams=injected
            ),
            workers=workers,
            label=label,
            seams=injected,
        )
        configs = configs_for(model)
        plan = runner.preflight(corpus, reps=reps, configs=configs, split=selected)
        typer.echo(f"BENCH {site_id}{' (dry run)' if dry_run else ''}")
        typer.echo(f"plan: {plan.summary()}")
        typer.echo(f"corpus: {root / site_id} — {len(corpus)} case(s), digest {corpus.digest}")
        typer.echo(f"workers: {runner.executor.workers}")
        if dry_run:
            typer.echo("Nothing was spent and nothing was written — drop --dry-run to run it.")
            return
        run = runner.run(corpus, reps=reps, configs=configs, split=selected, acknowledged=plan)
    except _REFUSALS as refusal:
        _refuse(f"BENCH: {refusal}")
    typer.echo(f"bench: {run.bench_id} → {run.directory}")
    typer.echo(f"record: {run.directory / BENCH_RECORD_NAME}")
    typer.echo(f"complete: {'yes' if run.complete else 'no'}")


def _attempt_for(
    settings: Any, *, site_id: str, model: str | None, dry_run: bool, seams: BenchSeams
) -> AttemptFn:
    """The model call this invocation would make — injected, live, or a dry run's refusal.

    A dry run is handed a callable that refuses to be called rather than a real one that is never
    called: the preflight asks nothing by construction, and the placeholder makes that a fact about
    the object instead of a promise about the control flow. It is also why ``--dry-run`` needs no
    key — :func:`~noctis.eval.bootstrap.live_attempt` is never reached.

    The site is named on the way in because a bench measures exactly one, and a site whose live ask
    is not a forced structured emit (the coder's is an authoring job) declares its own maker in the
    ask table. Which one that is stays a lookup there, so this body names no site.
    """
    if seams.attempt is not None:
        return seams.attempt
    if dry_run:
        return _never_asked
    return live_attempt(
        settings, site_id=site_id, model=model, registry=seams.registry, asks=seams.asks
    )


def _never_asked(request: AttemptRequest) -> Attempt:
    """The dry run's attempt callable: a preflight asks nothing, so this can only be a defect."""
    raise LiveModelUnavailable(
        f"a dry run asked {request.case.case_id!r} — a preflight spends nothing by construction"
    )


def report_corpus(
    site_id: str, *, config: str | None = None, seams: BenchSeams | None = None
) -> None:
    """Validate one site's corpus and print what is in it: the stats, and how it is divided.

    Validation is not a step here, it is the whole first half: every case file is loaded through the
    site's own reader (the lookup in :func:`~noctis.eval.bootstrap.corpus_provider`), so a file that
    no longer parses, a case labelled with an axis nobody declares and a bucket directory nobody
    named are refusals naming the file and the defect — never a count quietly taken over the files
    that happened to load. Only once every case is admitted is anything printed.

    Reading a corpus **writes nothing**: no split is stamped, no file is repaired, no index is
    touched. A corpus is the evidence a benchmark number is computed over, and a reader that edited
    its subject would be the one way a corpus stops being that.
    """
    settings = load_settings(config_path=config)
    injected = seams if seams is not None else BenchSeams()
    root = cases_root(settings.workspace_dir)
    try:
        provider = corpus_provider(site_id, cases_root=root, registry=injected.registry)
        reading = read_corpus(site_id, provider.load(site_id), vocabulary=site_vocabulary(site_id))
    except _REFUSALS as refusal:
        _refuse(f"CORPUS: {refusal}")
    typer.echo(render_corpus_report(reading, source=root / site_id, loader=type(provider).__name__))


def report_bench(bench_id: str, *, config: str | None = None) -> None:
    """Print one bench record's reading: the headline it carries, and the breakdown under it.

    The workspace is resolved exactly as every other command resolves it — through the settings the
    composition root and the entrypoints share — and the bench area hangs off the workspace root
    (``<workspace>/bench/``), run-neutral like the data lake, so a bench is addressed by its own
    minted id and never through a run.
    """
    settings = load_settings(config_path=config)
    root = bench_root(settings.workspace_dir)
    path = root / bench_id / BENCH_RECORD_NAME
    record = _record_or_refuse(path, bench_id=bench_id, root=root)
    problems = validate(record)
    if problems:
        _refuse(
            f"{path}: {len(problems)} schema problem(s), so nothing was rendered:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    typer.echo(render_bench_report(record, source=path))


def _record_or_refuse(path: Path, *, bench_id: str, root: Path) -> Mapping[str, Any]:
    """The record at ``path``, or a clean refusal naming the bench area that was searched."""
    if not path.is_file():
        _refuse(
            f"BENCH: no bench {bench_id!r} in {root} — expected its record at {path}. "
            "`bench report` addresses a bench by the id it was minted with."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _refuse(f"BENCH: {path} could not be read as a bench record: {exc}")
    if not isinstance(document, Mapping):
        _refuse(f"BENCH: {path} is not a bench record — its top level is not an object.")
    return document


def _refuse(message: str) -> NoReturn:
    """Red text on stderr and a non-zero exit — the one mapping every refusal here takes."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)
