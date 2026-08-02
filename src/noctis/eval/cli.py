"""The ``bench`` verbs' bodies — the eval layer's operator tooling, on its own side of the line.

``noctis bench …`` is declared in :mod:`noctis.cli` (a Typer group cannot be registered lazily) but
everything it *does* lives here, one function per verb, so the engine's CLI module carries a name
and an argument list rather than benchmark logic. The engine reaches this module through a single
deferred import that :mod:`noctis.eval.guard` names and enforces the shape of — see that module's
docstring for why one exemption exists and why it is the only one.

Each verb is a thin body of the shape every other Noctis command has: resolve settings, resolve the
artifact, hand the work to a pure function, print. The refusals are the interesting part, and they
are all **first**:

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
from noctis.eval.record import validate
from noctis.eval.runner import BENCH_RECORD_NAME, bench_root

__all__ = ["report_bench"]


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
