"""Run addressing — one string an operator typed becomes one run directory (story #287).

The single place an address is resolved, shared by every verb that names a run (``run --resume``,
``run-record``, ``report``, ``run-prune``), because a form invented twice would eventually mean two
things. Four forms in a **fixed** order — a ``run.json`` path, ``@label``, the reserved word
``latest``, a run id — and a bare address is *always* the id: the meaning of an address may not
depend on what a workspace happens to contain, and where two runs could answer one (a reassigned
label) this refuses with both ids rather than pick one.

It reads records and nothing else: :func:`~noctis.reporting.run_tree.record.read_record` off each
run dir, the record's own ``label`` and stamps, and
:func:`~noctis.reporting.run_record.resume_refusal` for what ``latest`` may hand back. Never the
derived ``index.json`` — resolving an address through a cache would make the answer depend on a
file that may be deleted at any moment — and never the lock or the store, so resolving ``@label``
takes no lock, computes no fingerprint and runs no collector.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from noctis.reporting.run_record import resume_refusal
from noctis.reporting.run_tree.record import RUN_RECORD_NAME, optional_str, read_record


class RunNotFoundError(LookupError):
    """No run answers this address. Raised by :func:`resolve_run_dir`, never by the listing."""


class RunAmbiguousError(RunNotFoundError):
    """More than one run answers this address — a label reassigned to a second run.

    A :class:`RunNotFoundError` by inheritance, so every caller that already refuses an
    unanswerable address refuses this one too, and deliberately its own type: "no run answers
    this" and "too many do" want different words, and only one of them can be fixed by typing an
    id. Never resolved by picking a candidate — an alias that silently chose between two runs
    would eventually append a night's work to the wrong record.
    """


# The reserved address forms. ``latest`` is a word, not a lookup: it means the same thing in every
# workspace, so no run can capture it by being named or labelled that (see :func:`resolve_run_dir`).
LATEST = "latest"
LABEL_SIGIL = "@"

# One sentence, appended to every refusal here: an address that resolved to nothing must always
# say how to find the ones that exist, or an operator's next move is to guess.
FIND_RUNS = "`noctis runs --all` lists every run this workspace has."


def resolve_run_dir(runs_dir: Path | str, address: str) -> Path:
    """Resolve one run **address** to its directory, or raise :class:`RunNotFoundError`.

    The single place an operator-typed address becomes a path, shared by every verb that
    addresses a run (``run-record``, ``--resume``). Four forms, tried in this **fixed** order so
    one string always names one run whatever happens to be on disk:

    1. a **path** — anything containing a separator, or named ``run.json``: the record file you
       are looking at, or the run dir holding it;
    2. ``@<label>`` — the ``@`` is the *label sigil*: it looks the name up as a label first, and
       only falls back to reading it as an id, so an id typed with a leading ``@`` still resolves;
    3. ``latest`` — a **reserved word**, always the most recently active resumable run
       (:func:`_resolve_latest`), never a run that happens to be *named* ``latest`` (address that
       by its path) or *labelled* ``latest`` (address that as ``@latest``);
    4. a **run id** — the identity itself, and the only form that is ever consulted for a bare
       string. A run *labelled* like an id is therefore never reachable without the sigil.

    The rules exist to be boring: the meaning of an address may not depend on what a workspace
    happens to contain, and where two runs could answer one address (a reassigned label) this
    **refuses** with both ids (:class:`RunAmbiguousError`) rather than silently picking one.

    A run dir with no readable ``run.json`` still resolves by id or path. The record is evidence
    *about* the run, and refusing to address a run because its evidence is corrupt would put the
    one case an operator most needs to inspect out of reach. It is skipped by ``latest`` and
    ``@label``, which have nothing to select it *on* — address it by id.
    """
    runs = Path(runs_dir)
    if _is_path_address(address):
        return _resolve_path(address)
    if address.startswith(LABEL_SIGIL):
        return _resolve_label(runs, address[len(LABEL_SIGIL) :])
    if address == LATEST:
        return _resolve_latest(runs)
    by_id = _by_id(runs, address)
    if by_id is not None:
        return by_id
    raise RunNotFoundError(_unknown(runs, address))


def _resolve_latest(runs: Path) -> Path:
    """The most recently active **resumable** run, or a raised :class:`RunNotFoundError`.

    *Most recently active* is read off the record — ``run.last_active_utc``, falling back to
    ``created_utc`` — and never off a filesystem mtime, which lies after a copy, a migration or a
    ``jq`` rewrite. Ties break on the run id (itself a UTC stamp), so the answer is total and
    deterministic rather than dependent on directory order.

    *Resumable* is :func:`~noctis.reporting.run_record.resume_refusal`, the same function the
    resume path itself checks, so ``latest`` can never hand back a run the next line refuses: a
    ``completed`` run is terminal and is skipped. A run whose record cannot be read is skipped
    too — it carries no stamp to be "most recent" by — and stays addressable by its id. A
    ``running`` run is *not* skipped: it is the one an operator most often means, and if another
    engine really is holding it the liveness lock refuses loudly a moment later.
    """
    summaries = _summaries(runs)
    resumable = [summary for summary in summaries if summary.resumable]
    if not resumable:
        raise RunNotFoundError(
            f"`--resume latest` found no resumable run under {runs}: {_shortfall(summaries)}. "
            f"{FIND_RUNS}"
        )
    return max(resumable, key=lambda summary: (summary.last_active, summary.run_id)).run_dir


def _resolve_label(runs: Path, label: str) -> Path:
    """One human alias, resolved off the records — or refused, never guessed.

    Exactly one run carrying the label resolves. **Two or more refuse**
    (:class:`RunAmbiguousError`, naming both ids): a label may be reassigned, the id is the
    identity, and choosing between two runs on an operator's behalf is how a night's work lands
    on the wrong record. None falls back to reading the name as an id, so an id typed with a
    leading ``@`` — the shape a habit or a copy-paste produces — still names its run instead of
    failing on punctuation.
    """
    matches = [summary for summary in _summaries(runs) if summary.label == label]
    if len(matches) == 1:
        return matches[0].run_dir
    if len(matches) > 1:
        named = ", ".join(summary.run_id for summary in matches)
        raise RunAmbiguousError(
            f"{len(matches)} runs are labelled {label!r}: {named}. A label is convenience — the "
            f"id is the identity, and it may be reassigned — so this refuses rather than pick "
            f"one for you. Address the run you mean by its id."
        )
    by_id = _by_id(runs, label)
    if by_id is not None:
        return by_id
    raise RunNotFoundError(f"no run labelled {label!r} under {runs}. {FIND_RUNS}")


def _shortfall(summaries: Sequence[_RunSummary]) -> str:
    """Why ``latest`` found nothing — in the operator's terms, never a bare "not found"."""
    if not summaries:
        return "there are no runs here yet, and every `noctis run` mints one"
    completed = sum(1 for summary in summaries if summary.readable and not summary.resumable)
    unreadable = sum(1 for summary in summaries if not summary.readable)
    counted = [
        f"{completed} completed (terminal, so they refuse resume)" if completed else "",
        f"{unreadable} with no readable record (address one by its id)" if unreadable else "",
    ]
    return f"of {len(summaries)} run(s): " + ", ".join(part for part in counted if part)


def read_run_record(runs_dir: Path | str, address: str) -> dict:
    """One addressed run's record, or a raised error — the read a **resume** starts from.

    Where :func:`read_record` reports "no readable record" as a value (a listing must survive one
    broken file), this raises: a resume that cannot read the record has nothing to resume *under*,
    and continuing would silently research under the current ``config.yaml`` instead of the run's
    own frozen one — the exact substitution config freezing exists to prevent.
    """
    run_dir = resolve_run_dir(runs_dir, address)
    record, reason = read_record(run_dir)
    if record is None:
        raise RunNotFoundError(
            f"run {address} has {reason}, so there is no frozen configuration to resume it under. "
            f"`noctis run-record {address} --validate` says what is wrong with it."
        )
    return record


def _by_id(runs: Path, name: str) -> Path | None:
    """The run this name **identifies**, or ``None``. The id form, and ``@label``'s fallback.

    Never joins a path form onto ``runs``: an address that could be a path is one, so ``../..``
    can never address its way out of the run tree through here.
    """
    candidate = runs / name if _is_run_id(name) else None
    return candidate if candidate is not None and candidate.is_dir() else None


def _is_run_id(address: str) -> bool:
    """A bare directory name — the identity form, and the complement of the path form."""
    return bool(address) and not _is_path_address(address)


def _is_path_address(address: str) -> bool:
    """Whether this address is a **path** rather than a name a lookup could answer.

    Anything carrying a separator, plus the bare record name (``run.json`` in the directory you
    are standing in) and the two directory names that are pure navigation. A run id can contain
    none of those, so the two forms cannot collide.
    """
    return "/" in address or "\\" in address or address in (".", "..", RUN_RECORD_NAME)


def _resolve_path(address: str) -> Path:
    """A ``run.json`` path (or the dir holding one) as the run it belongs to.

    Honoured wherever it points, including outside the configured ``runs_dir``: a path is an
    address an operator typed deliberately — a record copied off a server, a second workspace —
    and second-guessing it would defeat the one form whose whole purpose is "this file, here".
    Expanded and made absolute, never ``resolve()``d, so a symlinked workspace still answers as
    the operator addressed it.
    """
    path = Path(os.path.abspath(Path(address).expanduser()))
    if path.is_dir():
        return path
    if path.name == RUN_RECORD_NAME and path.is_file():
        return path.parent
    raise RunNotFoundError(
        f"no run at {path} — the path form addresses a {RUN_RECORD_NAME} file or the run "
        f"directory holding it. {FIND_RUNS}"
    )


def _unknown(runs: Path, address: str) -> str:
    """The refusal an address nobody answers gets: what was looked for, and how to find a run.

    A bare string is always read as an id, so the one near-miss worth naming is a *label* typed
    without its sigil — the operator is one character from the run they meant, and the message
    is the only place that can say so.
    """
    labelled = [summary for summary in _summaries(runs) if summary.label == address]
    hint = (
        f" {len(labelled)} run(s) are labelled {address!r} — a bare address is always the id, so "
        f"write `{LABEL_SIGIL}{address}` to address a run by its label."
        if labelled
        else ""
    )
    return f"no run {address!r} under {runs}.{hint} {FIND_RUNS}"


@dataclass(frozen=True)
class _RunSummary:
    """One run as *addressing* sees it: where it is, what it is called, when it was last active.

    Read from the record on disk, never from ``index.json``. The index is derived and may be
    deleted at any moment, so resolving an address through it would make an answer depend on a
    cache; the label lives in the record because the record is the source of truth.
    """

    run_dir: Path
    run_id: str
    label: str | None
    last_active: str
    resumable: bool
    readable: bool


def _summaries(runs: Path) -> list[_RunSummary]:
    """Every run under ``runs``, summarized for addressing. Sorted by id, so ordering is total."""
    directories = sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
    return [_summary_of(run_dir) for run_dir in directories]


def _summary_of(run_dir: Path) -> _RunSummary:
    record, _ = read_record(run_dir)
    run = record.get("run") if isinstance(record, dict) else None
    if not isinstance(run, Mapping):
        return _RunSummary(run_dir, run_dir.name, None, "", resumable=False, readable=False)
    return _RunSummary(
        run_dir=run_dir,
        run_id=str(run.get("run_id") or run_dir.name),
        label=optional_str(run.get("label")),
        last_active=str(run.get("last_active_utc") or run.get("created_utc") or ""),
        resumable=resume_refusal(record or {}) is None,
        readable=True,
    )
