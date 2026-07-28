"""The run store — the ONE module that touches the run tree (story #129, epic #126).

A run is a real, addressable, always-on entity: ``noctis run`` mints a fresh id (never derives one
from the config — two byte-identical configs are two runs) and the run gets its own tree::

    workspace/runs/
      index.json                ← the DERIVED listing roll-up (story #130)
      <run_id>/
        run.json    ← THE record
        run.lock    ← liveness lock (pid, hostname_hash, started, heartbeat)

**``run.json`` has no sidecars.** One ``fetch()`` of one URL returns everything a run page needs,
so a website needs no server-side logic; ``index.json`` beside it serves the *listing* page in one
more fetch. The index is **derived, never authoritative**: :func:`rebuild_index` regenerates it
from the records on disk at any moment, and a test pins that a rebuild reproduces the
incrementally-maintained file byte for byte. Anything that could only be learned from the index
would be a second source of truth, free to drift from the records it summarizes.

Everything in this module is I/O; everything about the record's *shape* is next door in
``run_record`` (pure) and ``schema`` (pure). :func:`collect` does every read and returns a
:class:`~noctis.reporting.run_record.RunArtifacts`; :func:`write` does the one write. That
boundary is what makes the golden-record and segmentation-equivalence tests cheap — they build a
``RunArtifacts`` in memory and never go near a disk.

**The lock is the one fatal failure in the whole epic.** Everything else here is latched (below),
because a reporting artifact must never take down a multi-week run. A lock is different in kind:
two engines writing one run's record — and, once story #131 moves state under the run, one champion
registry and one paper account — is *corruption*, not degradation. So a live lock is a hard,
informative refusal, never a silent downgrade to "write anyway". A **stale** lock is stealable,
because a crashed run must not need manual cleanup before it can be resumed: stale means a dead pid
on *this* host (a pid on another host tells you nothing about whether it is alive) or a heartbeat
gone colder than :data:`STALE_HEARTBEAT_S`. A steal is loud — one warning and an event in the
record — because it is the one moment a run's history could be attributed to the wrong process.

**The fail-safe latch**, straight from ``observability/debug/recorder.py``: the first internal
exception logs exactly one warning, disables the store, and every later call is a no-op — no retry,
no second warning, nothing raised into the engine. A latched store never gets to write its final
``complete: true``, so the record left on disk says ``complete: false`` and a partial record can
never pass for a whole one.

**Writes are synchronous and atomic.** No background thread (the engine spent four PRs removing
shutdown join hazards; a writer thread would put one back), a temp file plus ``os.replace`` so a
kill mid-write leaves the previous record intact, and an injected clock so no ``datetime.now()`` is
ever reached from here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from noctis.reporting.run_record import (
    EngineIdentity,
    RecordEvent,
    RunArtifacts,
    SegmentArtifact,
    build,
    mark_interrupted,
    prune_refusal,
    resume_refusal,
    seal,
    utc_iso,
)
from noctis.reporting.schema import SCHEMA_VERSION

__all__ = [
    "PRUNED_SUBDIRS",
    "RUNS_SUBDIR",
    "RUN_INDEX_KIND",
    "RUN_INDEX_NAME",
    "RUN_LOCK_NAME",
    "RUN_RECORD_NAME",
    "SHORT_RUN_S",
    "STALE_HEARTBEAT_S",
    "FinishOutcome",
    "PruneOutcome",
    "RunAmbiguousError",
    "RunCompletedError",
    "RunLockedError",
    "RunNotFoundError",
    "RunNotPrunableError",
    "RunStore",
    "assert_resumable",
    "collect",
    "finish_run",
    "index_entry",
    "open_run",
    "prune_run_state",
    "read_record",
    "read_run_record",
    "read_trials",
    "rebuild_index",
    "resolve_run_dir",
    "update_index",
    "visible_runs",
    "write",
    "write_index",
]

logger = logging.getLogger(__name__)

# The run tree's names — one place, so nothing spells them by hand.
RUNS_SUBDIR = "runs"
RUN_RECORD_NAME = "run.json"
RUN_LOCK_NAME = "run.lock"
RUN_INDEX_NAME = "index.json"

# The index's self-declared type, so a consumer can tell the roll-up from a run record at a glance.
RUN_INDEX_KIND = "noctis.run-index"

# What the default listing calls noise: a finished run that never accumulated a minute of runtime
# is a startup failure or a mistyped command, not an experiment. ``--all`` shows them.
SHORT_RUN_S = 60.0

# How cold a heartbeat must be before a lock we cannot otherwise check counts as abandoned.
# Deliberately generous — a week. The heartbeat is touched at each CLOSE, i.e. roughly once per
# trading day, and a live engine can sit in RESEARCH right through a long weekend; anything
# tighter would steal the lock from a running process, which is the exact corruption the refusal
# exists to prevent. The same-host dead-pid check is what catches the common crash promptly.
STALE_HEARTBEAT_S = 7 * 24 * 3600.0

# The only directories retention may ever remove (story #138), named here **as literal children of
# one run dir** — this list is the entire blast radius, and it is a constant so that reviewing it is
# reviewing the whole destructive surface. They are the heavy, re-derivable ones; ``run.json`` and
# ``index.json`` are never pruned (they are small, and they *are* the long-term progress history),
# and neither is anything else the tree happens to hold — the run's ``memory/`` and its ``qa/`` area
# (which has retention of its own) are left exactly where they are.
PRUNED_SUBDIRS = ("state", "strategies", "reports")


class RunLockedError(RuntimeError):
    """Another engine holds this run. The one hard refusal — see the module docstring."""


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


class RunCompletedError(RuntimeError):
    """A resume addressed a run that is ``completed`` — terminal, so it gains no more segments."""


class RunNotPrunableError(RuntimeError):
    """Retention addressed a run that could still be resumed, so its state was **not** deleted.

    The exact twin of :class:`RunCompletedError`, one gate down: that one refuses to continue a run
    that is finished, this one refuses to delete the state of a run that is not. Between them a run
    is either resumable or prunable and never both (:data:`~noctis.reporting.run_record.
    PRUNABLE_STATUSES` *is* ``TERMINAL_STATUSES``), which is what makes "a pruned run that later
    resumed" unreachable rather than merely unlikely.
    """


def open_run(
    runs_dir: Path | str,
    *,
    clock: Callable[[], datetime],
    argv: Sequence[str],
    election_metric: str,
    run_id: str | None = None,
    command: str = "run",
    label: str | None = None,
    engine_root: Path | None = None,
    writer: Callable[[Path, dict], None] | None = None,
    stale_after_s: float = STALE_HEARTBEAT_S,
    resume: bool = False,
    inputs: Mapping[str, object] | None = None,
    rebase_config: bool = False,
    engine_upgrade: Mapping[str, object] | None = None,
) -> RunStore:
    """Open a run for this process: mint or address it, lock it, append a segment, write.

    ``run_id`` defaults to a freshly minted id (identity is minted, never derived); passing one
    addresses that run — which is how a later invocation appends its own segment to the same
    record. Under ``resume`` it is a full **address** (:func:`resolve_run_dir`: an id, ``latest``,
    a ``run.json`` path or ``@label``) and the run's own id is taken from what it resolved to, so
    nothing an alias reached is ever locked or recorded under the alias. Raises
    :class:`RunLockedError` when another engine holds the run.

    ``resume=True`` says the caller means to *continue* an existing run rather than create one, so
    the two failures that are silent under creation become loud: an address with no run tree raises
    :class:`RunNotFoundError` (creating a run under an id an operator typed would answer the wrong
    question), and a ``completed`` run raises :class:`RunCompletedError`. Both are checked before
    the lock is taken, so a refused resume leaves nothing behind.

    ``inputs`` is this process's frozen configuration, and it is used **only when the run has none
    yet** — freezing happens once, at creation. Every later segment carries the record's own inputs
    forward untouched, which is what makes "the current ``config.yaml`` is ignored" true of the
    artifact and not just of the rehydration path.

    ``rebase_config`` is the one deliberate exception (story #134): the operator asked to adopt the
    current configuration, so ``inputs`` **replaces** what the record carried. It arrives already
    re-frozen — epoch bumped, before/after entry appended — because what a config change *is* stays
    in ``config.rehydrate``; this only decides that a rebased block wins over a carried one.

    ``engine_upgrade`` is the same deal one layer down (story #135): the ``engine_changes`` entry a
    deliberately accepted engine change produced, built by ``observability.engine_change``. Given
    one, the run is **re-frozen onto this process's engine** with the entry appended and the epoch
    it names — so a run whose arbiter moved mid-flight says so, and says where. Absent (the normal
    case) the run keeps the engine it was created under, whatever this process is.
    """
    from noctis.observability.debug import new_run_id

    now = clock()
    if resume:
        if run_id is None:
            raise RunNotFoundError("a resume needs a run id — there is nothing to continue without")
        run_dir = resolve_run_dir(runs_dir, run_id)
        # The address is how the run was *reached*; its directory name is what it *is*. Taking the
        # id from the resolution is what keeps `latest`, a path and `@label` from ever writing an
        # alias into the lock or the refusal messages.
        resolved_id = run_dir.name
        _assert_resumable(run_dir, resolved_id)
    else:
        resolved_id = run_id or new_run_id(now)
        run_dir = Path(runs_dir) / resolved_id
    run_dir.mkdir(parents=True, exist_ok=True)

    steal_note = acquire_lock(run_dir, run_id=resolved_id, now=now, stale_after_s=stale_after_s)
    artifacts = mark_interrupted(
        collect(run_dir, election_metric=election_metric, engine_root=engine_root)
    )
    return RunStore(
        run_dir,
        artifacts=artifacts,
        clock=clock,
        argv=argv,
        command=command,
        label=label if label is not None else artifacts.label,
        writer=writer or write,
        opening_note=steal_note,
        inputs=inputs,
        rebase_config=rebase_config,
        engine_upgrade=engine_upgrade,
    )


def assert_resumable(record: Mapping[str, object], run_id: str) -> None:
    """Refuse a resume this record rules out, naming the run.

    The refusal is derived from the record (:func:`~noctis.reporting.run_record.resume_refusal`),
    never re-decided here, and both callers share this one function: the composition root checks it
    the moment it reads the record — so an operator is told before a single line of kickoff banner
    — and :func:`open_run` checks it again as the last thing between an address and a new segment,
    so no future caller can reach the append without passing it.
    """
    refusal = resume_refusal(record)
    if refusal is not None:
        raise RunCompletedError(f"cannot resume {run_id}: {refusal}")


@dataclass(frozen=True)
class FinishOutcome:
    """What ``--finish`` did: which run, whether this call is what sealed it, and when.

    ``sealed=False`` is the **documented no-op** — the run was already ``completed``, so the stamp
    reported is the original one. Terminal means terminal: re-finishing must not rewrite the moment
    a published result was sealed, and a caller that wants to say "nothing to do" needs to be told.
    """

    run_id: str
    sealed: bool
    completed_utc: str | None


def finish_run(
    runs_dir: Path | str,
    address: str,
    *,
    clock: Callable[[], datetime],
    election_metric: str,
    engine_root: Path | None = None,
    writer: Callable[[Path, dict], None] | None = None,
    stale_after_s: float = STALE_HEARTBEAT_S,
) -> FinishOutcome:
    """Seal one addressed run as ``completed`` — terminal — **without running a segment**.

    The deliberate half of story #136 (the run-level cap is the derived half). ``completed`` is the
    one status a run never leaves, so this is how an operator says "this result is published":
    afterwards every resume is refused, and the numbers quoted from the record can never turn out to
    have been provisional.

    It opens nothing. No segment is appended, no engine starts, and the liveness lock is **read, not
    taken** — a run another process is actively working may not be sealed from underneath it
    (:class:`RunLockedError`), but sealing an idle run leaves no lock behind for the next invocation
    to trip over. One atomic rewrite of ``run.json`` (through the same builder every other write
    uses, so a sealed record cannot drift from a written one), then the derived index.

    An unclosed segment left by a kill is marked ``interrupted`` on the way past, exactly as an open
    would: the observation is honest at this moment, and it is the last moment anyone will make it.
    """
    now = clock()
    run_dir = resolve_run_dir(runs_dir, address)
    run_id = run_dir.name
    record, reason = read_record(run_dir)
    if record is None:
        raise RunNotFoundError(
            f"run {address} has {reason}, so there is nothing to seal. `noctis run-record "
            f"{address}` shows what is there."
        )
    _assert_unlocked(
        run_dir,
        run_id=run_id,
        now=now,
        stale_after_s=stale_after_s,
        consequence=(
            "so it cannot be sealed from underneath it: the segment it is running would land on a "
            "record that already says the run is finished. Stop that engine first, then finish the "
            "run."
        ),
    )
    if resume_refusal(record) is not None:  # already terminal — the documented no-op
        run = record.get("run")
        stamp = run.get("completed_utc") if isinstance(run, Mapping) else None
        return FinishOutcome(run_id=run_id, sealed=False, completed_utc=_optional_str(stamp))

    artifacts = mark_interrupted(
        collect(run_dir, election_metric=election_metric, engine_root=engine_root)
    )
    stamp = utc_iso(now)
    (writer or write)(run_dir, build(seal(artifacts, at=stamp)))
    update_index(run_dir.parent, run_id)
    return FinishOutcome(run_id=run_id, sealed=True, completed_utc=stamp)


def _assert_unlocked(
    run_dir: Path, *, run_id: str, now: datetime, stale_after_s: float, consequence: str
) -> None:
    """Refuse when another engine is live on this run — the read-only half of :func:`acquire_lock`.

    Deliberately does not *take* the lock: sealing (and pruning) is one write, and a lock file left
    behind by a command that started nothing would be friction the next invocation has to reason
    about. A stale lock (a dead pid here, a heartbeat gone cold) is no obstacle at all, for the same
    reason a resume may steal one — a crashed run must never need manual cleanup.

    ``consequence`` is the caller's half of the sentence: the holder is named identically whatever
    was asked for, and what would have gone wrong differs.
    """
    held = _read_lock(run_dir / RUN_LOCK_NAME)
    if held is None or _stale_reason(held, now=now, stale_after_s=stale_after_s) is not None:
        return
    raise RunLockedError(
        f"run {run_id} is open by pid {held.get('pid')} on host {held.get('hostname_hash')} "
        f"(heartbeat {held.get('heartbeat_utc')}), {consequence}"
    )


@dataclass(frozen=True)
class PruneOutcome:
    """What retention did (or, under ``dry_run``, what it would do) to one run's tree.

    ``removed`` names the directories, never paths, because the names *are* the policy
    (:data:`PRUNED_SUBDIRS`); ``freed_bytes`` carries its unit in its name, the record schema's own
    convention. Both are populated identically for a dry run — that is what makes the preview worth
    trusting: it is the same measurement of the same targets, taken one line before the removal it
    then does not perform.
    """

    run_id: str
    dry_run: bool
    removed: tuple[str, ...]
    freed_bytes: int


def prune_run_state(
    runs_dir: Path | str,
    address: str,
    *,
    clock: Callable[[], datetime],
    election_metric: str,
    dry_run: bool = False,
    engine_root: Path | None = None,
    writer: Callable[[Path, dict], None] | None = None,
    stale_after_s: float = STALE_HEARTBEAT_S,
) -> PruneOutcome:
    """Delete one **completed** run's heavy directories, keeping its record (story #138).

    Retention in this system is opt-in, one addressed run at a time, and it is the only code in
    Noctis that removes a run's own files — so every gate it passes through is here, in order, and
    nothing runs it on a schedule:

    1. the address must resolve to a run tree carrying a **readable record**. That is what stops a
       path address (which is honoured wherever it points, by design) from ever aiming this at an
       arbitrary directory: no ``run.json``, nothing deleted.
    2. the record must say the run may never gain another segment
       (:func:`~noctis.reporting.run_record.prune_refusal`). ``stopped``, ``interrupted`` and
       ``running`` all refuse, because :data:`PRUNED_SUBDIRS` is exactly what a resume would read
       back — deleting it would silently destroy the resumability this whole design promises.
       Checked before the lock, so a *crashed* run (whose record still says ``running`` while its
       lock has gone stale and stealable) is refused on the status that matters rather than let
       through on a lock nobody holds.
    3. no other engine may be live on the run. A stale lock is no obstacle — as everywhere else,
       a crashed run must never need manual cleanup — but a live one is: the directories this
       removes are the ones that engine is reading and writing.
    4. only the three named children of *this* run dir are touched, and only when each is a real
       directory that is not a symlink — retention never follows a link out of the run tree.

    The record is **collected before anything is deleted** and rewritten afterwards. That ordering
    is load-bearing: the run's trial count is derived from ``state/experiments/*.jsonl`` at write
    time, so rewriting after the removal would replace a run's own history with ``null`` — the
    precise opposite of what pruning is for. The rewritten record carries ``state_pruned: true`` and
    one event saying what went and when; a reader then knows the run's path-plus-hash references
    into those directories no longer resolve, while everything the record *embeds* is untouched.

    ``dry_run=True`` measures and reports, and writes nothing at all — not the marker, not the
    index, not a byte.
    """
    now = clock()
    run_dir = resolve_run_dir(runs_dir, address)
    run_id = run_dir.name
    record, reason = read_record(run_dir)
    if record is None:
        raise RunNotFoundError(
            f"run {address} has {reason}, so nothing here can be pruned: retention deletes a run's "
            f"state only when its own record says the run is completed. Nothing was removed."
        )
    refusal = prune_refusal(record)
    if refusal is not None:
        raise RunNotPrunableError(f"cannot prune {run_id}: {refusal} Nothing was removed.")
    _assert_unlocked(
        run_dir,
        run_id=run_id,
        now=now,
        stale_after_s=stale_after_s,
        consequence=(
            "so its state cannot be deleted from underneath it: the directories this would remove "
            "are the ones that engine is reading and writing. Stop that engine first, then prune "
            "the run. Nothing was removed."
        ),
    )

    targets = _prunable_dirs(run_dir)
    freed = sum(_dir_bytes(target) for target in targets)
    names = tuple(target.name for target in targets)
    if dry_run:
        return PruneOutcome(run_id=run_id, dry_run=True, removed=names, freed_bytes=freed)

    # Read the whole record BEFORE the removal — see the docstring: the trial count is counted off
    # the very journals this is about to delete.
    artifacts = mark_interrupted(
        collect(run_dir, election_metric=election_metric, engine_root=engine_root)
    )
    for target in targets:
        shutil.rmtree(target)
    pruned = replace(
        artifacts,
        state_pruned=True,
        events=(*artifacts.events, _prune_event(now, names=names, freed_bytes=freed)),
    )
    (writer or write)(run_dir, build(pruned))
    update_index(run_dir.parent, run_id)
    return PruneOutcome(run_id=run_id, dry_run=False, removed=names, freed_bytes=freed)


def _prune_event(now: datetime, *, names: Sequence[str], freed_bytes: int) -> RecordEvent:
    """The note a prune leaves on the record: what went, and when it went.

    The marker says the state is gone; this says when, and how much — so a record read a year later
    can tell "pruned last Tuesday" from "this run never wrote any state at all".
    """
    listed = ", ".join(f"{name}/" for name in names) if names else "nothing (already pruned)"
    return RecordEvent(
        t=utc_iso(now),
        kind="info",
        text=f"retention pruned this completed run's {listed} ({freed_bytes} bytes freed); "
        f"{RUN_RECORD_NAME} is kept, and references into the pruned directories no longer resolve",
    )


def _prunable_dirs(run_dir: Path) -> list[Path]:
    """The directories this run may lose — the *whole* destructive surface, computed one way.

    Only the constant names in :data:`PRUNED_SUBDIRS`, joined to this run dir (they carry no
    separator, so nothing an address contained can traverse out through here), only when the child
    is a real directory, and never through a symlink: a linked ``state/`` points somewhere the
    operator chose, and following it would delete a tree outside the run entirely.
    """
    return [
        child
        for child in (run_dir / name for name in PRUNED_SUBDIRS)
        if child.is_dir() and not child.is_symlink()
    ]


def _dir_bytes(path: Path) -> int:
    """How much this directory holds, in bytes — the number a dry run reports.

    Never follows a symlink (neither into a linked subdirectory nor through a linked file), so the
    figure describes exactly what removal would free and nothing that lives elsewhere. Unreadable
    entries are skipped: a byte count is information, never a reason to fail.
    """
    total = 0
    for parent, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            entry = Path(parent) / name
            try:
                if not entry.is_symlink():
                    total += entry.stat().st_size
            except OSError:  # pragma: no cover - a file that vanished mid-count
                continue
    return total


def _assert_resumable(run_dir: Path, run_id: str) -> None:
    """The on-disk half of the check. A run whose record cannot be read at all is *not* refused —
    a corrupt reporting file must never strand the run it describes, and the opening path already
    degrades it to a fresh record carrying an event that says so."""
    record, _ = read_record(run_dir)
    if record is not None:
        assert_resumable(record, run_id)


def collect(
    run_dir: Path | str,
    *,
    election_metric: str,
    engine_root: Path | None = None,
) -> RunArtifacts:
    """Read everything the run tree and this engine know about the run. **All the I/O lives here.**

    Returns the artifacts of an existing record (segments, stamps, label, events) or, for a run
    dir with no record yet, an empty run with this engine's identity. An unreadable record is not
    fatal: it degrades to a fresh record carrying an event that says so, because a corrupt
    reporting file must never stop a run from starting.
    """
    path = Path(run_dir) / RUN_RECORD_NAME
    engine = read_engine_identity(election_metric, root=engine_root)
    trials = read_trials(run_dir)
    prior, note = _read_record(path)
    if prior is not None:
        try:
            return _artifacts_from(prior, run_dir=Path(run_dir), current=engine, trials=trials)
        except Exception as exc:  # a hand-edited or foreign file, still valid JSON
            note = RecordEvent(
                t=None,
                kind="warn",
                text=f"this run had an unreadable {RUN_RECORD_NAME} "
                f"({type(exc).__name__}); a fresh record was started in its place",
            )
    return RunArtifacts(
        run_id=Path(run_dir).name,
        created_utc=None,
        last_active_utc=None,
        engine=engine,
        current_engine=engine,
        events=(note,) if note is not None else (),
        trials=trials,
    )


def read_trials(run_dir: Path | str) -> int | None:
    """How many trials this run has journaled, or ``None`` when it has journaled nothing.

    **Read, never counted.** The number comes from the run's own experiment journals — the very
    lines the exhaustion gate counts (``<run>/state/experiments/<name>.jsonl``) — so the record and
    the research discipline can never disagree about how much searching a run did, and no counter
    has to survive a restart to be right. It is therefore cumulative across every segment by
    construction, including the research-only ones ``noctis research --resume`` appends (story
    #137): the journals are the run's, not the process's.

    Both imports are deferred: the run store is written on the core install alone and must stay
    importable without pulling the research package (or the settings model) in behind it. The
    journal owns the record schema end-to-end, so nothing here parses an ``event`` string, and the
    state directory is derived by the one function that owns that derivation.

    Never raises: an unreadable journal is missing evidence, not a reason to fail a run's write.
    """
    from noctis.config.settings import run_scoped_paths
    from noctis.research.journal import ExperimentJournal

    try:
        state_dir = run_scoped_paths(Path(run_dir))["state_dir"]
        totals = ExperimentJournal(state_dir).totals()
    except Exception:  # pragma: no cover - a journal we cannot read is evidence we do not have
        return None
    return None if totals is None else totals.n_trials


def _artifacts_from(
    prior: Mapping[str, object],
    *,
    run_dir: Path,
    current: EngineIdentity,
    trials: int | None = None,
) -> RunArtifacts:
    """One prior record, parsed back into artifacts. Raises on a shape it cannot read.

    ``trials`` is deliberately **not** read back off the record: it is derived from the journals at
    every write (:func:`read_trials`), and a number carried forward from a prior write could only
    ever go stale.
    """
    run = prior.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("the 'run' section is missing or is not an object")
    return RunArtifacts(
        run_id=str(run.get("run_id") or run_dir.name),
        created_utc=_optional_str(run.get("created_utc")),
        last_active_utc=_optional_str(run.get("last_active_utc")),
        # Frozen at creation and carried forward verbatim, exactly like ``inputs``: the engine a
        # run was created under is the side every later resume is compared against (story #135),
        # so a write must never restamp it with whatever engine happens to be running now.
        engine=_frozen_engine(prior.get("engine")) or current,
        current_engine=current,
        segments=tuple(_segment_from(raw) for raw in _listed(prior, "segments")),
        label=_optional_str(run.get("label")),
        completed_utc=_optional_str(run.get("completed_utc")),
        complete=bool(run.get("complete", False)),
        events=tuple(_event_from(raw) for raw in _listed(prior, "events")),
        errors=tuple(_event_from(raw) for raw in _listed(prior, "errors")),
        # Read straight back and carried forward verbatim: the run's configuration was frozen at
        # creation, so every later segment restores it rather than re-deriving it from files that
        # may have changed in between.
        inputs=_frozen_inputs(prior.get("inputs")),
        trials=trials,
        # Carried forward verbatim, never re-derived from what is on disk: it states that the heavy
        # directories were deliberately removed, and a later write must not un-say it because
        # something recreated an empty ``state/``.
        state_pruned=bool(run.get("state_pruned", False)),
    )


def _frozen_engine(engine: object) -> EngineIdentity | None:
    """The engine identity a record froze at creation, or ``None`` when it carries none readable.

    Tolerant on purpose: a record from before engine epochs (or one a hand-edit mangled) still
    hands back everything it does have, and a section that cannot be read at all degrades to this
    process's own identity rather than stranding the run. The digests are taken **verbatim**, not
    re-validated against the component map — what the run froze is what it froze, even if this
    Noctis names its components differently.
    """
    if not isinstance(engine, Mapping):
        return None
    fingerprint = engine.get("fingerprint")
    version = engine.get("engine_version")
    if not isinstance(fingerprint, Mapping) or not isinstance(version, int):
        return None
    epoch = engine.get("engine_epoch")
    changes = engine.get("engine_changes")
    return EngineIdentity(
        engine_version=version,
        fingerprint=dict(fingerprint),  # type: ignore[arg-type]
        comparable_key=str(engine.get("comparable_key", "")),
        noctis_version=str(engine.get("noctis_version", "")),
        engine_epoch=epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else 1,
        engine_changes=tuple(change for change in changes if isinstance(change, Mapping))
        if isinstance(changes, list)
        else (),
    )


def _frozen_inputs(inputs: object) -> Mapping[str, object] | None:
    """The record's frozen ``inputs``, or ``None`` for a run that never froze a configuration.

    Deliberately **not** parsed into a typed shape: the freezing policy lives in
    ``config.rehydrate``, and a second reader here would be a second interpretation of it. This
    only says whether there is a block to carry forward.
    """
    return inputs if isinstance(inputs, Mapping) else None


def _listed(prior: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    """One of the record's lists, or a raised error if it is not the shape we wrote."""
    values = prior.get(key, [])
    if not isinstance(values, list) or not all(isinstance(item, Mapping) for item in values):
        raise TypeError(f"the {key!r} section is not a list of objects")
    return values  # type: ignore[return-value]


def read_engine_identity(election_metric: str, root: Path | None = None) -> EngineIdentity:
    """This engine's identity: the declared version, the per-component digests, the bucket key.

    Computed at every open (it reads source files, so it belongs on this side of the boundary),
    and stamped onto the segment as well as the run — a run resumed after a code change ran two
    engines and the record must be able to say so.
    """
    from noctis.observability.engine_id import comparable_key, fingerprint

    fp = fingerprint(root)
    return EngineIdentity(
        engine_version=fp.engine_version,
        fingerprint=fp.digests(),
        comparable_key=str(comparable_key(election_metric, fp)),
        noctis_version=_noctis_version(),
    )


def _upgraded(
    frozen: EngineIdentity, current: EngineIdentity, entry: Mapping[str, object] | None
) -> EngineIdentity:
    """The run's engine identity after a deliberately accepted engine change (story #135).

    Re-frozen onto **this process's** engine — its digests, and therefore its comparable key, which
    is the honest consequence of accepting that the arbiter moved: the run's later numbers belong
    to a different bucket than its earlier ones, and ``mixed_engine`` plus this entry are what say
    so. The prior entries are carried forward, never rewritten, so a run upgraded twice keeps both
    stories. Without an entry the frozen identity is returned untouched, which is every other open.
    """
    if entry is None:
        return frozen
    epoch = entry.get("to_epoch")
    return EngineIdentity(
        engine_version=current.engine_version,
        fingerprint=dict(current.fingerprint),
        comparable_key=current.comparable_key,
        noctis_version=current.noctis_version,
        engine_epoch=epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else 1,
        engine_changes=(*frozen.engine_changes, dict(entry)),
    )


def write(run_dir: Path | str, record: Mapping[str, object]) -> None:
    """Write ``run.json`` atomically: a temp file beside it, then ``os.replace``.

    ``os.replace`` is atomic on every platform Noctis supports, so a reader (or a kill) sees
    either the whole previous record or the whole new one — never a half-written file. The temp
    file is removed on failure so a crashed write leaves no litter beside the record.
    """
    _write_json(Path(run_dir) / RUN_RECORD_NAME, record)


def _write_json(target: Path, document: Mapping[str, object]) -> None:
    """One atomic JSON write, shared by the record and the index — same discipline, one copy."""
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ── addressing ─────────────────────────────────────────────────────────────────────────────

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
        label=_optional_str(run.get("label")),
        last_active=str(run.get("last_active_utc") or run.get("created_utc") or ""),
        resumable=resume_refusal(record or {}) is None,
        readable=True,
    )


# ── the derived index ──────────────────────────────────────────────────────────────────────


def read_record(run_dir: Path | str) -> tuple[dict | None, str | None]:
    """One run's record, or ``(None, why)`` when there is not a readable one.

    The reading half of "a broken record is evidence, not a crash": the caller gets a reason it
    can *show* — no record yet, unreadable JSON, a foreign shape — instead of an exception that
    would take a whole listing down with one bad file.
    """
    return _record_at(Path(run_dir) / RUN_RECORD_NAME)


def index_entry(run_dir: Path | str) -> dict:
    """One run's listing entry, derived from its record alone — no sidecar, no other file.

    Carries ``comparable_key`` (always, ``null`` when unknown), so a leaderboard partitions
    structurally instead of trusting a human to remember which runs may be pooled. Every key is
    always present: an absent value is an explicit ``null``, the record's own convention.
    """
    path = Path(run_dir)
    record, note = read_record(path)
    if record is not None:
        try:
            return _entry_from(record, run_dir=path)
        except Exception as exc:  # a hand-edited or foreign file, still valid JSON
            note = f"an unreadable {RUN_RECORD_NAME} ({type(exc).__name__}: {exc})"
    return _unreadable_entry(path.name, note)


def rebuild_index(runs_dir: Path | str) -> dict:
    """Regenerate the whole roll-up from the records on disk. Cheap, pure of history, idempotent.

    This is what "derived, never authoritative" means operationally: the index can be deleted at
    any moment and this reproduces it exactly, so nothing downstream ever has to trust it more
    than the records it summarizes.
    """
    runs = Path(runs_dir)
    directories = [p for p in runs.iterdir() if p.is_dir()] if runs.is_dir() else []
    return _index_of(index_entry(run_dir) for run_dir in directories)


def update_index(runs_dir: Path | str, run_id: str) -> None:
    """Refresh one run's entry in the index, leaving every other entry alone.

    Re-derived from that run's record **on disk**, never from a caller's in-memory copy, so the
    incrementally-maintained file cannot describe a record that was never written. An index that
    is missing, unreadable, or of another shape is rebuilt from scratch rather than patched: it
    is derived, so throwing it away costs nothing.
    """
    runs = Path(runs_dir)
    index = _read_index(runs)
    if index is None:
        write_index(runs, rebuild_index(runs))
        return
    others = [entry for entry in index["runs"] if entry.get("run_id") != run_id]
    write_index(runs, _index_of([*others, index_entry(runs / run_id)]))


def write_index(runs_dir: Path | str, index: Mapping[str, object]) -> None:
    """Write ``index.json`` atomically — the same tmp + ``os.replace`` the record uses."""
    _write_json(Path(runs_dir) / RUN_INDEX_NAME, index)


def visible_runs(
    entries: Sequence[Mapping[str, object]], *, include_all: bool = False
) -> list[Mapping[str, object]]:
    """The default listing: every run **except** finished ones shorter than :data:`SHORT_RUN_S`.

    A run that stopped after a handful of seconds produced no evidence — it is a startup failure,
    a mistyped command or a config typo — and a board full of those hides the experiments an
    operator came to compare. ``include_all`` (the CLI's ``--all``) widens to everything.

    Three kinds are **never** hidden, whatever their runtime: a run that is still ``running`` (the
    one you are most likely looking for), a run whose record could not be read (breakage is
    exactly what a listing exists to surface, so tidiness must not swallow it), and a run with no
    segments at all — the adopted-history shape (story #131), which is the opposite of noise: a
    failed start still writes the segment it failed in, so zero segments means the run's contents
    predate runs entirely rather than that nothing happened.
    """
    if include_all:
        return list(entries)
    return [entry for entry in entries if not _is_noise(entry)]


def _is_noise(entry: Mapping[str, object]) -> bool:
    if not entry.get("readable", True) or entry.get("status") == "running":
        return False
    if entry.get("segments") == 0:  # adopted history, never a startup failure
        return False
    runtime = entry.get("cumulative_runtime_s")
    return isinstance(runtime, int | float) and float(runtime) < SHORT_RUN_S


def _index_of(entries: Iterable[Mapping[str, object]]) -> dict:
    """The index document: newest run first, and nothing that varies between two rebuilds.

    Deliberately carries **no generation stamp** — a derived file that changed on every rebuild
    could not be compared against the incrementally-maintained one, and that comparison is the
    only thing keeping the two paths honest.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_INDEX_KIND,
        "runs": sorted(entries, key=lambda entry: str(entry.get("run_id") or ""), reverse=True),
    }


def _entry_from(record: Mapping[str, object], *, run_dir: Path) -> dict:
    """One record, reduced to its listing entry. Raises on a shape it cannot read."""
    run = record.get("run")
    engine = record.get("engine")
    segments = record.get("segments")
    if not isinstance(run, Mapping) or not isinstance(engine, Mapping):
        raise TypeError("the 'run' or 'engine' section is missing or is not an object")
    if not isinstance(segments, list):
        raise TypeError("the 'segments' section is missing or is not a list")
    version = engine.get("engine_version")
    return {
        "run_id": str(run.get("run_id") or run_dir.name),
        "label": _optional_str(run.get("label")),
        "status": _optional_str(run.get("status")),
        "created_utc": _optional_str(run.get("created_utc")),
        "last_active_utc": _optional_str(run.get("last_active_utc")),
        "segments": len(segments),
        "cumulative_runtime_s": _optional_number(run.get("cumulative_runtime_s")),
        # The compute the run was given, beside the compute it has used: a listing that shows one
        # without the other cannot answer "are these two runs comparable?", which is the question
        # the cap exists to make answerable (100 research hours and 30 are not one experiment).
        "run_limit_hours": _optional_number(run.get("run_limit_hours")),
        "complete": bool(run.get("complete", False)),
        "engine_version": version if isinstance(version, int) else None,
        "comparable_key": _optional_str(engine.get("comparable_key")),
        "mixed_engine": bool(engine.get("mixed_engine", False)),
        "readable": True,
        "note": None,
    }


def _unreadable_entry(run_id: str, note: str | None) -> dict:
    """A run that could not be read, listed as exactly that — same keys, honest nulls."""
    return {
        "run_id": run_id,
        "label": None,
        "status": None,
        "created_utc": None,
        "last_active_utc": None,
        "segments": None,
        "cumulative_runtime_s": None,
        "run_limit_hours": None,
        "complete": False,
        "engine_version": None,
        "comparable_key": None,
        "mixed_engine": None,
        "readable": False,
        "note": note,
    }


def _read_index(runs_dir: Path) -> dict | None:
    """The index as written, or ``None`` when there is nothing here worth patching."""
    try:
        index = json.loads((runs_dir / RUN_INDEX_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or index.get("kind") != RUN_INDEX_KIND:
        return None
    if index.get("schema_version") != SCHEMA_VERSION:
        return None
    entries = index.get("runs")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        return None
    return index


# ── the lock ───────────────────────────────────────────────────────────────────────────────


def acquire_lock(
    run_dir: Path,
    *,
    run_id: str,
    now: datetime,
    stale_after_s: float = STALE_HEARTBEAT_S,
) -> RecordEvent | None:
    """Take the run's liveness lock, or refuse. Returns the steal note when one was needed.

    **Refusal is the point.** A held lock means another engine is working this run, and two
    engines writing one run corrupt it, so this raises :class:`RunLockedError` naming the holder
    and the lock file rather than degrading to a shared write. Only a *stale* lock is taken
    (:func:`_stale_reason`), and taking one is recorded: a warning here and an event on the
    record, so a run's history is never silently re-attributed.
    """
    lock_path = run_dir / RUN_LOCK_NAME
    held = _read_lock(lock_path)
    note: RecordEvent | None = None
    if held is not None:
        reason = _stale_reason(held, now=now, stale_after_s=stale_after_s)
        if reason is None:
            raise RunLockedError(
                f"run {run_id} is already open by pid {held.get('pid')} on host "
                f"{held.get('hostname_hash')} (heartbeat {held.get('heartbeat_utc')}). "
                f"Two engines writing one run would corrupt it, so this one refuses to start. "
                f"Stop the other engine, or remove {lock_path} once you are certain it is gone."
            )
        text = f"stole a stale run lock held by pid {held.get('pid')}: {reason}"
        logger.warning("run %s: %s", run_id, text)
        note = RecordEvent(t=utc_iso(now), kind="warn", text=text)
    _write_lock(lock_path, run_id=run_id, started=now, heartbeat=now)
    return note


def _stale_reason(lock: Mapping[str, object], *, now: datetime, stale_after_s: float) -> str | None:
    """Why this lock may be taken, or ``None`` when it must be respected.

    Two independent pieces of evidence, in order of strength: a pid that is provably gone **on
    this host** (checking a pid on another host would be meaningless — the number belongs to a
    different process table), and a heartbeat colder than the threshold, which is the only signal
    available for a holder we cannot inspect.
    """
    pid = lock.get("pid")
    if lock.get("hostname_hash") == _hostname_hash() and isinstance(pid, int):
        if not _pid_alive(pid):
            return f"pid {pid} is not running on this host"
    age = _heartbeat_age_s(lock.get("heartbeat_utc"), now=now)
    if age is not None and age > stale_after_s:
        return f"its heartbeat is {int(age)}s cold (older than the {int(stale_after_s)}s threshold)"
    return None


def _heartbeat_age_s(heartbeat: object, *, now: datetime) -> float | None:
    if not isinstance(heartbeat, str):
        return None
    try:
        stamp = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (now - stamp).total_seconds()


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` exists on this host. ``kill(pid, 0)`` signals nothing; it only asks."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # someone else's process — alive, just not ours to signal
        return True
    except OSError:
        return False
    return True


def _hostname_hash() -> str:
    """A stable, non-identifying host id: ``sha256(hostname)[:12]``.

    Hashed, not raw, because the record is meant to be shareable — two segments on one machine
    are still provably the same host, without publishing a machine name.
    """
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]


def _write_lock(path: Path, *, run_id: str, started: datetime, heartbeat: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "hostname_hash": _hostname_hash(),
                "started_utc": utc_iso(started),
                "heartbeat_utc": utc_iso(heartbeat),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_lock(path: Path) -> dict | None:
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return held if isinstance(held, dict) else None


# ── the store ──────────────────────────────────────────────────────────────────────────────


class RunStore:
    """One process invocation's handle on a run: its open segment, its lock, its record.

    Built by :func:`open_run` (which does the locking and the reading), then driven by the engine:
    :meth:`checkpoint` at each CLOSE, :meth:`note` for anything worth recording, :meth:`close` when
    the process stops. Every public method runs behind the fail-safe latch, so none of them can
    raise into the engine.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        artifacts: RunArtifacts,
        clock: Callable[[], datetime],
        argv: Sequence[str],
        command: str = "run",
        label: str | None = None,
        writer: Callable[[Path, dict], None] = write,
        opening_note: RecordEvent | None = None,
        inputs: Mapping[str, object] | None = None,
        rebase_config: bool = False,
        engine_upgrade: Mapping[str, object] | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._clock = clock
        self._writer = writer
        self._disabled = False
        self._closed = False
        self._complete = False

        now = clock()
        prior = tuple(artifacts.segments)
        # The runtime this run had accumulated before this process opened its segment — read once,
        # off the record's own derived total, and constant for the life of the segment. It is what
        # a run-level cap is measured against (story #136).
        self._prior_runtime_s = _runtime_of(build(artifacts))
        current = artifacts.current_engine or artifacts.engine
        self._segment = SegmentArtifact(
            index=len(prior),
            started_utc=utc_iso(now),
            # This process's engine, not the run's: a segment records what actually produced it.
            engine=current,
            status="running",
            argv=tuple(argv),
            command=command,
            resumed=bool(prior),
            counters={},
        )
        self._artifacts = RunArtifacts(
            run_id=artifacts.run_id,
            created_utc=artifacts.created_utc or utc_iso(now),
            last_active_utc=utc_iso(now),
            # Frozen at creation, unless an engine change was deliberately accepted.
            engine=_upgraded(artifacts.engine, current, engine_upgrade),
            current_engine=current,
            segments=prior + (self._segment,),
            label=label,
            completed_utc=artifacts.completed_utc,
            complete=False,
            events=tuple(artifacts.events),
            errors=tuple(artifacts.errors),
            # Frozen at creation: the record's own inputs win, and this process's are taken only
            # by a run that has never frozen any (a fresh one, or an adopted history) — or by a
            # deliberate ``--rebase-config``, which is the operator saying "adopt these instead".
            inputs=inputs if rebase_config or artifacts.inputs is None else artifacts.inputs,
            trials=artifacts.trials,
        )
        if opening_note is not None:
            self._append(
                RecordEvent(
                    t=opening_note.t,
                    kind=opening_note.kind,
                    text=opening_note.text,
                    segment=self._segment.index,
                )
            )
        self._guarded(self._flush)

    # ── the public surface ─────────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._artifacts.run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def record_path(self) -> Path:
        return self._run_dir / RUN_RECORD_NAME

    @property
    def disabled(self) -> bool:
        """Whether the latch has tripped. Once ``True`` it never clears."""
        return self._disabled

    def record(self) -> dict:
        """The record as it stands — what the next write would put on disk."""
        return build(self._artifacts)

    @property
    def prior_runtime_s(self) -> float:
        """Runtime this run had already accumulated **before** this segment opened, in seconds.

        What a run-level cap is measured against (story #136): the engine adds its own elapsed time
        to this and stops between phases once the sum crosses the cap. Taken once at open from the
        record's own derived total, so a resumed process inherits the run's history rather than
        starting the count again — and so the number the engine measures against does not move
        under it while the segment runs.
        """
        return self._prior_runtime_s

    def checkpoint(
        self,
        counters: Mapping[str, int] | None = None,
        *,
        phase_seconds: Mapping[str, float] | None = None,
    ) -> None:
        """Rewrite the record and touch the heartbeat. Called at each CLOSE."""
        self._guarded(self._checkpoint, counters, phase_seconds)

    def note(self, text: str, *, kind: str = "warn") -> None:
        """Record one event against the open segment (``kind="error"`` files it under errors)."""
        self._guarded(self._note, text, kind)

    def close(
        self,
        *,
        reason: str,
        counters: Mapping[str, int] | None = None,
        phase_seconds: Mapping[str, float] | None = None,
    ) -> None:
        """Close the segment (stop stamp, duration, reason), write, release the lock. Idempotent.

        The lock is released whatever happened to the record — including after the latch tripped —
        because a lock nobody holds must never be the thing that blocks the next invocation.
        """
        self._guarded(self._close, reason, counters, phase_seconds)
        self._release_lock()

    # ── the fail-safe latch ────────────────────────────────────────────────────────────────

    def _guarded(self, work: Callable[..., None], *args: object) -> None:
        """One choke point every public body delegates through, so none can forget the guard."""
        if self._disabled:
            return
        try:
            work(*args)
        except Exception as exc:
            self._trip(exc)

    def _trip(self, exc: BaseException) -> None:
        """Disable the store for good on the first internal failure — a LATCH, not a retry.

        Exactly one warning, then silence: a run record is evidence, and evidence that cannot be
        written must not take the run down with it. The record left on disk keeps
        ``complete: false`` (the final clean-close write never happens), so nothing that reads it
        later can mistake a truncated record for a whole one.
        """
        if self._disabled:
            return
        self._disabled = True
        self._complete = False
        logger.warning(
            "run record %s self-disabled after an internal failure (%s: %s); the record on disk "
            "is marked incomplete and will not be updated again this segment",
            self._artifacts.run_id,
            type(exc).__name__,
            exc,
        )

    # ── the guarded bodies ─────────────────────────────────────────────────────────────────

    def _checkpoint(
        self, counters: Mapping[str, int] | None, phase_seconds: Mapping[str, float] | None
    ) -> None:
        now = self._clock()
        changes = self._measurements(counters, phase_seconds)
        if changes:
            self._replace_segment(**changes)
        self._touch(now)
        self._flush()

    def _note(self, text: str, kind: str) -> None:
        self._append(
            RecordEvent(t=utc_iso(self._clock()), kind=kind, text=text, segment=self._segment.index)
        )
        self._flush()

    def _close(
        self,
        reason: str,
        counters: Mapping[str, int] | None,
        phase_seconds: Mapping[str, float] | None,
    ) -> None:
        if self._closed:
            return
        now = self._clock()
        self._replace_segment(
            **self._measurements(counters, phase_seconds),
            stopped_utc=utc_iso(now),
            stopped_reason=reason,
            status="stopped",
        )
        self._touch(now)
        self._complete = True
        self._flush()
        self._closed = True

    # ── internals (no I/O beyond the injected writer) ───────────────────────────────────────

    @staticmethod
    def _measurements(
        counters: Mapping[str, int] | None, phase_seconds: Mapping[str, float] | None
    ) -> dict[str, object]:
        """The two per-segment measurements a caller may hand in, kept apart from what is absent.

        An omitted measurement leaves what the segment already carries alone (a checkpoint that
        knows only about counters must not blank the phase timings, and vice versa), which is what
        makes the last write of a segment the sum of everything measured during it.
        """
        changes: dict[str, object] = {}
        if counters:
            changes["counters"] = dict(counters)
        if phase_seconds:
            changes["phase_seconds"] = dict(phase_seconds)
        return changes

    def _replace_segment(self, **changes: object) -> None:
        """Swap the open segment for an updated copy, in the artifacts as well as the handle."""
        updated = replace(self._segment, **changes)  # type: ignore[arg-type]
        self._segment = updated
        self._replace_artifacts(segments=tuple(self._artifacts.segments[:-1]) + (updated,))

    def _append(self, event: RecordEvent) -> None:
        if event.kind == "error":
            self._replace_artifacts(errors=tuple(self._artifacts.errors) + (event,))
        else:
            self._replace_artifacts(events=tuple(self._artifacts.events) + (event,))

    def _touch(self, now: datetime) -> None:
        self._replace_artifacts(last_active_utc=utc_iso(now))
        _write_lock(
            self._run_dir / RUN_LOCK_NAME,
            run_id=self._artifacts.run_id,
            started=now,
            heartbeat=now,
        )

    def _replace_artifacts(self, **changes: object) -> None:
        current = self._artifacts
        self._artifacts = RunArtifacts(
            run_id=current.run_id,
            created_utc=current.created_utc,
            last_active_utc=str(changes.get("last_active_utc", current.last_active_utc)),
            engine=current.engine,
            current_engine=current.current_engine,
            segments=changes.get("segments", current.segments),  # type: ignore[arg-type]
            label=current.label,
            completed_utc=current.completed_utc,
            complete=self._complete,
            events=changes.get("events", current.events),  # type: ignore[arg-type]
            errors=changes.get("errors", current.errors),  # type: ignore[arg-type]
            inputs=current.inputs,
            trials=changes.get("trials", current.trials),  # type: ignore[arg-type]
        )

    def _flush(self) -> None:
        # Derived at write time, like every cumulative number the record carries: the run's own
        # journals are re-counted here rather than handed forward, so a segment that journalled
        # trials all night lands them on disk without anyone tracking a total in memory.
        self._replace_artifacts(trials=read_trials(self._run_dir))
        self._writer(self._run_dir, build(self._artifacts))
        # The roll-up follows the record, never leads it: it is refreshed *after* a successful
        # write and re-read from the file just written, so the listing can never advertise a
        # record that is not on disk. A failed write latches the store and skips this entirely.
        update_index(self._run_dir.parent, self._artifacts.run_id)

    def _release_lock(self) -> None:
        """Best effort, always attempted: a stale lock file is friction for the next invocation,
        never a reason to fail this one."""
        try:
            (self._run_dir / RUN_LOCK_NAME).unlink(missing_ok=True)
        except OSError:  # pragma: no cover - a lock we cannot remove is the next open's problem
            pass


# ── reading a record back ──────────────────────────────────────────────────────────────────


def _record_at(path: Path) -> tuple[dict | None, str | None]:
    """The parsed record, or ``(None, reason)`` — the one place a record is read off disk.

    The reason is written to be shown to an operator as-is, because both callers show it: the
    listing puts it in the run's index entry, and the opening path folds it into the record's own
    events. One phrasing, so a broken record is described the same way wherever it surfaces.
    """
    if not path.is_file():
        return None, f"no {path.name} yet"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"an unreadable {path.name} ({type(exc).__name__})"
    if not isinstance(record, dict):
        return None, f"an unreadable {path.name} (not an object)"
    return record, None


def _read_record(path: Path) -> tuple[dict | None, RecordEvent | None]:
    """The prior record for an *opening* run, or ``(None, note)`` when it cannot be read.

    A missing record is the normal case for a fresh run and carries no note; anything else is
    worth an event, because a run whose history could not be read must say so in the record it
    starts in its place.
    """
    record, reason = _record_at(path)
    if record is not None or not path.is_file():
        return record, None
    return None, RecordEvent(
        t=None,
        kind="warn",
        text=f"this run had {reason}; a fresh record was started in its place",
    )


def _segment_from(raw: Mapping[str, object]) -> SegmentArtifact:
    engine = None
    fingerprint = raw.get("engine_fingerprint")
    version = raw.get("engine_version")
    if isinstance(fingerprint, Mapping) and isinstance(version, int):
        engine = EngineIdentity(
            engine_version=version,
            fingerprint=dict(fingerprint),  # type: ignore[arg-type]
            comparable_key=str(raw.get("comparable_key", "")),
            noctis_version=str(raw.get("noctis_version", "")),
        )
    counters = raw.get("counters")
    phases = raw.get("phase_seconds")
    argv = raw.get("argv")
    index = raw.get("index")
    return SegmentArtifact(
        index=index if isinstance(index, int) else 0,
        started_utc=str(raw.get("started_utc")),
        engine=engine,
        stopped_utc=_optional_str(raw.get("stopped_utc")),
        stopped_reason=_optional_str(raw.get("stopped_reason")),
        status=str(raw.get("status", "running")),
        argv=tuple(str(part) for part in argv) if isinstance(argv, Sequence) else (),
        command=str(raw.get("command", "run")),
        resumed=bool(raw.get("resumed", False)),
        counters=dict(counters) if isinstance(counters, Mapping) else {},  # type: ignore[arg-type]
        # Read back so the run's cumulative research/trading seconds are re-derived from every
        # segment on disk at every write — the totals are never carried in memory across a restart.
        phase_seconds=dict(phases) if isinstance(phases, Mapping) else None,  # type: ignore[arg-type]
    )


def _event_from(raw: Mapping[str, object]) -> RecordEvent:
    segment = raw.get("segment")
    return RecordEvent(
        t=_optional_str(raw.get("t")),
        kind=str(raw.get("kind", "info")),
        text=str(raw.get("text", "")),
        segment=segment if isinstance(segment, int) else None,
    )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _runtime_of(record: Mapping[str, object]) -> float:
    """One record's cumulative runtime in seconds, or ``0.0`` when it carries none readable."""
    run = record.get("run")
    total = run.get("cumulative_runtime_s") if isinstance(run, Mapping) else None
    return float(total) if isinstance(total, int | float) and not isinstance(total, bool) else 0.0


def _optional_number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _noctis_version() -> str:
    """The package literal — informational beside the engine version, never a comparison key."""
    from importlib import metadata

    try:
        return metadata.version("noctis")
    except Exception:  # not pip-installed (editable/source tree)
        from noctis import __version__

        return __version__
