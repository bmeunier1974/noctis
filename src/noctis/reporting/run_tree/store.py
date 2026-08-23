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
more fetch. Both readers of that tree are modules of their own, holding nothing but the record:
:mod:`~noctis.reporting.run_tree.address` (one operator-typed string → one run dir) and
:mod:`~noctis.reporting.run_tree.index` (the roll-up, **derived, never authoritative**). This
module drives them — it resolves the address a verb was given, and refreshes the index after every
write it makes.

Everything in this module is I/O; everything about the record's *shape* is next door in
``run_record`` (pure) and ``schema`` (pure). A record is read in two halves — :func:`read_artifacts`
parses what the run already said about itself, and
:func:`~noctis.reporting.run_tree.evidence.derive_evidence` reads the run's own durable artifacts
once per write — joined by :func:`with_evidence` into the
:class:`~noctis.reporting.run_record.RunArtifacts` that
:func:`~noctis.reporting.run_tree.record.write` puts on disk in one write. That boundary is what
makes the golden-record and segmentation-equivalence tests cheap — they build a ``RunArtifacts`` in
memory and never go near a disk.

**The lock is the one fatal failure in the whole epic**, and it is a module of its own:
:mod:`~noctis.reporting.run_tree.lock`. Everything here is latched (below), because a reporting
artifact must never take down a multi-week run; two engines writing one run is *corruption*, not
degradation, so a live lock is a hard refusal instead. This module only drives the four verbs —
take the lock at an open, touch it at each checkpoint, release it at close, and assert it unheld
before sealing or pruning.

**The fail-safe latch**, straight from ``observability/debug/recorder.py``: the first internal
exception logs exactly one warning, disables the store, and every later call is a no-op — no retry,
no second warning, nothing raised into the engine. A latched store never gets to write its final
``complete: true``, so the record left on disk says ``complete: false`` and a partial record can
never pass for a whole one.

**Writes are synchronous and atomic.** No background thread (the engine spent four PRs removing
shutdown join hazards; a writer thread would put one back), the temp file plus ``os.replace`` of
:mod:`~noctis.reporting.run_tree.record` so a kill mid-write leaves the previous record intact,
and an injected clock so no ``datetime.now()`` is ever reached from here.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from noctis.reporting import schema as schema_module
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
from noctis.reporting.run_tree.address import RunNotFoundError, resolve_run_dir
from noctis.reporting.run_tree.evidence import Evidence, derive_evidence, read_engine_identity
from noctis.reporting.run_tree.index import update_index
from noctis.reporting.run_tree.lock import (
    STALE_HEARTBEAT_S,
    acquire_lock,
    assert_unlocked,
    release_lock,
    touch_lock,
)
from noctis.reporting.run_tree.record import (
    RUN_RECORD_NAME,
    optional_str,
    read_record,
    write,
)

__all__ = [
    "PRUNED_SUBDIRS",
    "FinishOutcome",
    "PruneOutcome",
    "RunCompletedError",
    "RunNotPrunableError",
    "RunStore",
    "assert_resumable",
    "finish_run",
    "open_run",
    "prune_run_state",
    "read_artifacts",
    "with_evidence",
]

logger = logging.getLogger(__name__)

# The only directories retention may ever remove (story #138), named here **as literal children of
# one run dir** — this list is the entire blast radius, and it is a constant so that reviewing it is
# reviewing the whole destructive surface. They are the heavy, re-derivable ones; ``run.json`` and
# ``index.json`` are never pruned (they are small, and they *are* the long-term progress history),
# and neither is anything else the tree happens to hold — the run's ``memory/`` and its ``qa/`` area
# (which has retention of its own) are left exactly where they are.
PRUNED_SUBDIRS = ("state", "strategies", "reports")


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
    environment: Mapping[str, object] | None = None,
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

    ``environment`` is the machine **this process** is on (story #139), already captured through
    the injected probes of ``observability.environment``. The store is the I/O side of the record's
    boundary, but the probes are wired where every other collaborator is — the composition root —
    so nothing here reads hardware, shells out to ``git`` or imports an optional package. It lands
    on the appending segment and nowhere else: earlier segments keep the machines they actually
    ran on, which is the entire reason the block is per segment.
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
        # `completed` is terminal, not "terminal as long as the caller passed resume=True". The
        # two arguments always travel together today, so this is unreachable from the CLI — but a
        # published run silently gaining a segment is the failure this status exists to prevent,
        # and the guard belongs where the segment is opened rather than in each caller.
        if run_id is not None and run_dir.is_dir():
            _assert_resumable(run_dir, resolved_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    steal_note = acquire_lock(run_dir, run_id=resolved_id, now=now, stale_after_s=stale_after_s)
    # The record, and nothing derived: an open parses what the run already said about itself, and
    # the evidence is read once by the flush that ends ``RunStore.__init__`` (story #288).
    engine = read_engine_identity(election_metric, root=engine_root)
    artifacts = mark_interrupted(read_artifacts(run_dir, current=engine))
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
        environment=environment,
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
    assert_unlocked(
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
        return FinishOutcome(run_id=run_id, sealed=False, completed_utc=optional_str(stamp))

    # Both halves of the read: sealing writes the record itself, without a store, so it derives
    # the evidence under the record's **own** frozen inputs exactly as a flush would.
    engine = read_engine_identity(election_metric, root=engine_root)
    artifacts = mark_interrupted(read_artifacts(run_dir, current=engine))
    artifacts = with_evidence(artifacts, derive_evidence(run_dir, artifacts.inputs))
    stamp = utc_iso(now)
    (writer or write)(run_dir, build(seal(artifacts, at=stamp)))
    update_index(run_dir.parent, run_id)
    return FinishOutcome(run_id=run_id, sealed=True, completed_utc=stamp)


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
    assert_unlocked(
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
    # the very journals this is about to delete, so both halves of the read happen here, above the
    # ``rmtree``, and never after it.
    engine = read_engine_identity(election_metric, root=engine_root)
    artifacts = mark_interrupted(read_artifacts(run_dir, current=engine))
    artifacts = with_evidence(artifacts, derive_evidence(run_dir, artifacts.inputs))
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


def read_artifacts(run_dir: Path | str, *, current: EngineIdentity) -> RunArtifacts:
    """The run's own record, parsed back into artifacts — the record, and nothing derived.

    Returns the artifacts of an existing record (segments, stamps, label, inputs, events) or, for
    a run dir with no record yet, an empty run under this engine's identity. The seven derived
    fields are left at their defaults on purpose: they are
    :func:`~noctis.reporting.run_tree.evidence.derive_evidence`'s half of the read, taken once by
    whoever is about to write. Opening a run needs only this half, which is why an open no longer
    reads the run's journals, its ledgers and the lake before the flush reads them again.

    An unreadable record is not fatal: it degrades to a fresh record carrying an event that says
    so, because a corrupt reporting file must never stop a run from starting.
    """
    prior, note = _read_record(Path(run_dir))
    prior, upgrade_note = _upgraded_schema(prior)
    if prior is not None:
        try:
            artifacts = _artifacts_from(prior, run_dir=Path(run_dir), current=current)
            if upgrade_note is None:
                return artifacts
            # The schema upgrade goes on the run's own event stream, where the run says what
            # happened to it: the next write puts the upgraded document on disk, and this is the
            # line that says the shape changed under a consumer comparing two of its segments.
            return replace(artifacts, events=(*artifacts.events, upgrade_note))
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
        engine=current,
        current_engine=current,
        events=(note,) if note is not None else (),
    )


def with_evidence(artifacts: RunArtifacts, evidence: Evidence) -> RunArtifacts:
    """The artifacts with their derived fields replaced by evidence just read off the run tree.

    The other half of the pair, and the one line that says what "derived" means: everything a
    record carries about a run *other* than these seven fields is parsed from the record and
    carried forward verbatim.
    """
    # One copy over seven differently-typed fields, splatted by name: ``Evidence``'s own field
    # names *are* the derived fields, so this cannot name one ``RunArtifacts`` does not have — but
    # a ``**`` of a heterogeneous mapping is exactly what a static type cannot say.
    changes: dict[str, Any] = evidence.changes()
    return replace(artifacts, **changes)


def _upgraded_schema(prior: dict | None) -> tuple[dict | None, RecordEvent | None]:
    """Bring a record written under an older schema up to this engine's, and say so (story #143).

    The epic's promise to a multi-week run is that **today's run is still resumable by tomorrow's
    engine**, and this is where that promise is kept: the version walk itself is pure and lives in
    ``schema.upgrade``; the upgrade lands *in place* because the next ordinary write puts the
    upgraded document on disk, and the event returned beside it is what stops the change being
    silent. A record already at this version is returned untouched and produces no event, so a
    run that is simply resumed does not accumulate a note per night.

    The event carries no segment for the same reason the unreadable-record note next to it does
    not: the observation is made while the run is being *opened*, before this process's segment
    exists.

    Never fatal, like everything else in this module except the lock: an upgrade step that raises
    leaves the record exactly as it was found and files the reason, because a reporting artifact
    must not be what stops a multi-week run from opening.
    """
    if prior is None:
        return None, None
    try:
        upgrade = schema_module.upgrade(prior)
    except Exception as exc:  # pragma: no cover - unreachable while no step is registered
        return prior, RecordEvent(
            t=None,
            kind="warn",
            text=f"this run's record could not be upgraded to schema version "
            f"{schema_module.SCHEMA_VERSION} ({type(exc).__name__}); it is being read as written",
        )
    note = upgrade.note()
    if note is None:
        return prior, None
    return upgrade.record, RecordEvent(t=None, kind="info", text=note)


def _artifacts_from(
    prior: Mapping[str, object],
    *,
    run_dir: Path,
    current: EngineIdentity,
) -> RunArtifacts:
    """One prior record, parsed back into artifacts. Raises on a shape it cannot read.

    The seven derived fields — ``trials``, ``spend``, ``pricing_table_version``, ``champions``,
    ``strategies``, the realised ``sessions`` and the ``benchmark`` — are deliberately **not** read
    back off the record and are left at their defaults here: every one of them is derived from the
    run's own durable artifacts at every write
    (:func:`~noctis.reporting.run_tree.evidence.derive_evidence`), and a value carried forward from
    a prior write could only ever go stale — or, worse for the equity curve, be double-counted by a
    segment that appended to it.
    """
    run = prior.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("the 'run' section is missing or is not an object")
    return RunArtifacts(
        run_id=str(run.get("run_id") or run_dir.name),
        created_utc=optional_str(run.get("created_utc")),
        last_active_utc=optional_str(run.get("last_active_utc")),
        # Frozen at creation and carried forward verbatim, exactly like ``inputs``: the engine a
        # run was created under is the side every later resume is compared against (story #135),
        # so a write must never restamp it with whatever engine happens to be running now.
        engine=_frozen_engine(prior.get("engine")) or current,
        current_engine=current,
        segments=tuple(_segment_from(raw) for raw in _listed(prior, "segments")),
        label=optional_str(run.get("label")),
        completed_utc=optional_str(run.get("completed_utc")),
        complete=bool(run.get("complete", False)),
        events=tuple(_event_from(raw) for raw in _listed(prior, "events")),
        errors=tuple(_event_from(raw) for raw in _listed(prior, "errors")),
        # Read straight back and carried forward verbatim: the run's configuration was frozen at
        # creation, so every later segment restores it rather than re-deriving it from files that
        # may have changed in between.
        inputs=_frozen_inputs(prior.get("inputs")),
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
        fingerprint=dict(fingerprint),
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
    return values


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
        environment: Mapping[str, object] | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._clock = clock
        self._writer = writer
        self._disabled = False
        self._closed = False

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
            # This process's machine, not the run's: like the engine digests beside it, a segment
            # records what actually produced it (story #139).
            environment=dict(environment) if environment else None,
        )
        # A COPY, not a rebuild (story #289): everything this constructor does not name is carried
        # forward by construction, which is what makes the record's "carried forward verbatim"
        # promise — ``inputs``, ``completed_utc``, ``state_pruned`` — hold in code rather than by
        # luck. The seven derived fields are carried as read (since story #288, at their defaults)
        # and the flush that ends this constructor fills every one of them in a single pass before
        # the first byte reaches disk.
        self._artifacts = replace(
            artifacts,
            created_utc=artifacts.created_utc or utc_iso(now),
            last_active_utc=utc_iso(now),
            # Frozen at creation, unless an engine change was deliberately accepted.
            engine=_upgraded(artifacts.engine, current, engine_upgrade),
            current_engine=current,
            segments=prior + (self._segment,),
            label=label,
            complete=False,
            events=tuple(artifacts.events),
            errors=tuple(artifacts.errors),
            # Frozen at creation: the record's own inputs win, and this process's are taken only
            # by a run that has never frozen any (a fresh one, or an adopted history) — or by a
            # deliberate ``--rebase-config``, which is the operator saying "adopt these instead".
            inputs=inputs if rebase_config or artifacts.inputs is None else artifacts.inputs,
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
        release_lock(self._run_dir)

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
        later can mistake a truncated record for a whole one — and the artifacts held here are
        un-said with it, so a latch that tripped *during* the closing flush leaves nothing behind
        that still claims the run finished cleanly.
        """
        if self._disabled:
            return
        self._disabled = True
        self._artifacts = replace(self._artifacts, complete=False)
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
        # The one write that may say so: ``complete`` flips here, one line before the flush that
        # puts it on disk, and nowhere else.
        self._artifacts = replace(self._artifacts, complete=True)
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
        self._artifacts = replace(
            self._artifacts, segments=tuple(self._artifacts.segments[:-1]) + (updated,)
        )

    def _append(self, event: RecordEvent) -> None:
        if event.kind == "error":
            self._artifacts = replace(
                self._artifacts, errors=tuple(self._artifacts.errors) + (event,)
            )
        else:
            self._artifacts = replace(
                self._artifacts, events=tuple(self._artifacts.events) + (event,)
            )

    def _touch(self, now: datetime) -> None:
        self._artifacts = replace(self._artifacts, last_active_utc=utc_iso(now))
        touch_lock(self._run_dir, run_id=self._artifacts.run_id, now=now)

    def _flush(self) -> None:
        # Derived at write time, like every cumulative number the record carries: the run's own
        # journals, ledgers, champion board and equity curve are re-read here rather than handed
        # forward, so a segment that journalled trials and burned tokens all night lands them on
        # disk without anyone tracking a total in memory (epic D4). **One pass per write** — the
        # open before this one derived nothing (story #288) — priced and embedded under the run's
        # own frozen inputs, and joined onto the artifacts by the same ``with_evidence`` the two
        # storeless verbs use (story #289).
        self._artifacts = with_evidence(
            self._artifacts, derive_evidence(self._run_dir, self._artifacts.inputs)
        )
        self._writer(self._run_dir, build(self._artifacts))
        # The roll-up follows the record, never leads it: it is refreshed *after* a successful
        # write and re-read from the file just written, so the listing can never advertise a
        # record that is not on disk. A failed write latches the store and skips this entirely.
        update_index(self._run_dir.parent, self._artifacts.run_id)


# ── reading a record back ──────────────────────────────────────────────────────────────────


def _read_record(run_dir: Path) -> tuple[dict | None, RecordEvent | None]:
    """The prior record for an *opening* run, or ``(None, note)`` when it cannot be read.

    A missing record is the normal case for a fresh run and carries no note; anything else is
    worth an event, because a run whose history could not be read must say so in the record it
    starts in its place.
    """
    record, reason = read_record(run_dir)
    if record is not None or not (run_dir / RUN_RECORD_NAME).is_file():
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
            fingerprint=dict(fingerprint),
            comparable_key=str(raw.get("comparable_key", "")),
            noctis_version=str(raw.get("noctis_version", "")),
        )
    counters = raw.get("counters")
    phases = raw.get("phase_seconds")
    environment = raw.get("environment")
    argv = raw.get("argv")
    index = raw.get("index")
    return SegmentArtifact(
        index=index if isinstance(index, int) else 0,
        started_utc=str(raw.get("started_utc")),
        engine=engine,
        stopped_utc=optional_str(raw.get("stopped_utc")),
        stopped_reason=optional_str(raw.get("stopped_reason")),
        status=str(raw.get("status", "running")),
        argv=tuple(str(part) for part in argv) if isinstance(argv, Sequence) else (),
        command=str(raw.get("command", "run")),
        resumed=bool(raw.get("resumed", False)),
        counters=dict(counters) if isinstance(counters, Mapping) else {},
        # Read back so the run's cumulative research/trading seconds are re-derived from every
        # segment on disk at every write — the totals are never carried in memory across a restart.
        phase_seconds=dict(phases) if isinstance(phases, Mapping) else None,
        # Carried forward verbatim, like the segment's engine digests: the machine a *past*
        # segment ran on is history, and re-stamping it with whatever this process is running on
        # would be the exact misattribution the per-segment block exists to prevent.
        environment=dict(environment) if isinstance(environment, Mapping) else None,
    )


def _event_from(raw: Mapping[str, object]) -> RecordEvent:
    segment = raw.get("segment")
    return RecordEvent(
        t=optional_str(raw.get("t")),
        kind=str(raw.get("kind", "info")),
        text=str(raw.get("text", "")),
        segment=segment if isinstance(segment, int) else None,
    )


def _runtime_of(record: Mapping[str, object]) -> float:
    """One record's cumulative runtime in seconds, or ``0.0`` when it carries none readable."""
    run = record.get("run")
    total = run.get("cumulative_runtime_s") if isinstance(run, Mapping) else None
    return float(total) if isinstance(total, int | float) and not isinstance(total, bool) else 0.0
