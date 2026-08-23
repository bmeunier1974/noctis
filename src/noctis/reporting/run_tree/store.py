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
``run_record`` (pure) and ``schema`` (pure). :func:`collect` does every read and returns a
:class:`~noctis.reporting.run_record.RunArtifacts`;
:func:`~noctis.reporting.run_tree.record.write` does the one write. That boundary is what makes
the golden-record and segmentation-equivalence tests cheap — they build a ``RunArtifacts`` in
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

import hashlib
import logging
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from noctis.reporting import schema as schema_module
from noctis.reporting.metrics import Benchmark, DailySession, TradeFill
from noctis.reporting.run_record import (
    EMBED_ALL_SOURCES_SETTING,
    EngineIdentity,
    RecordEvent,
    RunArtifacts,
    SegmentArtifact,
    SpendEntry,
    StrategyArtifact,
    build,
    mark_interrupted,
    prune_refusal,
    resume_refusal,
    seal,
    utc_iso,
)
from noctis.reporting.run_tree.address import RunNotFoundError, resolve_run_dir
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
from noctis.reporting.schema import (
    PROMOTED_OUTCOME,
    REJECTED_OUTCOME,
    UNDECIDED_OUTCOME,
)

__all__ = [
    "PRUNED_SUBDIRS",
    "STRATEGIES_SUBDIR",
    "STRATEGY_TIER_SUBDIRS",
    "FinishOutcome",
    "PruneOutcome",
    "RunCompletedError",
    "RunNotPrunableError",
    "RunStore",
    "assert_resumable",
    "collect",
    "finish_run",
    "open_run",
    "prune_run_state",
    "read_benchmark",
    "read_sessions",
    "read_strategies",
    "read_trials",
]

logger = logging.getLogger(__name__)

# The run's champion board, inside its own state dir — the denominator of the record's
# per-champion numbers. The one run-tree name this module still owns: the record's, the lock's and
# the index's are in the modules that own those (``record.RUN_RECORD_NAME``, ``lock.RUN_LOCK_NAME``,
# ``index.RUN_INDEX_NAME``) — one place each, so nothing spells one by hand. The registry owns this
# file's schema.
CHAMPIONS_NAME = "champions.json"

# The only directories retention may ever remove (story #138), named here **as literal children of
# one run dir** — this list is the entire blast radius, and it is a constant so that reviewing it is
# reviewing the whole destructive surface. They are the heavy, re-derivable ones; ``run.json`` and
# ``index.json`` are never pruned (they are small, and they *are* the long-term progress history),
# and neither is anything else the tree happens to hold — the run's ``memory/`` and its ``qa/`` area
# (which has retention of its own) are left exactly where they are.
PRUNED_SUBDIRS = ("state", "strategies", "reports")

# The run's own strategy library, and the two writable tiers inside it — the layout
# ``strategies.library.LibraryPaths.from_settings`` writes and this reads back for the record's
# strategies section (story #141). Spelled here rather than imported: the library module pulls
# pandas and numpy in behind it, and a record write must stay as cheap as the rest of this file
# (``tests/test_run_tree_store.py`` pins that no heavy package is reachable from here). One test
# asserts the two spellings still describe the same directories, which is where a drift would
# surface. The committed ``strategies/`` seeds are deliberately absent: they are read-only input
# every run starts from, so they are nobody's candidate.
STRATEGIES_SUBDIR = "strategies"
TMP_TIER = "__tmp"
CHAMPIONS_TIER = "champions"
# Lowest precedence first, the order the library itself discovers them in: a champion of the same
# name overrides a working-tier draft.
STRATEGY_TIER_SUBDIRS = (TMP_TIER, CHAMPIONS_TIER)


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

    artifacts = mark_interrupted(
        collect(run_dir, election_metric=election_metric, engine_root=engine_root)
    )
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
    engine = read_engine_identity(election_metric, root=engine_root)
    trials = read_trials(run_dir)
    prior, note = _read_record(Path(run_dir))
    prior, upgrade_note = _upgraded_schema(prior)
    # Spend is derived here, beside the trial count and for the same reason, and it is priced
    # under the run's **own** frozen configuration — so resuming tomorrow under different prices
    # cannot restate what last night cost.
    spend, table_version = read_spend(
        run_dir, _frozen_inputs(prior.get("inputs")) if prior else None
    )
    champions = read_champions(run_dir)
    # Candidates are read under the run's own frozen inputs too: whether a source is embedded is
    # the run's choice, made once at creation, not this process's.
    strategies = read_strategies(run_dir, _frozen_inputs(prior.get("inputs")) if prior else None)
    # The realised record (story #142), derived like every cumulative fact beside it: the sessions
    # off the run's own account ledger, and the benchmark priced from the shared lake under the
    # run's own frozen data settings — a read, never a fetch.
    sessions = read_sessions(run_dir)
    benchmark = read_benchmark(sessions, _frozen_inputs(prior.get("inputs")) if prior else None)
    if prior is not None:
        try:
            artifacts = _artifacts_from(
                prior,
                run_dir=Path(run_dir),
                current=engine,
                trials=trials,
                spend=spend,
                pricing_table_version=table_version,
                champions=champions,
                strategies=strategies,
                sessions=sessions,
                benchmark=benchmark,
            )
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
        engine=engine,
        current_engine=engine,
        events=(note,) if note is not None else (),
        trials=trials,
        spend=spend,
        pricing_table_version=table_version,
        champions=champions,
        strategies=strategies,
        sessions=sessions,
        benchmark=benchmark,
    )


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


def read_spend(
    run_dir: Path | str, inputs: Mapping[str, object] | None = None
) -> tuple[tuple[SpendEntry, ...] | None, str | None]:
    """What this run has spent on model judgments, read off its own session ledgers (story #140).

    **Read, never counted** — the exact twin of :func:`read_trials`, and for the same reason: the
    ledgers under ``<run>/state/sessions/*.jsonl`` are the run's, not the process's, so a total
    summed from them is cumulative across every segment by construction and a rewrite after a crash
    cannot double-count it. One :class:`~noctis.reporting.run_record.SpendEntry` per journaled
    episode, priced here (pricing needs the table, and the table comes from the run's frozen
    configuration — both of which are this side of the I/O boundary), leaving the record builder
    with nothing but arithmetic.

    Returns ``(entries, table_version)``: ``(None, None)`` when the run journaled no ledger at all —
    the shape of a run with no LLM key, which must report an unknown bill rather than a free one —
    and an *empty* tuple when a session ran and spent nothing, which is a real zero.

    ``inputs`` is the run's frozen inputs, the only source of a price override: a run prices under
    the table it was created with, so resuming it tomorrow with a different ``research.pricing`` in
    ``config.yaml`` cannot restate what last night cost.

    Never raises: unreadable evidence is missing evidence, not a reason to fail a run's write.
    """
    from noctis.config.settings import run_scoped_paths
    from noctis.research.ledger import SESSIONS_DIRNAME, SessionLedger, episode_usage
    from noctis.research.pricing import table_from_config

    try:
        table = table_from_config(_price_overrides(inputs))
        state_dir = run_scoped_paths(Path(run_dir))["state_dir"]
        ledgers = sorted((Path(state_dir) / SESSIONS_DIRNAME).glob("*.jsonl"))
        if not ledgers:
            return None, None
        entries: list[SpendEntry] = []
        for path in ledgers:
            for episode in SessionLedger.from_path(path).episodes():
                usage = episode_usage(episode)
                entries.append(
                    SpendEntry(
                        at=episode.at or None,
                        stage=episode.stage,
                        model=episode.model,
                        tokens=episode.tokens,
                        usage=usage,
                        usd_estimate=table.estimate_usd(episode.model, usage),
                    )
                )
        return tuple(entries), table.version
    except Exception:  # pragma: no cover - a ledger we cannot read is evidence we do not have
        return None, None


def _price_overrides(inputs: Mapping[str, object] | None) -> Mapping[str, Mapping[str, object]]:
    """The run's own ``research.pricing`` block, read out of its frozen settings (or nothing).

    A plain read, deliberately tolerant: a record that carries no inputs, or carries something
    else in that slot, prices under the shipped table rather than failing a write.
    """
    node: object = inputs
    for key in ("settings", "resolved", "research", "pricing"):
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, Mapping) else {}  # type: ignore[return-value]


def read_sessions(run_dir: Path | str) -> tuple[DailySession, ...]:
    """Every session this run's paper account closed, off its own daily ledger (story #142).

    **Read, never counted** — the same family as :func:`read_trials`, :func:`read_spend` and
    :func:`read_strategies`, and for the same reason: ``<run>/state/equity_curve.jsonl`` is the
    *run's*, not the process's, so the curve derived from it is cumulative across every segment by
    construction. Nothing about the equity curve is carried in memory across a restart; a resumed
    segment re-reads the whole ledger at every write, which is what makes "three short nights equal
    one long night" true of the curve rather than merely intended.

    One :class:`~noctis.reporting.metrics.DailySession` per session date (the ledger deduplicates,
    last write wins), oldest first. Never raises: an unreadable ledger is a curve we do not have,
    not a reason to fail a run's write.
    """
    from noctis.broker.persistence import EQUITY_CURVE_NAME, EquityLedger

    try:
        state_dir = _state_dir(Path(run_dir))
        marks = EquityLedger(state_dir / EQUITY_CURVE_NAME).marks()
    except Exception:  # pragma: no cover - an unreadable ledger is evidence we do not have
        return ()
    return tuple(_daily_session(mark) for mark in marks)


def _daily_session(mark: Mapping[str, object]) -> DailySession:
    """One journaled mark as the value the metrics module consumes."""
    positions = mark.get("positions_end")
    trades = mark.get("trades")
    return DailySession(
        date=str(mark.get("date", "")),
        equity=_number(mark.get("equity")) or 0.0,
        start_equity=_number(mark.get("start_equity")),
        end_equity=_number(mark.get("end_equity")),
        realized_pnl=_number(mark.get("realized_pnl")),
        orders_submitted=int(_number(mark.get("orders_submitted")) or 0),
        fills=tuple(
            _trade_fill(trade)
            for trade in (trades if isinstance(trades, list) else [])
            if isinstance(trade, Mapping)
        ),
        positions_end={
            str(symbol): float(quantity)
            for symbol, quantity in (positions if isinstance(positions, Mapping) else {}).items()
            if isinstance(quantity, int | float) and not isinstance(quantity, bool)
        },
    )


def _trade_fill(trade: Mapping[str, object]) -> TradeFill:
    """One journaled fill. The ledger stores the *report's* field names, which is deliberate — one
    trade shape is written at CLOSE and read back here, rather than two that could drift."""
    return TradeFill(
        ts=optional_str(trade.get("ts")),
        symbol=str(trade.get("symbol", "")),
        side=str(trade.get("side", "")),
        quantity=_number(trade.get("quantity")) or 0.0,
        price=_number(trade.get("price")) or 0.0,
        fees_usd=_number(trade.get("fees")) or 0.0,
        slippage_bps=_number(trade.get("slippage_bps")),
        champion=optional_str(trade.get("champion")),
        rationale=optional_str(trade.get("rationale")),
    )


# The bar schema the engine trades and researches on, and therefore the one the benchmark is priced
# from. Stated here rather than read from the frozen settings because it is not one of the keys the
# provenance block froze (``inputs.data`` carries provider/dataset/lake_dir); a symbol the lake
# holds under another schema simply yields no bars, and the benchmark degrades to a note.
BAR_SCHEMA = "ohlcv-1m"


def read_benchmark(
    sessions: Sequence[DailySession], inputs: Mapping[str, object] | None
) -> Benchmark:
    """Equal-weight buy-and-hold over the names this run traded, priced from the **shared lake**.

    The fair question a results page has to answer — did the strategy beat simply holding the names
    it traded? — computed with **no vendor call and no new spend**: the bars either are already in
    the workspace lake or they are not, and a symbol that is not is left out with a note rather
    than fetched. That is why this is a read and not an ``ensure_coverage``.

    The roster is derived from the run's own fills, the window from its own session dates, and the
    weights are set at the first session mark and never rebalanced (the convention is stated on the
    record so the comparison is reproducible). Daily levels are the last close of each UTC date; a
    symbol with no bar on a date carries its previous close forward, so one missing session cannot
    silently re-weight the basket.

    Never raises, and never reads a bar outside the run's own session window: a benchmark is
    evidence, and a record write must not fail on a parquet file it could not open.
    """
    symbols = sorted({fill.symbol for session in sessions for fill in session.fills if fill.symbol})
    dates = [session.date for session in sessions if session.date]
    if not symbols or not dates:
        return Benchmark(symbols=(), points=(), note="this run has traded nothing to benchmark")
    window = f"{dates[0]}…{dates[-1]}"
    try:
        closes = _lake_closes(symbols, dates, inputs)
    except Exception:  # pragma: no cover - an unreadable lake is a benchmark we do not have
        closes = {}
    points = _equal_weight_levels(closes, dates)
    if len(points) < 2:
        return Benchmark(
            symbols=tuple(symbols),
            points=(),
            note=(
                f"the shared lake holds no usable bars for {', '.join(symbols)} over {window}, so "
                "this run is not benchmarked — a benchmark is never worth a vendor fetch"
            ),
        )
    return Benchmark(symbols=tuple(symbols), points=tuple(points))


def _lake_closes(
    symbols: Sequence[str], dates: Sequence[str], inputs: Mapping[str, object] | None
) -> dict[str, dict[str, float]]:
    """``{symbol: {date: last close}}`` for the run's window, read straight off the catalog.

    Deferred imports, as everywhere in this module: the run store is written on the core install
    and must stay importable without pulling pandas in behind it. Nothing here writes, fetches or
    creates a directory — a lake that is not there yields nothing.
    """
    import pandas as pd

    from noctis.data.types import SeriesKey

    data = inputs.get("data") if isinstance(inputs, Mapping) else None
    section: Mapping[str, object] = data if isinstance(data, Mapping) else {}
    lake_dir = Path(str(section.get("lake_dir") or "data_lake"))
    dataset = str(section.get("dataset") or "")
    if not dataset or not lake_dir.is_dir():
        return {}
    wanted = set(dates)
    closes: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        path = lake_dir / SeriesKey(dataset, BAR_SCHEMA, symbol).rel_path
        if not path.is_file():
            continue
        frame = pd.read_parquet(path, columns=["ts_event", "close"])
        if frame.empty:
            continue
        stamps = pd.to_datetime(frame["ts_event"], unit="ns", utc=True)
        frame = frame.assign(session=stamps.dt.strftime("%Y-%m-%d"))
        # The run's own window and nothing else: bars from before the account opened or after its
        # last mark are never read into the comparison.
        frame = frame[frame["session"].isin(wanted)]
        if frame.empty:
            continue
        last = frame.groupby("session")["close"].last()
        closes[symbol] = {str(day): float(value) for day, value in last.items()}
    return closes


def _equal_weight_levels(
    closes: Mapping[str, Mapping[str, float]], dates: Sequence[str]
) -> list[tuple[str, float]]:
    """The equal-weight buy-and-hold level per session date, starting at 1.0.

    Weights are set on the first date the lake can price at all, and never rebalanced: each
    symbol's contribution is its own close over its base close, and the level is their mean. A
    symbol with no bar on a later date carries its last known close forward rather than dropping
    out, which would re-weight the basket without saying so.
    """
    priced = [day for day in dates if any(day in series for series in closes.values())]
    if not priced:
        return []
    base_day = priced[0]
    basis = {
        symbol: series[base_day] for symbol, series in closes.items() if series.get(base_day, 0) > 0
    }
    if not basis:
        return []
    carried = dict(basis)
    levels: list[tuple[str, float]] = []
    for day in priced:
        for symbol in basis:
            value = closes[symbol].get(day)
            if value is not None:
                carried[symbol] = value
        levels.append((day, sum(carried[s] / basis[s] for s in basis) / len(basis)))
    return levels


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def read_champions(run_dir: Path | str) -> int | None:
    """How many champions this run currently holds, off its own board — or ``None`` if unreadable.

    The denominator of the record's two per-champion numbers, and read at write time like every
    other cumulative fact (epic D4). The *board*, not the promotion history: a champion that was
    displaced is no longer something this run has, and the board is the run's actual product.

    ``None`` — never ``0`` — when there is no board to read (a fresh run, a pruned one), because
    "no champions yet" and "nobody could look" are different claims and only one of them is a
    number. Capacity is irrelevant to a count, so the registry is opened with none.
    """
    from noctis.champions.registry import ChampionRegistry
    from noctis.config.settings import run_scoped_paths

    try:
        state_dir = run_scoped_paths(Path(run_dir))["state_dir"]
        board = Path(state_dir) / CHAMPIONS_NAME
        if not board.is_file():
            return None
        return len(ChampionRegistry(board, capacity=0).list())
    except Exception:  # pragma: no cover - an unreadable board is evidence we do not have
        return None


def read_strategies(
    run_dir: Path | str, inputs: Mapping[str, object] | None = None
) -> tuple[StrategyArtifact, ...]:
    """Every candidate this run considered, off its own board, journals and strategy tiers (#141).

    **Read, never counted** — the third of the same family as :func:`read_trials` and
    :func:`read_spend`, and for the same reason: all three sources are the *run's*, not the
    process's, so the section is cumulative across every segment by construction and a rewrite
    after a crash cannot double-count it. Three reads, joined on the candidate's name:

    * ``state/champions.json``'s decision history — the only durable trace a *rejected* candidate
      leaves, and since story #141 it carries each decision's structured gate evidence beside its
      prose rationale. The last decision journaled for a name is the decision of record;
    * ``state/experiments/<name>.jsonl`` — how many trials that candidate cost;
    * ``strategies/__tmp/`` and ``strategies/champions/`` — the file itself, for the path, the
      content hash, and (for a champion) the text.

    A name known to any one of them is a candidate. A file with no verdict is ``undecided``, which
    is a real state and not a gap: it is the width of the funnel above the gates.

    **The source policy lives here**, because it is a decision about what to open: a champion's
    file is embedded in full, every other candidate is referenced by a run-relative path plus its
    sha256, and a run created with ``embed_all_sources`` embeds them all. That is what holds a
    fortnight's record to :data:`~noctis.reporting.schema.RECORD_SIZE_BUDGET_BYTES` rather than a
    megabyte; the cost — a rejected candidate's code is readable only while the run's tree survives
    — is stated on the record itself by ``run.state_pruned``.

    ``inputs`` is the run's frozen inputs, the only source of that choice: it is fixed at creation
    like the compute cap beside it, so a resumed segment can never quietly drop the sources an
    earlier one embedded.

    Never raises, and the three reads degrade **independently**: an unparseable board costs the
    record its verdicts, not the candidates whose files are sitting right there.
    """
    run = Path(run_dir)
    decisions = _last_decisions(run)
    files = _strategy_files(run)
    trials = _journaled_trials(run)
    embed_all = _embed_all_sources(inputs)
    return tuple(
        _strategy_artifact(
            run,
            name,
            decision=decisions.get(name),
            located=files.get(name),
            trials=trials.get(name),
            embed_all=embed_all,
        )
        for name in sorted(set(decisions) | set(files) | set(trials))
    )


def _state_dir(run_dir: Path) -> Path:
    from noctis.config.settings import run_scoped_paths

    return Path(run_scoped_paths(run_dir)["state_dir"])


def _last_decisions(run_dir: Path) -> dict[str, Mapping[str, object]]:
    """The decision of record for each candidate: the **last** one the board journaled for it.

    A candidate may be judged more than once (tuned, re-run, re-considered), and a champion may
    later be dropped by a reset. The latest entry is what the run currently says about that name,
    and it carries its own rationale — so a reader is never left reconciling two verdicts.

    Each of the three reads behind the section degrades **on its own**: a board nobody can parse
    costs the record its verdicts, never the candidates whose files and journals are right there.
    Partial evidence is still evidence, and a record that dropped it all because one file was
    corrupt would be the least informative exactly when an operator most needs to look.
    """
    from noctis.champions.registry import ChampionRegistry

    latest: dict[str, Mapping[str, object]] = {}
    try:
        board = _state_dir(run_dir) / CHAMPIONS_NAME
        if not board.is_file():
            return {}
        for entry in ChampionRegistry(board, capacity=0).history:
            name = entry.get("family")
            if isinstance(name, str) and name:
                latest[name] = entry
    except Exception:  # a board we cannot read is a verdict we do not have
        return {}
    return latest


def _strategy_files(run_dir: Path) -> dict[str, tuple[str, Path]]:
    """Each candidate's file and the tier it sits in, later tiers overriding earlier ones."""
    located: dict[str, tuple[str, Path]] = {}
    for tier in STRATEGY_TIER_SUBDIRS:
        directory = run_dir / STRATEGIES_SUBDIR / tier
        if not directory.is_dir():
            continue
        try:
            paths = sorted(directory.glob("*.py"))
        except OSError:  # pragma: no cover - a tier we cannot list
            continue
        for path in paths:
            located[path.stem] = (tier, path)
    return located


def _journaled_trials(run_dir: Path) -> dict[str, int]:
    """How many trials each candidate cost, off its own experiment journal."""
    from noctis.research.journal import ExperimentJournal

    try:
        journal = ExperimentJournal(_state_dir(run_dir))
        return {name: journal.stats(name).n_trials for name in journal.strategies()}
    except Exception:  # pragma: no cover - an unreadable journal is evidence we do not have
        return {}


def _strategy_artifact(
    run_dir: Path,
    name: str,
    *,
    decision: Mapping[str, object] | None,
    located: tuple[str, Path] | None,
    trials: int | None,
    embed_all: bool,
) -> StrategyArtifact:
    """One candidate, assembled from whichever of the three sources knew about it."""
    outcome = UNDECIDED_OUTCOME
    gates: tuple[Mapping[str, object], ...] = ()
    rationale = None
    decided_utc = None
    if decision is not None:
        outcome = PROMOTED_OUTCOME if decision.get("promoted") else REJECTED_OUTCOME
        journaled = decision.get("gates")
        gates = (
            tuple(gate for gate in journaled if isinstance(gate, Mapping))
            if isinstance(journaled, Sequence) and not isinstance(journaled, str | bytes)
            else ()
        )
        rationale = optional_str(decision.get("rationale"))
        decided_utc = _record_stamp(decision.get("at"))
    tier, path = located if located is not None else (None, None)
    # A champion by *either* measure — the tier its file sits in, or the verdict the board
    # journaled — because the one source worth never losing is the run's own product.
    embed = embed_all or tier == CHAMPIONS_TIER or outcome == PROMOTED_OUTCOME
    return StrategyArtifact(
        name=name,
        outcome=outcome,
        tier=tier,
        decided_utc=decided_utc,
        trials=trials,
        gates=gates,
        rationale=rationale,
        source_path=path.relative_to(run_dir).as_posix() if path is not None else None,
        source_sha256=_file_sha256(path) if path is not None else None,
        source=_source_text(path) if path is not None and embed else None,
    )


def _file_sha256(path: Path) -> str | None:
    """The content hash of the file as stored — what a path reference is checkable against."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - a file that vanished between the glob and the read
        return None


def _source_text(path: Path) -> str | None:
    """A strategy file's text, or ``None`` when it cannot be read as the UTF-8 source it is.

    Deliberately strict: a lossy decode would embed text whose hash does not match the reference
    beside it, and a record that quotes something the file does not say is worse than one that
    quotes nothing and points at the path.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):  # pragma: no cover - not source after all
        return None


def _record_stamp(value: object) -> str | None:
    """One foreign ISO stamp in the record's own shape, or ``None`` if it is not a stamp.

    The champion board stamps its history with ``datetime.now(UTC).isoformat()``; the record's
    contract is UTC ISO-8601 with a ``Z``. Converting here — rather than teaching the board a
    second format — keeps one timestamp shape in the record without moving a byte in the arbiter.
    """
    if not isinstance(value, str):
        return None
    try:
        return utc_iso(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _embed_all_sources(inputs: Mapping[str, object] | None) -> bool:
    """Whether this run archives every candidate's source, off its own frozen settings.

    The same tolerant read as the price overrides beside it: a run that froze no configuration, or
    carries something else in that slot, gets the default — champions only.
    """
    node: object = inputs
    for key in ("settings", "resolved", EMBED_ALL_SOURCES_SETTING):
        node = node.get(key) if isinstance(node, Mapping) else None
    return bool(node)


def _artifacts_from(
    prior: Mapping[str, object],
    *,
    run_dir: Path,
    current: EngineIdentity,
    trials: int | None = None,
    spend: tuple[SpendEntry, ...] | None = None,
    pricing_table_version: str | None = None,
    champions: int | None = None,
    strategies: tuple[StrategyArtifact, ...] = (),
    sessions: tuple[DailySession, ...] = (),
    benchmark: Benchmark | None = None,
) -> RunArtifacts:
    """One prior record, parsed back into artifacts. Raises on a shape it cannot read.

    ``trials``, ``spend``, ``champions``, ``strategies`` and the realised ``sessions`` are
    deliberately **not** read back off the record: every one of them is derived from the run's own
    durable artifacts at every write (:func:`read_trials`, :func:`read_spend`,
    :func:`read_champions`, :func:`read_strategies`, :func:`read_sessions`), and a value carried
    forward from a prior write could only ever go stale — or, worse for the equity curve, be
    double-counted by a segment that appended to it.
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
        trials=trials,
        spend=spend,
        pricing_table_version=pricing_table_version,
        champions=champions,
        strategies=strategies,
        sessions=sessions,
        benchmark=benchmark,
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
            # This process's machine, not the run's: like the engine digests beside it, a segment
            # records what actually produced it (story #139).
            environment=dict(environment) if environment else None,
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
            strategies=artifacts.strategies,
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
        touch_lock(self._run_dir, run_id=self._artifacts.run_id, now=now)

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
            spend=changes.get("spend", current.spend),  # type: ignore[arg-type]
            pricing_table_version=changes.get(  # type: ignore[arg-type]
                "pricing_table_version", current.pricing_table_version
            ),
            champions=changes.get("champions", current.champions),  # type: ignore[arg-type]
            strategies=changes.get("strategies", current.strategies),  # type: ignore[arg-type]
            sessions=changes.get("sessions", current.sessions),  # type: ignore[arg-type]
            benchmark=changes.get("benchmark", current.benchmark),  # type: ignore[arg-type]
            state_pruned=current.state_pruned,
        )

    def _flush(self) -> None:
        # Derived at write time, like every cumulative number the record carries: the run's own
        # journals, ledgers and champion board are re-read here rather than handed forward, so a
        # segment that journalled trials and burned tokens all night lands them on disk without
        # anyone tracking a total in memory (epic D4). Spend prices under the run's frozen inputs.
        spend, table_version = read_spend(self._run_dir, self._artifacts.inputs)
        # The equity curve and the trade log are re-read here for the same reason as everything
        # above: the ledger is the run's, not the process's, so the curve a resumed segment
        # publishes is the whole run's without a byte of it having survived the restart in memory.
        sessions = read_sessions(self._run_dir)
        self._replace_artifacts(
            trials=read_trials(self._run_dir),
            spend=spend,
            pricing_table_version=table_version,
            champions=read_champions(self._run_dir),
            strategies=read_strategies(self._run_dir, self._artifacts.inputs),
            sessions=sessions,
            benchmark=read_benchmark(sessions, self._artifacts.inputs),
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
            fingerprint=dict(fingerprint),  # type: ignore[arg-type]
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
        counters=dict(counters) if isinstance(counters, Mapping) else {},  # type: ignore[arg-type]
        # Read back so the run's cumulative research/trading seconds are re-derived from every
        # segment on disk at every write — the totals are never carried in memory across a restart.
        phase_seconds=dict(phases) if isinstance(phases, Mapping) else None,  # type: ignore[arg-type]
        # Carried forward verbatim, like the segment's engine digests: the machine a *past*
        # segment ran on is history, and re-stamping it with whatever this process is running on
        # would be the exact misattribution the per-segment block exists to prevent.
        environment=dict(environment) if isinstance(environment, Mapping) else None,  # type: ignore[arg-type]
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


def _noctis_version() -> str:
    """The package literal — informational beside the engine version, never a comparison key."""
    from importlib import metadata

    try:
        return metadata.version("noctis")
    except Exception:  # not pip-installed (editable/source tree)
        from noctis import __version__

        return __version__
