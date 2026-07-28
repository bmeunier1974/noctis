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
import socket
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from noctis.reporting.run_record import (
    EngineIdentity,
    RecordEvent,
    RunArtifacts,
    SegmentArtifact,
    build,
    mark_interrupted,
    utc_iso,
)
from noctis.reporting.schema import SCHEMA_VERSION

__all__ = [
    "RUNS_SUBDIR",
    "RUN_INDEX_KIND",
    "RUN_INDEX_NAME",
    "RUN_LOCK_NAME",
    "RUN_RECORD_NAME",
    "SHORT_RUN_S",
    "STALE_HEARTBEAT_S",
    "RunLockedError",
    "RunNotFoundError",
    "RunStore",
    "collect",
    "index_entry",
    "open_run",
    "read_record",
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


class RunLockedError(RuntimeError):
    """Another engine holds this run. The one hard refusal — see the module docstring."""


class RunNotFoundError(LookupError):
    """No run answers this address. Raised by :func:`resolve_run_dir`, never by the listing."""


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
) -> RunStore:
    """Open a run for this process: mint or address it, lock it, append a segment, write.

    ``run_id`` defaults to a freshly minted id (identity is minted, never derived); passing one
    addresses that run — which is how a later invocation appends its own segment to the same
    record. Raises :class:`RunLockedError` when another engine holds the run.
    """
    from noctis.observability.debug import new_run_id

    now = clock()
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
    )


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
    prior, note = _read_record(path)
    if prior is not None:
        try:
            return _artifacts_from(prior, run_dir=Path(run_dir), engine=engine)
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
        events=(note,) if note is not None else (),
    )


def _artifacts_from(
    prior: Mapping[str, object], *, run_dir: Path, engine: EngineIdentity
) -> RunArtifacts:
    """One prior record, parsed back into artifacts. Raises on a shape it cannot read."""
    run = prior.get("run")
    if not isinstance(run, Mapping):
        raise TypeError("the 'run' section is missing or is not an object")
    return RunArtifacts(
        run_id=str(run.get("run_id") or run_dir.name),
        created_utc=_optional_str(run.get("created_utc")),
        last_active_utc=_optional_str(run.get("last_active_utc")),
        engine=engine,
        segments=tuple(_segment_from(raw) for raw in _listed(prior, "segments")),
        label=_optional_str(run.get("label")),
        completed_utc=_optional_str(run.get("completed_utc")),
        complete=bool(run.get("complete", False)),
        events=tuple(_event_from(raw) for raw in _listed(prior, "events")),
        errors=tuple(_event_from(raw) for raw in _listed(prior, "errors")),
    )


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


def resolve_run_dir(runs_dir: Path | str, address: str) -> Path:
    """Resolve one run **address** to its directory, or raise :class:`RunNotFoundError`.

    The single place an operator-typed address becomes a path, shared by every verb that
    addresses a run (``run-record`` today, ``--resume`` in story #133). This slice understands
    exactly one form — a **run id**, the identity itself. The later forms (``latest``, a
    ``run.json`` path, ``@label`` through the index) are additional branches here, deliberately
    not implemented yet: an address form invented in two places would eventually resolve two
    different runs from one string.

    A run dir with no readable ``run.json`` still resolves. The record is evidence *about* the
    run, and refusing to address a run because its evidence is corrupt would put the one case an
    operator most needs to inspect out of reach.
    """
    runs = Path(runs_dir)
    candidate = runs / address if _is_run_id(address) else None
    if candidate is not None and candidate.is_dir():
        return candidate
    raise RunNotFoundError(
        f"no run {address!r} under {runs}. `noctis runs --all` lists every run this workspace has."
    )


def _is_run_id(address: str) -> bool:
    """A bare directory name. Anything with a separator is a path form (story #133), not an id —
    and treating it as one would let ``../..`` address its way out of the run tree."""
    if not address or address in (".", ".."):
        return False
    return "/" not in address and "\\" not in address


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

    Two kinds are **never** hidden, whatever their runtime: a run that is still ``running`` (the
    one you are most likely looking for), and a run whose record could not be read (breakage is
    exactly what a listing exists to surface, so tidiness must not swallow it).
    """
    if include_all:
        return list(entries)
    return [entry for entry in entries if not _is_noise(entry)]


def _is_noise(entry: Mapping[str, object]) -> bool:
    if not entry.get("readable", True) or entry.get("status") == "running":
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
    ) -> None:
        self._run_dir = Path(run_dir)
        self._clock = clock
        self._writer = writer
        self._disabled = False
        self._closed = False
        self._complete = False

        now = clock()
        prior = tuple(artifacts.segments)
        self._segment = SegmentArtifact(
            index=len(prior),
            started_utc=utc_iso(now),
            engine=artifacts.engine,
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
            engine=artifacts.engine,
            segments=prior + (self._segment,),
            label=label,
            completed_utc=artifacts.completed_utc,
            complete=False,
            events=tuple(artifacts.events),
            errors=tuple(artifacts.errors),
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

    def checkpoint(self, counters: Mapping[str, int] | None = None) -> None:
        """Rewrite the record and touch the heartbeat. Called at each CLOSE."""
        self._guarded(self._checkpoint, counters)

    def note(self, text: str, *, kind: str = "warn") -> None:
        """Record one event against the open segment (``kind="error"`` files it under errors)."""
        self._guarded(self._note, text, kind)

    def close(self, *, reason: str, counters: Mapping[str, int] | None = None) -> None:
        """Close the segment (stop stamp, duration, reason), write, release the lock. Idempotent.

        The lock is released whatever happened to the record — including after the latch tripped —
        because a lock nobody holds must never be the thing that blocks the next invocation.
        """
        self._guarded(self._close, reason, counters)
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

    def _checkpoint(self, counters: Mapping[str, int] | None) -> None:
        now = self._clock()
        if counters:
            self._replace_segment(counters=dict(counters))
        self._touch(now)
        self._flush()

    def _note(self, text: str, kind: str) -> None:
        self._append(
            RecordEvent(t=utc_iso(self._clock()), kind=kind, text=text, segment=self._segment.index)
        )
        self._flush()

    def _close(self, reason: str, counters: Mapping[str, int] | None) -> None:
        if self._closed:
            return
        now = self._clock()
        self._replace_segment(
            counters=dict(counters) if counters else None,
            stopped_utc=utc_iso(now),
            stopped_reason=reason,
            status="stopped",
        )
        self._touch(now)
        self._complete = True
        self._flush()
        self._closed = True

    # ── internals (no I/O beyond the injected writer) ───────────────────────────────────────

    def _replace_segment(self, **changes: object) -> None:
        """Swap the open segment for an updated copy, in the artifacts as well as the handle."""
        current = self._segment
        updated = SegmentArtifact(
            index=current.index,
            started_utc=current.started_utc,
            engine=current.engine,
            stopped_utc=changes.get("stopped_utc", current.stopped_utc),  # type: ignore[arg-type]
            stopped_reason=changes.get("stopped_reason", current.stopped_reason),  # type: ignore[arg-type]
            status=str(changes.get("status", current.status)),
            argv=current.argv,
            command=current.command,
            resumed=current.resumed,
            counters=changes.get("counters") or current.counters,  # type: ignore[arg-type]
        )
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
            segments=changes.get("segments", current.segments),  # type: ignore[arg-type]
            label=current.label,
            completed_utc=current.completed_utc,
            complete=self._complete,
            events=changes.get("events", current.events),  # type: ignore[arg-type]
            errors=changes.get("errors", current.errors),  # type: ignore[arg-type]
        )

    def _flush(self) -> None:
        self._replace_artifacts()  # re-stamp ``complete`` from the current lifecycle state
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
