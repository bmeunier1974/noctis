"""The run record, built — a **pure** function from collected artifacts to one JSON document.

``run_store.collect(run_dir) -> RunArtifacts`` does every read; this module turns that value into
the document ``run.json`` holds; ``schema.validate(record)`` checks it. The split is the epic's
load-bearing decision, and it is worth stating why: the run record has to be snapshot-tested
against a committed golden, and — once resumption lands — a three-segment run has to be proved
identical to a one-segment run over the same work. Both tests are trivial when the builder is a
function over a value you can write by hand, and both are slow, flaky integration tests when the
builder reads the disk itself.

So **nothing here touches I/O, a clock or the configuration**. Every timestamp arrives as data
inside :class:`RunArtifacts`, already formatted by :func:`utc_iso` (which takes the moment as an
argument and never asks for it). A test enforces that structurally, by AST, rather than trusting
the convention to survive.

Two honesty rules the builder implements rather than documents:

* **``interrupted`` is observed, never guessed.** :func:`build` reports exactly the segment
  statuses it was handed; a segment killed mid-flight only becomes ``interrupted`` when the run is
  next unsealed and :func:`mark_interrupted` says so. A writer that guessed at write time would
  have to guess wrong at least once — at the moment of the crash, when nothing is there to write.
* **Every cap that bites is named.** Bulky lists are bounded by ``schema``'s caps, and a bounded
  list writes ``truncated: {"events": {"kept": N, "total": M}}``. Segments are never capped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from noctis.reporting.schema import EVENT_CAP, KIND, SCHEMA_VERSION

__all__ = [
    "EngineIdentity",
    "RecordEvent",
    "RunArtifacts",
    "SegmentArtifact",
    "build",
    "mark_interrupted",
    "utc_iso",
]


def utc_iso(moment: datetime) -> str:
    """``2026-07-27T14:22:33.418Z`` — UTC ISO-8601, milliseconds, an explicit ``Z``.

    The record's one timestamp shape (schema convention). The moment is an **argument**: this
    module never reads a clock, so a caller's injected clock is the only source of time anywhere
    in the record. An aware datetime is normalized to UTC so the ``Z`` stays honest; a naive one
    is taken as UTC as-is.
    """
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class RecordEvent:
    """One notable thing that happened to the run: a lock steal, a degradation, an error.

    Deliberately flat and small — the record is evidence a website renders, not a log. Rich
    per-event detail stays in the ``--debug`` QA tree, which the record references. There is one
    event stream for the whole run and each entry names the ``segment`` it happened in, so a
    reader gets a single timeline without having to stitch per-segment lists together.
    """

    t: str | None
    kind: str  # info | warn | error
    text: str
    segment: int | None = None

    def as_dict(self) -> dict:
        return {"t": self.t, "segment": self.segment, "kind": self.kind, "text": self.text}


@dataclass(frozen=True)
class EngineIdentity:
    """What produced these numbers: the declared version plus the computed per-component digests.

    Both, because neither works alone — see ``observability/engine_id.py``. Carried on the run
    *and* on every segment, since a run resumed after a code change ran two engines.
    """

    engine_version: int
    fingerprint: Mapping[str, str | None]
    comparable_key: str
    noctis_version: str


@dataclass(frozen=True)
class SegmentArtifact:
    """One process invocation. The stop-each-morning / resume-each-night pattern makes one a night.

    Its own counters, because per-segment throughput is only comparable when it is attributed to
    the process that produced it.
    """

    index: int
    started_utc: str
    engine: EngineIdentity | None = None
    stopped_utc: str | None = None
    stopped_reason: str | None = None
    status: str = "running"
    argv: Sequence[str] = ()
    command: str = "run"
    resumed: bool = False
    counters: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RunArtifacts:
    """Everything the run tree and this engine know about one run — the builder's only input.

    ``complete`` is false whenever a segment is open, the process was killed, or the writer
    latched off after a failure: a partial record must never pass for a whole one.
    """

    run_id: str
    created_utc: str | None
    last_active_utc: str | None
    engine: EngineIdentity
    segments: Sequence[SegmentArtifact] = ()
    label: str | None = None
    completed_utc: str | None = None
    complete: bool = False
    events: Sequence[RecordEvent] = ()
    errors: Sequence[RecordEvent] = ()


def mark_interrupted(artifacts: RunArtifacts) -> RunArtifacts:
    """Return the artifacts with every unclosed segment marked ``interrupted``.

    Called when a run is **opened**, which is the only moment the observation can honestly be
    made: a segment carrying a start stamp and no stop stamp belongs to a process that is no
    longer here. Pure and idempotent — a cleanly closed segment is untouched.
    """
    segments = tuple(
        segment if segment.stopped_utc is not None else _replace_status(segment, "interrupted")
        for segment in artifacts.segments
    )
    return RunArtifacts(
        run_id=artifacts.run_id,
        created_utc=artifacts.created_utc,
        last_active_utc=artifacts.last_active_utc,
        engine=artifacts.engine,
        segments=segments,
        label=artifacts.label,
        completed_utc=artifacts.completed_utc,
        complete=artifacts.complete,
        events=tuple(artifacts.events),
        errors=tuple(artifacts.errors),
    )


def build(artifacts: RunArtifacts) -> dict:
    """Assemble the run record. Pure: same artifacts in, byte-identical document out."""
    events, events_note = _capped(artifacts.events, EVENT_CAP)
    errors, errors_note = _capped(artifacts.errors, EVENT_CAP)
    truncated: dict[str, dict[str, int]] = {}
    if events_note is not None:
        truncated["events"] = events_note
    if errors_note is not None:
        truncated["errors"] = errors_note

    segments = [_segment(segment) for segment in artifacts.segments]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run": {
            "run_id": artifacts.run_id,
            "label": artifacts.label,
            "status": _status(artifacts),
            "created_utc": artifacts.created_utc,
            "last_active_utc": artifacts.last_active_utc,
            "completed_utc": artifacts.completed_utc,
            "cumulative_runtime_s": _cumulative_runtime_s(segments),
            "complete": bool(artifacts.complete),
            "truncated": truncated,
        },
        "segments": segments,
        "engine": _engine(artifacts),
        "events": [event.as_dict() for event in events],
        "errors": [error.as_dict() for error in errors],
    }


def _status(artifacts: RunArtifacts) -> str:
    """The run's lifecycle status, **derived** from the segments rather than carried beside them.

    Derived, so the two can never disagree: a run is ``running`` while its last segment is open,
    ``interrupted`` while its last segment is one a kill left behind, ``stopped`` (resumable) once
    it closed cleanly, and ``completed`` only when something deliberately sealed it.
    """
    if artifacts.completed_utc is not None:
        return "completed"
    if not artifacts.segments:
        return "stopped"
    last = artifacts.segments[-1].status
    return last if last in ("running", "interrupted") else "stopped"


def _segment(segment: SegmentArtifact) -> dict:
    engine = segment.engine
    return {
        "index": segment.index,
        "started_utc": segment.started_utc,
        "stopped_utc": segment.stopped_utc,
        "duration_s": _duration_s(segment.started_utc, segment.stopped_utc),
        "stopped_reason": segment.stopped_reason,
        "status": segment.status,
        "argv": list(segment.argv),
        "command": segment.command,
        "resumed": bool(segment.resumed),
        "counters": dict(segment.counters),
        "engine_version": engine.engine_version if engine is not None else None,
        "engine_fingerprint": dict(engine.fingerprint) if engine is not None else None,
    }


def _engine(artifacts: RunArtifacts) -> dict:
    """The run-level engine identity, plus whether any segment ran a different engine."""
    engine = artifacts.engine
    fingerprints = [
        dict(segment.engine.fingerprint)
        for segment in artifacts.segments
        if segment.engine is not None
    ]
    mixed = any(digests != dict(engine.fingerprint) for digests in fingerprints)
    return {
        "engine_version": engine.engine_version,
        "noctis_version": engine.noctis_version,
        "fingerprint": dict(engine.fingerprint),
        "comparable_key": engine.comparable_key,
        "mixed_engine": mixed,
    }


def _cumulative_runtime_s(segments: Sequence[Mapping[str, object]]) -> float:
    """Recomputed from the segments at every write, never incremented — so a rewrite after a
    crash is idempotent and three short segments total exactly what one long one would."""
    total = 0.0
    for segment in segments:
        duration = segment.get("duration_s")
        if isinstance(duration, int | float):
            total += float(duration)
    return round(total, 3)


def _duration_s(started: str | None, stopped: str | None) -> float | None:
    """Seconds between two record stamps, or ``None`` while a segment is still open."""
    if started is None or stopped is None:
        return None
    return round((_parse(stopped) - _parse(started)).total_seconds(), 3)


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _capped(
    events: Sequence[RecordEvent], cap: int
) -> tuple[Sequence[RecordEvent], dict[str, int] | None]:
    """Bound one list, and say so when the bound bit.

    The **earliest** entries are kept: they are the ones that explain how a run reached its
    state (a stolen lock, a degraded seam, the first error), and the note names the total so a
    reader always knows what is not here. Silent truncation is forbidden.
    """
    total = len(events)
    if total <= cap:
        return events, None
    return events[:cap], {"kept": cap, "total": total}


def _replace_status(segment: SegmentArtifact, status: str) -> SegmentArtifact:
    return SegmentArtifact(
        index=segment.index,
        started_utc=segment.started_utc,
        engine=segment.engine,
        stopped_utc=segment.stopped_utc,
        stopped_reason=segment.stopped_reason,
        status=status,
        argv=tuple(segment.argv),
        command=segment.command,
        resumed=segment.resumed,
        counters=dict(segment.counters),
    )
