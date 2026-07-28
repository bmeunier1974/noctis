"""The run record's schema — its version, its caps, and a pure validator (story #129, epic #126).

One run, one self-describing file: ``workspace/runs/<run_id>/run.json``. This module holds the
contract that file promises to anything reading it (a website, a later Noctis, an operator with
``jq``), and nothing else — no I/O, no clock, no config, not even the record's construction. It is
deliberately the smallest module in the epic, because it is also the one the engine fingerprint
tracks as the ``schema`` component (``observability/engine_id.py``): changing what is recorded
should move a digest, and only that.

**Versioning is additive-only.** :data:`SCHEMA_VERSION` is 1. New fields may be added at any time;
an existing field never changes meaning or type, and a reader ignores keys it does not know. That
is what lets a record written tonight still be read by the Noctis that resumes the run in a month.
A breaking change bumps the version and upgrades on read — it never silently repurposes a key.

**Two conventions are part of the contract, not style.** Units are explicit in the field name
(``_usd``, ``_pct``, ``_bps``, ``_s``, ``_bytes``), and a known-absent value is an explicit
``null`` rather than an omitted key — so a consumer can tell "not applicable" from "this schema
version did not have it". :func:`validate` enforces both where it can.

**Caps are honest.** Bulky lists are bounded (:data:`TRADE_CAP`, :data:`EVENT_CAP`) and every cap
that bites writes a ``truncated`` note carrying kept/total counts. Silent truncation is forbidden:
a truncated record must never pass for a complete one. Segments and sessions are deliberately
uncapped — they are the run's spine, and losing one would make every derived total a lie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "EVENT_CAP",
    "KIND",
    "REQUIRED_SECTIONS",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "SEGMENT_STATUSES",
    "TRADE_CAP",
    "validate",
]

# The record contract's version. Additive-only: bump this only for a breaking change.
SCHEMA_VERSION = 1

# The record's self-declared type, so a consumer can tell a run record from any other JSON.
KIND = "noctis.run"

# The top-level sections every record carries. Later stories in the epic add sections
# (``inputs``, ``research``, ``strategies``, ``sessions``, ``performance``, ``assumptions``);
# additive-only means this set may grow, never shrink.
REQUIRED_SECTIONS = ("run", "segments", "engine", "events", "errors")

# The run's lifecycle. ``interrupted`` is observed on the next open (a segment with a start stamp
# and no stop stamp), never guessed at write time. ``completed`` is terminal.
RUN_STATUSES = ("running", "stopped", "interrupted", "completed")

# One segment = one process invocation. It is open (``running``), closed cleanly (``stopped``), or
# was killed mid-flight and found that way on the next open (``interrupted``).
SEGMENT_STATUSES = ("running", "stopped", "interrupted")

# Bounds on the two lists that grow without limit in a long run. A record is meant to be fetched
# whole by a website, so these keep a multi-week run in the tens of kilobytes.
TRADE_CAP = 5_000
EVENT_CAP = 2_000

# The keys the ``run`` section always carries — presence is the contract, ``null`` is a value.
_RUN_KEYS = (
    "run_id",
    "label",
    "status",
    "created_utc",
    "last_active_utc",
    "completed_utc",
    "cumulative_runtime_s",
    "complete",
    "truncated",
)

_SEGMENT_KEYS = (
    "index",
    "started_utc",
    "stopped_utc",
    "duration_s",
    "stopped_reason",
    "status",
    "argv",
    "command",
    "resumed",
    "counters",
    "engine_version",
    "engine_fingerprint",
)

_ENGINE_KEYS = (
    "engine_version",
    "noctis_version",
    "fingerprint",
    "comparable_key",
    "mixed_engine",
)

_EVENT_KEYS = ("t", "segment", "kind", "text")


def validate(record: Mapping[str, object]) -> list[str]:
    """Check one record against the schema; return the problems, one line each (``[]`` = valid).

    A **list, not an exception**: a validator that raises reports one problem per run, and the
    caller who most needs this — an operator asking "is this record readable?" — wants the whole
    list at once. Nothing here reads a file or a clock, so a record can be checked in memory
    exactly as it will be checked on disk.
    """
    problems: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version: expected {SCHEMA_VERSION}, found {record.get('schema_version')!r}"
        )
    if record.get("kind") != KIND:
        problems.append(f"kind: expected {KIND!r}, found {record.get('kind')!r}")
    for section in REQUIRED_SECTIONS:
        if section not in record:
            problems.append(f"{section}: section is missing")

    run = record.get("run")
    if isinstance(run, Mapping):
        problems += _check_keys("run", run, _RUN_KEYS)
        problems += _check_status("run.status", run.get("status"), RUN_STATUSES)
        problems += _check_stamp("run.created_utc", run.get("created_utc"))
        problems += _check_stamp("run.last_active_utc", run.get("last_active_utc"))
        problems += _check_stamp("run.completed_utc", run.get("completed_utc"))
    elif "run" in record:
        problems.append("run: section must be an object")

    problems += _check_segments(record.get("segments"))

    engine = record.get("engine")
    if isinstance(engine, Mapping):
        problems += _check_keys("engine", engine, _ENGINE_KEYS)
    elif "engine" in record:
        problems.append("engine: section must be an object")

    for name in ("events", "errors"):
        problems += _check_events(name, record.get(name))
    return problems


def _check_segments(segments: object) -> list[str]:
    """Segments are an append-only sequence, indexed from 0 with no gaps — the run's spine."""
    if segments is None:
        return []
    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        return ["segments: must be a list"]
    problems: list[str] = []
    for position, segment in enumerate(segments):
        label = f"segments[{position}]"
        if not isinstance(segment, Mapping):
            problems.append(f"{label}: must be an object")
            continue
        problems += _check_keys(label, segment, _SEGMENT_KEYS)
        if segment.get("index") != position:
            problems.append(
                f"{label}.index: expected {position}, found {segment.get('index')!r} "
                "(segments are append-only and indexed from 0)"
            )
        problems += _check_status(f"{label}.status", segment.get("status"), SEGMENT_STATUSES)
        problems += _check_stamp(f"{label}.started_utc", segment.get("started_utc"))
        problems += _check_stamp(f"{label}.stopped_utc", segment.get("stopped_utc"))
    return problems


def _check_events(label: str, events: object) -> list[str]:
    if events is None:
        return []
    if not isinstance(events, Sequence) or isinstance(events, str | bytes):
        return [f"{label}: must be a list"]
    problems: list[str] = []
    for position, event in enumerate(events):
        if not isinstance(event, Mapping):
            problems.append(f"{label}[{position}]: must be an object")
            continue
        problems += _check_keys(f"{label}[{position}]", event, _EVENT_KEYS)
        problems += _check_stamp(f"{label}[{position}].t", event.get("t"))
    return problems


def _check_keys(label: str, section: Mapping[str, object], keys: Sequence[str]) -> list[str]:
    """Presence, not truthiness: an absent value is an explicit ``null``, never a missing key."""
    return [
        f"{label}.{key}: key is missing (absent values are explicit nulls)"
        for key in keys
        if key not in section
    ]


def _check_status(label: str, value: object, allowed: Sequence[str]) -> list[str]:
    if value in allowed:
        return []
    return [f"{label}: {value!r} is not one of {', '.join(allowed)}"]


def _check_stamp(label: str, value: object) -> list[str]:
    """UTC ISO-8601 with a ``Z`` marker, or ``null``. Anything else is ambiguous by timezone."""
    if value is None:
        return []
    if isinstance(value, str) and value.endswith("Z") and "T" in value:
        return []
    return [f"{label}: {value!r} is not a UTC ISO-8601 timestamp ending in 'Z'"]
