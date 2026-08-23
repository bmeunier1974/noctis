"""The derived listing roll-up — ``runs/index.json``, and never a source of truth (story #287).

One entry per run, each derived from that run's record **alone**: the index is regenerable from
the records on disk at any moment (:func:`rebuild_index`), and a test pins that a rebuild
reproduces the incrementally-maintained file byte for byte. Anything that could only be learned
from the index would be a second source of truth, free to drift from the records it summarizes —
so the file may be deleted at any time and nothing downstream trusts it more than a record.

It serves the *listing* page in one ``fetch()`` beside the record's own, which is why every entry
carries the same keys with explicit nulls, an unreadable run included: a listing exists to surface
breakage, not to hide it. Like addressing beside it, this module reads records and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from noctis.reporting import schema as schema_module
from noctis.reporting.run_tree.record import (
    RUN_RECORD_NAME,
    optional_str,
    read_record,
    write_json,
)

# The names this module owns — one place each, so nothing spells one by hand.
RUN_INDEX_NAME = "index.json"

# The index's self-declared type, so a consumer can tell the roll-up from a run record at a glance.
RUN_INDEX_KIND = "noctis.run-index"

# What the default listing calls noise: a finished run that never accumulated a minute of runtime
# is a startup failure or a mistyped command, not an experiment. ``--all`` shows them.
SHORT_RUN_S = 60.0


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
    write_json(Path(runs_dir) / RUN_INDEX_NAME, index)


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
        # The listing shares the record's contract version, read off the module for the same
        # reason the record does: an engine whose schema has moved writes both at its own version.
        "schema_version": schema_module.SCHEMA_VERSION,
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
        "label": optional_str(run.get("label")),
        "status": optional_str(run.get("status")),
        "created_utc": optional_str(run.get("created_utc")),
        "last_active_utc": optional_str(run.get("last_active_utc")),
        "segments": len(segments),
        "cumulative_runtime_s": _optional_number(run.get("cumulative_runtime_s")),
        # The compute the run was given, beside the compute it has used: a listing that shows one
        # without the other cannot answer "are these two runs comparable?", which is the question
        # the cap exists to make answerable (100 research hours and 30 are not one experiment).
        "run_limit_hours": _optional_number(run.get("run_limit_hours")),
        "complete": bool(run.get("complete", False)),
        "engine_version": version if isinstance(version, int) else None,
        "comparable_key": optional_str(engine.get("comparable_key")),
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
    if index.get("schema_version") != schema_module.SCHEMA_VERSION:
        return None
    entries = index.get("runs")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        return None
    return index


def _optional_number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
