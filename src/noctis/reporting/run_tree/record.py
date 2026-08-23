"""The run tree's one narrow read and its one atomic write (story #286, epic #284).

The bottom of the package: the two names that *are* the layout (``runs/<run_id>/run.json``), the
read every other module goes through, the write every record goes out of, and the two small
helpers they share. It imports nothing from :mod:`noctis.reporting.run_tree` — which is the point.
A caller that only needs to *read* a run record (``eval.decide_miner``) imports this module and
gets ten lines of JSON handling, not the store, the lock and the collectors behind them.

**Writes are synchronous and atomic**: a temp file beside the target plus ``os.replace``, so a
kill mid-write leaves the previous document intact and never a half-written one. **Reads are
total**: a missing, unparseable or foreign record comes back as a *reason a caller can show*,
because a corrupt reporting file must never take a listing — or a run — down with it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

# The run tree's names — one place, so nothing spells them by hand.
RUNS_SUBDIR = "runs"
RUN_RECORD_NAME = "run.json"


def read_record(run_dir: Path | str) -> tuple[dict | None, str | None]:
    """One run's record, or ``(None, why)`` when there is not a readable one.

    The reading half of "a broken record is evidence, not a crash": the caller gets a reason it
    can *show* — no record yet, unreadable JSON, a foreign shape — instead of an exception that
    would take a whole listing down with one bad file.
    """
    return _record_at(Path(run_dir) / RUN_RECORD_NAME)


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


def write(run_dir: Path | str, record: Mapping[str, object]) -> None:
    """Write ``run.json`` atomically: a temp file beside it, then ``os.replace``.

    ``os.replace`` is atomic on every platform Noctis supports, so a reader (or a kill) sees
    either the whole previous record or the whole new one — never a half-written file. The temp
    file is removed on failure so a crashed write leaves no litter beside the record.
    """
    write_json(Path(run_dir) / RUN_RECORD_NAME, record)


def write_json(target: Path, document: Mapping[str, object]) -> None:
    """One atomic JSON write, shared by the record and the index — same discipline, one copy."""
    tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def optional_str(value: object) -> str | None:
    return None if value is None else str(value)
