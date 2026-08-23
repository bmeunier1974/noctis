"""Shared fixtures for the run-tree tests (epic #284): a hand-moved clock, a record written
through the pure builder, and the one ``run.lock`` writer in the suite.

The narrow modules of ``reporting/run_tree`` read **files**, not stores, so a test of one writes
the files it reads: a ``run.json`` built in memory through the pure
:func:`noctis.reporting.run_record.build` and written with :func:`noctis.reporting.run_tree.write`
— the way the golden-record tests already build records — and a ``run.lock`` written by the lock
module's own writer. No engine fingerprint is computed, no collector runs and no lock is taken to
test an address.

``hold_lock`` is deliberately derived from :func:`noctis.reporting.run_tree.lock.touch_lock`
rather than hand-rolling the document a second time: three test files used to spell the lock's
keys and its hostname hash themselves, which is three copies free to drift from the protocol they
are meant to stand in for.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from noctis.reporting.run_record import (
    EngineIdentity,
    RunArtifacts,
    SegmentArtifact,
    build,
)
from noctis.reporting.run_tree import write
from noctis.reporting.run_tree.lock import RUN_LOCK_NAME, touch_lock

# The moment every run-tree fixture starts at, and the two stamps derived from it: one hour of
# runtime, so a written run looks like a run that did some work.
START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)
CREATED_UTC = "2026-07-27T14:22:33.418Z"
LAST_ACTIVE_UTC = "2026-07-27T15:22:33.418Z"

# A frozen engine identity: a literal, because computing a real fingerprint is exactly the cost
# these fixtures exist to avoid. The digests are opaque to every module that reads a record.
ENGINE = EngineIdentity(
    engine_version=1,
    fingerprint={
        "gates": "f63d47b7b9604ab1",
        "backtest": "3ba3e0bf1c97134f",
        "research": "4baf9dea0c82c8cc",
        "prompts": "14eb169506a6b5aa",
        "profiles": "6803b9d26c63d6ae",
        "seeds": "4826fe7224641eb4",
        "memory_seed": "3337fa2cbf896932",
        "schema": None,
    },
    comparable_key="1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe",
    noctis_version="0.1.0",
)

# The statuses a record *derives* from its own segments: an open segment is a running run, a
# segment a kill left behind is an interrupted one, and a sealed run carries a completed stamp.
_OPEN_STATUSES = ("running", "interrupted")


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock read reaches the store."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def write_run(
    runs_dir: Path | str,
    run_id: str,
    *,
    label: str | None = None,
    status: str = "stopped",
    created_utc: str = CREATED_UTC,
    last_active_utc: str = LAST_ACTIVE_UTC,
    complete: bool = False,
    comparable_key: str | None = None,
) -> Path:
    """Write one run's ``run.json`` the way a store would, and return its run dir.

    ``status`` is the record's own derived lifecycle word — ``stopped`` (one closed segment),
    ``running`` / ``interrupted`` (one segment with no stop stamp) or ``completed`` (sealed with
    ``completed_utc``) — so a fixture asks for the state it means instead of hand-patching the
    record afterwards.
    """
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    engine = ENGINE if comparable_key is None else replace(ENGINE, comparable_key=comparable_key)
    closed = status not in _OPEN_STATUSES
    segment = SegmentArtifact(
        index=0,
        started_utc=created_utc,
        engine=engine,
        stopped_utc=last_active_utc if closed else None,
        stopped_reason="time_limit" if closed else None,
        status="stopped" if closed else status,
        argv=("run", "-v"),
        command="run",
        counters={"cycles": 1},
    )
    artifacts = RunArtifacts(
        run_id=run_id,
        created_utc=created_utc,
        last_active_utc=last_active_utc,
        engine=engine,
        current_engine=engine,
        segments=(segment,),
        label=label,
        completed_utc=last_active_utc if status == "completed" else None,
        complete=complete,
    )
    write(run_dir, build(artifacts))
    return run_dir


def hold_lock(
    run_dir: Path | str,
    *,
    run_id: str,
    pid: int | None = None,
    hostname_hash: str | None = None,
    heartbeat_utc: str | None = None,
) -> dict:
    """Hold ``run.lock`` on a run — the exact document a live store writes — and return it.

    The base document comes from the lock module's own writer, so a test can never stand in for
    the protocol with a spelling of its own; the three keyword arguments override the fields a
    staleness scenario chooses (a lock from another host, a dead pid, a cold heartbeat).
    """
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    touch_lock(path, run_id=run_id, now=datetime.now(UTC))
    lock_path = path / RUN_LOCK_NAME
    held = json.loads(lock_path.read_text(encoding="utf-8"))
    overrides = {"pid": pid, "hostname_hash": hostname_hash, "heartbeat_utc": heartbeat_utc}
    chosen = {key: value for key, value in overrides.items() if value is not None}
    if chosen:
        held.update(chosen)
        lock_path.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    return held
