"""The run store — the one module that touches the run tree (story #129, epic #126).

Everything asserted here is external: which files exist under ``workspace/runs/<run_id>/``, what
``run.json`` parses to, what ``run.lock`` says, which warnings were logged. Never a call sequence
and never private state — the store is a black box driven by an injected clock and an injectable
writer, exactly as ``tests/test_recorder.py`` drives the QA recorder.

Two contracts get particular attention because they are the ones that bite in production:

* **The lock is the one fatal failure.** Two engines writing one run is corruption, not
  degradation, so a live lock is a hard refusal. A *stale* lock — a dead pid on this host, or a
  heartbeat gone cold from a host we cannot check — is stolen, loudly, with an event in the record.
* **Everything else is latched, never fatal.** A failing writer logs exactly one warning, disables
  itself, and leaves an honestly incomplete record behind. A reporting artifact must never take
  down a multi-week run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.cli import app
from noctis.observability.debug import RUN_ID_RE
from noctis.observability.engine_id import ENGINE_VERSION
from noctis.reporting import schema
from noctis.reporting.run_store import (
    RUN_INDEX_KIND,
    RUN_INDEX_NAME,
    RUN_LOCK_NAME,
    RUN_RECORD_NAME,
    STALE_HEARTBEAT_S,
    RunLockedError,
    RunNotFoundError,
    index_entry,
    open_run,
    rebuild_index,
    resolve_run_dir,
    write,
    write_index,
)

runner = CliRunner()

START = datetime(2026, 7, 27, 14, 22, 33, 418000, tzinfo=UTC)


class FakeClock:
    """A deterministic clock the test moves by hand — no wall-clock read reaches the store."""

    def __init__(self, start: datetime = START) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> FakeClock:
        self.now = self.now + timedelta(seconds=seconds)
        return self


def _open(runs_dir: Path, clock: FakeClock, **kwargs):
    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs_dir, clock=clock, **kwargs)


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _lock(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_LOCK_NAME).read_text())


def _write_lock(run_dir: Path, **fields) -> None:
    lock = {
        "run_id": run_dir.name,
        "pid": os.getpid(),
        "hostname_hash": _our_host_hash(),
        "started_utc": "2026-07-27T14:22:33.418Z",
        "heartbeat_utc": "2026-07-27T14:22:33.418Z",
    }
    lock.update(fields)
    (run_dir / RUN_LOCK_NAME).write_text(json.dumps(lock))


def _our_host_hash() -> str:
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]


def _dead_pid() -> int:
    """A pid that is provably not running: a child we started and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


# ── the run tree ───────────────────────────────────────────────────────────────────────────


def test_opening_a_run_creates_its_tree_with_a_record_and_a_lock(tmp_path):
    store = _open(tmp_path / "runs", FakeClock())

    assert RUN_ID_RE.match(store.run_id)
    assert store.run_dir == tmp_path / "runs" / store.run_id
    assert sorted(p.name for p in store.run_dir.iterdir()) == sorted(
        [RUN_LOCK_NAME, RUN_RECORD_NAME]
    )


def test_every_open_mints_a_new_run_even_with_byte_identical_inputs(tmp_path):
    """Identity is minted, never derived: two identical invocations are two runs."""
    runs = tmp_path / "runs"
    first = _open(runs, FakeClock())
    first.close(reason="stopped")
    second = _open(runs, FakeClock())
    second.close(reason="stopped")

    assert first.run_id != second.run_id
    # The derived roll-up lives beside the run trees, one per workspace, never inside a run.
    assert sorted(p.name for p in runs.iterdir()) == sorted(
        [first.run_id, second.run_id, RUN_INDEX_NAME]
    )


def test_the_written_record_declares_schema_version_one_kind_and_validates(tmp_path):
    store = _open(tmp_path / "runs", FakeClock())

    record = _record(store.run_dir)

    assert record["schema_version"] == 1
    assert record["kind"] == "noctis.run"
    assert schema.validate(record) == []
    assert record["run"]["run_id"] == store.run_id
    assert record["run"]["created_utc"] == "2026-07-27T14:22:33.418Z"


def test_the_record_carries_this_engines_identity_from_the_first_write(tmp_path):
    """Every record ever written carries an engine identity — one without it could never be
    retrofitted into a comparison bucket."""
    store = _open(tmp_path / "runs", FakeClock(), election_metric="sortino")

    engine = _record(store.run_dir)["engine"]

    assert engine["engine_version"] == ENGINE_VERSION
    assert engine["fingerprint"]["gates"] is not None  # computed off this checkout
    assert engine["comparable_key"].endswith("|sortino")
    assert engine["mixed_engine"] is False


# ── segments: one per process invocation ───────────────────────────────────────────────────


def test_the_open_segment_is_running_with_its_argv_and_no_stop_stamp(tmp_path):
    store = _open(tmp_path / "runs", FakeClock(), argv=["run", "--verbose"])

    segment = _record(store.run_dir)["segments"][0]

    assert segment["index"] == 0
    assert segment["status"] == "running"
    assert segment["argv"] == ["run", "--verbose"]
    assert segment["command"] == "run"
    assert segment["resumed"] is False
    assert segment["stopped_utc"] is None and segment["duration_s"] is None
    assert _record(store.run_dir)["run"]["status"] == "running"


def test_closing_a_segment_stamps_stop_reason_duration_status_and_counters(tmp_path):
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock)

    clock.advance(3600)
    store.close(reason="time_limit", counters={"cycles": 2, "trades": 7})

    record = _record(store.run_dir)
    segment = record["segments"][0]
    assert segment["stopped_utc"] == "2026-07-27T15:22:33.418Z"
    assert segment["duration_s"] == 3600.0
    assert segment["stopped_reason"] == "time_limit"
    assert segment["status"] == "stopped"
    assert segment["counters"] == {"cycles": 2, "trades": 7}
    assert record["run"]["status"] == "stopped"
    assert record["run"]["cumulative_runtime_s"] == 3600.0
    assert record["run"]["last_active_utc"] == "2026-07-27T15:22:33.418Z"


def test_a_second_invocation_appends_its_own_segment_with_its_own_counters(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    clock.advance(3600)
    first.close(reason="time_limit", counters={"cycles": 1})

    clock.advance(36000)
    second = _open(runs, clock, run_id=first.run_id, argv=["run"])
    clock.advance(1800)
    second.close(reason="stop_requested", counters={"cycles": 3})

    record = _record(second.run_dir)
    assert [s["index"] for s in record["segments"]] == [0, 1]
    assert [s["resumed"] for s in record["segments"]] == [False, True]
    assert [s["counters"]["cycles"] for s in record["segments"]] == [1, 3]
    assert [s["stopped_reason"] for s in record["segments"]] == ["time_limit", "stop_requested"]
    assert record["run"]["cumulative_runtime_s"] == 5400.0
    assert record["run"]["created_utc"] == "2026-07-27T14:22:33.418Z"  # the FIRST segment's
    assert schema.validate(record) == []


def test_a_run_killed_mid_segment_is_marked_interrupted_on_the_next_open(tmp_path, caplog):
    runs = tmp_path / "runs"
    clock = FakeClock()
    killed = _open(runs, clock)
    # At write time the record says exactly what was true: a segment is open. Nothing is guessed.
    assert _record(killed.run_dir)["segments"][0]["status"] == "running"
    # The kill: the process is gone, its lock left behind holding a pid that no longer exists.
    _write_lock(killed.run_dir, pid=_dead_pid())

    clock.advance(7200)
    with caplog.at_level(logging.WARNING):
        reopened = _open(runs, clock, run_id=killed.run_id)

    record = _record(reopened.run_dir)
    assert record["segments"][0]["status"] == "interrupted"
    assert record["segments"][0]["stopped_utc"] is None
    assert record["segments"][1]["status"] == "running"
    assert record["run"]["status"] == "running"  # the new segment is live again
    assert schema.validate(record) == []


def test_an_interrupted_run_left_unopened_still_reads_as_interrupted(tmp_path):
    """The detection happens on the next open, and the reopened record keeps saying so."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    killed = _open(runs, clock)
    _write_lock(killed.run_dir, pid=_dead_pid())

    reopened = _open(runs, clock, run_id=killed.run_id)
    reopened.close(reason="stopped")

    record = _record(reopened.run_dir)
    assert [s["status"] for s in record["segments"]] == ["interrupted", "stopped"]
    assert record["run"]["status"] == "stopped"


# ── the lock: the one place a failure is fatal ─────────────────────────────────────────────


def test_a_second_process_refuses_to_open_a_live_locked_run(tmp_path):
    runs = tmp_path / "runs"
    live = _open(runs, FakeClock())

    with pytest.raises(RunLockedError) as excinfo:
        _open(runs, FakeClock(), run_id=live.run_id)

    message = str(excinfo.value)
    assert live.run_id in message
    assert str(os.getpid()) in message  # names the holder
    assert "run.lock" in message  # …and where the evidence is


def test_a_live_lock_from_another_host_is_never_stolen(tmp_path):
    """A pid on another host tells you nothing, so only a cold heartbeat may condemn it."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    live = _open(runs, clock)
    live.close(reason="stopped")
    _write_lock(
        live.run_dir,
        hostname_hash="0" * 12,
        pid=999_999,
        heartbeat_utc="2026-07-27T14:22:33.418Z",
    )

    with pytest.raises(RunLockedError):
        _open(runs, clock, run_id=live.run_id)


def test_a_dead_pid_on_this_host_is_a_stale_lock_stolen_with_a_warning_and_an_event(
    tmp_path, caplog
):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    first.close(reason="stopped")
    dead = _dead_pid()
    _write_lock(first.run_dir, pid=dead)

    with caplog.at_level(logging.WARNING):
        store = _open(runs, clock, run_id=first.run_id)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(dead) in warnings[0].getMessage()
    events = _record(store.run_dir)["events"]
    assert [e["kind"] for e in events] == ["warn"]
    assert str(dead) in events[0]["text"]
    assert events[0]["segment"] == 1
    assert _lock(store.run_dir)["pid"] == os.getpid()  # …and the lock is ours now


def test_a_cold_heartbeat_is_a_stale_lock_stolen_with_a_warning_and_an_event(tmp_path, caplog):
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock)
    first.close(reason="stopped")
    cold = clock.now - timedelta(seconds=STALE_HEARTBEAT_S + 60)
    _write_lock(
        first.run_dir,
        hostname_hash="0" * 12,
        pid=999_999,
        heartbeat_utc=f"{cold:%Y-%m-%dT%H:%M:%S}.000Z",
    )

    with caplog.at_level(logging.WARNING):
        store = _open(runs, clock, run_id=first.run_id)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "heartbeat" in warnings[0].getMessage()
    assert "heartbeat" in _record(store.run_dir)["events"][0]["text"]


def test_closing_the_run_releases_the_lock(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    store.close(reason="stopped")

    assert not (store.run_dir / RUN_LOCK_NAME).exists()
    reopened = _open(runs, clock, run_id=store.run_id)  # no steal, no refusal
    assert _record(reopened.run_dir)["events"] == []


def test_the_lock_carries_a_hashed_hostname_never_the_raw_one(tmp_path):
    """The record and its lock are meant to be shareable: a machine name is not."""
    store = _open(tmp_path / "runs", FakeClock())

    lock = _lock(store.run_dir)

    assert lock["hostname_hash"] == _our_host_hash()
    assert len(lock["hostname_hash"]) == 12
    assert socket.gethostname() not in (store.run_dir / RUN_LOCK_NAME).read_text()
    assert lock["pid"] == os.getpid()
    assert lock["heartbeat_utc"].endswith("Z")


def test_the_heartbeat_is_touched_at_each_checkpoint(tmp_path):
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock)
    assert _lock(store.run_dir)["heartbeat_utc"] == "2026-07-27T14:22:33.418Z"

    clock.advance(600)
    store.checkpoint()

    assert _lock(store.run_dir)["heartbeat_utc"] == "2026-07-27T14:32:33.418Z"


# ── durability: incremental, atomic, latched ───────────────────────────────────────────────


def test_each_checkpoint_rewrites_the_record_incrementally(tmp_path):
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock)

    clock.advance(60)
    store.checkpoint(counters={"cycles": 1, "trades": 3})
    first = _record(store.run_dir)
    clock.advance(60)
    store.checkpoint(counters={"cycles": 2, "trades": 5})
    second = _record(store.run_dir)

    assert first["segments"][0]["counters"] == {"cycles": 1, "trades": 3}
    assert second["segments"][0]["counters"] == {"cycles": 2, "trades": 5}
    assert first["run"]["last_active_utc"] == "2026-07-27T14:23:33.418Z"
    assert second["run"]["last_active_utc"] == "2026-07-27T14:24:33.418Z"
    assert second["run"]["complete"] is False  # an open segment is not a finished run


def test_a_record_is_complete_only_after_a_clean_segment_close(tmp_path):
    store = _open(tmp_path / "runs", FakeClock())
    assert _record(store.run_dir)["run"]["complete"] is False

    store.close(reason="stopped")

    assert _record(store.run_dir)["run"]["complete"] is True


def test_a_failing_writer_latches_off_logs_one_warning_and_leaves_the_record_incomplete(
    tmp_path, caplog
):
    """The fail-safe latch: a reporting artifact must never take down a multi-week run."""
    calls: list[int] = []

    def flaky_writer(run_dir: Path, record: dict) -> None:
        calls.append(1)
        if len(calls) > 1:  # the opening write lands; everything after fails
            raise OSError("disk went away")
        write(run_dir, record)

    clock = FakeClock()
    with caplog.at_level(logging.WARNING):
        store = _open(tmp_path / "runs", clock, writer=flaky_writer)
        clock.advance(60)
        store.checkpoint(counters={"cycles": 1})
        clock.advance(60)
        store.checkpoint(counters={"cycles": 2})
        clock.advance(60)
        store.close(reason="time_limit", counters={"cycles": 3})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert store.run_id in warnings[0].getMessage()
    assert store.disabled is True
    assert len(calls) == 2  # tripped once, then every later write is a no-op — no retry storm
    record = _record(store.run_dir)
    assert record["run"]["complete"] is False  # honestly partial, never passing for whole
    assert schema.validate(record) == []
    assert not (store.run_dir / RUN_LOCK_NAME).exists()  # the lock is still released


def test_a_kill_mid_write_never_leaves_a_corrupt_record(tmp_path, monkeypatch):
    """Atomic tmp + replace: the record on disk is either the old one or the new one."""
    store = _open(tmp_path / "runs", FakeClock())
    store.close(reason="stopped")
    good = _record(store.run_dir)

    def killed(*_args, **_kwargs):
        raise KeyboardInterrupt("killed between the tmp write and the replace")

    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(KeyboardInterrupt):
        write(store.run_dir, {"schema_version": 1, "kind": "noctis.run", "half": "written"})

    assert _record(store.run_dir) == good
    assert [p.name for p in store.run_dir.iterdir()] == [RUN_RECORD_NAME]  # no tmp litter


def test_an_unreadable_record_does_not_crash_the_next_open(tmp_path, caplog):
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    store.close(reason="stopped")
    (store.run_dir / RUN_RECORD_NAME).write_text('{"schema_version": 1, "run"')

    with caplog.at_level(logging.WARNING):
        reopened = _open(runs, clock, run_id=store.run_id)

    record = _record(reopened.run_dir)
    assert schema.validate(record) == []
    assert any("unreadable" in e["text"] for e in record["events"])


def test_a_record_of_the_wrong_shape_does_not_crash_the_next_open(tmp_path):
    """Valid JSON, foreign shape (hand-edited, or another tool's file): start fresh and say so
    rather than failing a run's startup on a reporting artifact."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock)
    store.close(reason="stopped")
    (store.run_dir / RUN_RECORD_NAME).write_text('{"run": 5, "segments": "nope"}')

    reopened = _open(runs, clock, run_id=store.run_id)

    record = _record(reopened.run_dir)
    assert schema.validate(record) == []
    assert [s["index"] for s in record["segments"]] == [0]
    assert any("unreadable" in e["text"] for e in record["events"])


def test_the_store_writes_nothing_outside_its_own_run_tree(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "state").mkdir(parents=True)
    before = {p for p in workspace.rglob("*")}

    store = _open(workspace / "runs", FakeClock())
    store.checkpoint(counters={"cycles": 1})
    store.close(reason="stopped")

    new = {p for p in workspace.rglob("*")} - before
    assert new  # it did write something
    assert all(p.is_relative_to(workspace / "runs") for p in new)
    assert {p.name for p in store.run_dir.iterdir()} == {RUN_RECORD_NAME}


def test_the_store_runs_no_background_thread(tmp_path):
    """Synchronous by design: a writer thread would reintroduce the shutdown join hazard the
    engine spent four PRs removing."""
    before = threading.active_count()

    store = _open(tmp_path / "runs", FakeClock())
    store.checkpoint()
    store.close(reason="stopped")

    assert threading.active_count() == before


def test_the_run_tree_is_gitignored(tmp_path):
    """Nothing a run writes may reach git — the record is the operator's, not the repo's."""
    checked = subprocess.run(
        ["git", "check-ignore", "-q", "workspace/runs"],
        cwd=Path(__file__).resolve().parents[1],
    )
    assert checked.returncode == 0


def test_the_run_store_pulls_no_optional_extra_and_reads_no_settings():
    """The record is written on the core install alone: no vendor seam, no LLM, no config."""
    code = (
        "import sys\n"
        "from noctis.reporting.run_store import collect, open_run, write\n"
        "assert 'noctis.config' not in sys.modules, 'the run store read config'\n"
        "heavy = {'nautilus_trader', 'vectorbt', 'optuna', 'quantstats', "
        "'databento', 'exchange_calendars', 'anthropic', 'litellm', 'pandas'}\n"
        "loaded = heavy & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ── the golden record, written by a real (simulated-clock) run ─────────────────────────────

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "run_json_golden.json"


def _masked(record: dict) -> dict:
    """Mask what is meant to vary between checkouts: the minted id and the engine's digests.

    Everything else — every field, every stamp, every derived total — is pinned by the golden,
    so a schema change shows up as a reviewable diff instead of as a surprise on a website.
    """
    masked = json.loads(json.dumps(record))
    masked["run"]["run_id"] = "<run_id>"
    for section in [masked["engine"], *masked["segments"]]:
        section["engine_version"] = "<engine_version>"
        key = "fingerprint" if "fingerprint" in section else "engine_fingerprint"
        section[key] = sorted(section[key])
        if "comparable_key" in section:
            section["comparable_key"] = "<comparable_key>"
        if "noctis_version" in section:
            section["noctis_version"] = "<noctis_version>"
    return masked


def test_a_two_segment_fixture_run_matches_the_committed_golden_record(tmp_path):
    """Snapshot the file a real run leaves on disk — two segments, a kill in between, a stolen
    lock — so drift between the store and the record contract is visible in review."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, argv=["run", "-v"])
    clock.advance(3600)
    first.checkpoint(counters={"cycles": 1, "research_iterations": 4, "trades": 2})
    clock.advance(3600)
    first.close(reason="time_limit", counters={"cycles": 2, "research_iterations": 9, "trades": 2})

    clock.advance(36000)
    _write_lock(
        first.run_dir, pid=999_999, hostname_hash="0" * 12, heartbeat_utc="2020-01-01T00:00:00.000Z"
    )
    second = _open(runs, clock, run_id=first.run_id, argv=["run"])
    clock.advance(1800)
    second.close(reason="stop_requested", counters={"cycles": 3})

    record = _record(second.run_dir)
    assert schema.validate(record) == []
    assert _masked(record) == json.loads(GOLDEN.read_text())


# ── the derived index.json roll-up (story #130) ────────────────────────────────────────────


def _index(runs_dir: Path) -> dict:
    return json.loads((runs_dir / RUN_INDEX_NAME).read_text())


def _finished_run(runs: Path, clock: FakeClock, *, seconds: float = 3600.0, **kwargs):
    """One run that actually did some work: opened, ran for a while, closed cleanly."""
    store = _open(runs, clock, **kwargs)
    clock.advance(seconds)
    store.close(reason="time_limit")
    clock.advance(60)
    return store


def test_the_run_tree_carries_a_derived_index_of_every_run(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _finished_run(runs, clock, label="nightly-momo")

    index = _index(runs)

    assert index["schema_version"] == 1
    assert index["kind"] == RUN_INDEX_KIND
    (entry,) = index["runs"]
    assert entry["run_id"] == store.run_id
    assert entry["label"] == "nightly-momo"
    assert entry["status"] == "stopped"
    assert entry["segments"] == 1
    assert entry["cumulative_runtime_s"] == 3600.0
    assert entry["created_utc"] == "2026-07-27T14:22:33.418Z"
    assert entry["readable"] is True


def test_the_index_is_rebuildable_from_the_records_alone_byte_for_byte(tmp_path):
    """The index is DERIVED, never authoritative. A rebuild that read only the records on disk
    must reproduce the incrementally-maintained file exactly — otherwise the roll-up is a second
    source of truth, free to drift from the records it summarizes."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    for label in ("alpha", "beta", "gamma"):
        _finished_run(runs, clock, label=label)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    write_index(elsewhere, rebuild_index(runs))

    assert (elsewhere / RUN_INDEX_NAME).read_bytes() == (runs / RUN_INDEX_NAME).read_bytes()
    assert [e["label"] for e in _index(runs)["runs"]] == ["gamma", "beta", "alpha"]  # newest first


def test_every_index_entry_and_every_record_carries_the_same_comparable_key(tmp_path):
    """The key a leaderboard partitions on structurally, so nobody has to remember which runs
    are poolable — on the record and on its index entry, never on one alone."""
    runs = tmp_path / "runs"
    store = _finished_run(runs, FakeClock(), election_metric="sortino")

    record = _record(store.run_dir)
    (entry,) = _index(runs)["runs"]

    assert record["engine"]["comparable_key"].endswith("|sortino")
    assert entry["comparable_key"] == record["engine"]["comparable_key"]
    assert entry["engine_version"] == record["engine"]["engine_version"] == ENGINE_VERSION
    assert entry["mixed_engine"] is False


def test_a_run_with_no_record_or_an_unreadable_one_is_listed_as_such(tmp_path):
    """A broken record is evidence, not a crash: the entry says what is wrong and the rest of
    the listing is unaffected."""
    runs = tmp_path / "runs"
    good = _finished_run(runs, FakeClock())
    (runs / "20260101T000000Z-empty0").mkdir()
    broken = runs / "20260102T000000Z-brokn0"
    broken.mkdir()
    (broken / RUN_RECORD_NAME).write_text('{"schema_version": 1, "run"')

    entries = {entry["run_id"]: entry for entry in rebuild_index(runs)["runs"]}

    assert entries[good.run_id]["readable"] is True
    assert entries[good.run_id]["note"] is None
    assert entries["20260101T000000Z-empty0"]["readable"] is False
    assert "no run.json" in entries["20260101T000000Z-empty0"]["note"]
    assert entries["20260102T000000Z-brokn0"]["readable"] is False
    assert "unreadable" in entries["20260102T000000Z-brokn0"]["note"]
    # Explicit nulls, never missing keys: every entry answers every question (schema convention).
    assert set(entries["20260101T000000Z-empty0"]) == set(entries[good.run_id])
    assert entries["20260101T000000Z-empty0"]["comparable_key"] is None
    assert entries["20260102T000000Z-brokn0"]["status"] is None


def test_a_record_of_a_foreign_shape_is_listed_as_unreadable_too(tmp_path):
    """Valid JSON, foreign shape — hand-edited or another tool's file. Same degradation."""
    runs = tmp_path / "runs"
    store = _finished_run(runs, FakeClock())
    (store.run_dir / RUN_RECORD_NAME).write_text('{"run": 5, "segments": "nope"}')

    (entry,) = rebuild_index(runs)["runs"]

    assert entry["run_id"] == store.run_id
    assert entry["readable"] is False
    assert "unreadable" in entry["note"]


def test_the_index_is_regenerable_from_scratch_at_any_time(tmp_path):
    runs = tmp_path / "runs"
    clock = FakeClock()
    for label in ("alpha", "beta"):
        _finished_run(runs, clock, label=label)
    before = (runs / RUN_INDEX_NAME).read_bytes()

    (runs / RUN_INDEX_NAME).unlink()
    write_index(runs, rebuild_index(runs))

    assert (runs / RUN_INDEX_NAME).read_bytes() == before


def test_a_run_is_addressed_by_its_id(tmp_path):
    runs = tmp_path / "runs"
    store = _finished_run(runs, FakeClock())

    assert resolve_run_dir(runs, store.run_id) == store.run_dir


def test_an_unknown_address_is_a_clean_lookup_failure_naming_the_run_tree(tmp_path):
    runs = tmp_path / "runs"
    _finished_run(runs, FakeClock())

    with pytest.raises(RunNotFoundError) as excinfo:
        resolve_run_dir(runs, "20260101T000000Z-nope00")

    assert "20260101T000000Z-nope00" in str(excinfo.value)
    assert str(runs) in str(excinfo.value)


def test_the_record_has_no_sidecar_files_and_stands_alone(tmp_path):
    """One `fetch()` of one URL returns everything a run page needs: the run dir holds exactly
    one file, and that file alone reproduces the run's whole index entry."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    store = _open(runs, clock, label="nightly-momo")
    clock.advance(600)
    store.checkpoint(counters={"cycles": 1})
    clock.advance(600)
    store.close(reason="time_limit", counters={"cycles": 2})

    assert [p.name for p in store.run_dir.iterdir()] == [RUN_RECORD_NAME]
    isolated = tmp_path / "isolated" / store.run_id
    isolated.mkdir(parents=True)
    shutil.copy(store.run_dir / RUN_RECORD_NAME, isolated / RUN_RECORD_NAME)
    assert index_entry(isolated) == index_entry(store.run_dir)


def test_the_index_lands_under_the_gitignored_workspace(tmp_path):
    checked = subprocess.run(
        ["git", "check-ignore", "-q", f"workspace/runs/{RUN_INDEX_NAME}"],
        cwd=Path(__file__).resolve().parents[1],
    )
    assert checked.returncode == 0


# ── the CLI: always-on run identity ────────────────────────────────────────────────────────


def _config(tmp_path) -> str:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\nstate_dir: {tmp_path}/state/\n"
    )
    return str(cfg)


def _runs_dir(tmp_path) -> Path:
    # conftest pins NOCTIS_WORKSPACE at <tmp_path>/workspace for every test.
    return tmp_path / "workspace" / "runs"


def _run_dirs(runs_dir: Path) -> list[Path]:
    """The run trees under ``runs/`` — everything but the derived ``index.json`` beside them."""
    return sorted(p for p in runs_dir.iterdir() if p.is_dir())


def test_noctis_run_mints_a_new_run_and_creates_its_tree_every_invocation(tmp_path):
    cfg = _config(tmp_path)

    first = runner.invoke(app, ["run", "--config", cfg])
    second = runner.invoke(app, ["run", "--config", cfg])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    minted = [p.name for p in _run_dirs(_runs_dir(tmp_path))]
    assert len(minted) == 2
    assert all(RUN_ID_RE.match(name) for name in minted)
    for name in minted:
        record = json.loads((_runs_dir(tmp_path) / name / RUN_RECORD_NAME).read_text())
        assert schema.validate(record) == []
        assert record["run"]["status"] == "stopped"
        assert record["run"]["complete"] is True
        assert record["segments"][0]["command"] == "run"
        assert record["segments"][0]["stopped_reason"]
        assert record["engine"]["engine_version"] == ENGINE_VERSION


def test_noctis_run_echoes_the_run_id_and_the_record_path(tmp_path):
    result = runner.invoke(app, ["run", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    run_id = _run_dirs(_runs_dir(tmp_path))[0].name
    assert run_id in result.output
    assert RUN_RECORD_NAME in result.output


def test_the_run_leaves_no_lock_behind_after_a_clean_stop(tmp_path):
    runner.invoke(app, ["run", "--config", _config(tmp_path)])

    run_dir = _run_dirs(_runs_dir(tmp_path))[0]
    assert not (run_dir / RUN_LOCK_NAME).exists()


def test_the_debug_qa_tree_reuses_the_runs_own_id(tmp_path):
    """One run, one id, one tree: the --debug QA area is filed under the same minted id rather
    than a second one nobody can correlate."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"mode: paper\ndata:\n  lake_dir: {tmp_path}/lake\n"
        f"state_dir: {tmp_path}/state/\nqa_dir: {tmp_path}/qa\n"
    )

    result = runner.invoke(app, ["run", "--config", str(cfg), "--debug"])

    assert result.exit_code == 0, result.output
    run_ids = [p.name for p in _run_dirs(_runs_dir(tmp_path))]
    qa_ids = [p.name for p in (tmp_path / "qa").iterdir() if p.is_dir()]
    assert run_ids == qa_ids


def test_no_secret_reaches_the_record(tmp_path, monkeypatch):
    """A record is meant to be shared. Nothing the settings digest excludes may appear in one."""
    from noctis.bootstrap import _DIGEST_SECRET_FIELDS

    secrets = {field: f"sk-{field}-do-not-leak" for field in _DIGEST_SECRET_FIELDS}
    for field, value in secrets.items():
        monkeypatch.setenv(field.upper(), value)

    result = runner.invoke(app, ["run", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    text = next(_runs_dir(tmp_path).rglob(RUN_RECORD_NAME)).read_text()
    for value in secrets.values():
        assert value not in text


# ── the engine loop writes at each CLOSE ───────────────────────────────────────────────────


def test_the_record_is_rewritten_at_every_close_and_at_segment_close(tmp_path):
    """Incremental durability: a multi-day run's record is current after every CLOSE, not only
    when the process finally stops."""
    from zoneinfo import ZoneInfo

    from noctis.bootstrap import segment_counters
    from noctis.config import load_settings
    from noctis.data import MarketDataLake
    from noctis.data.types import to_ns
    from noctis.engine import SimulatedSleeper, build_runtime
    from noctis.memory import MemoryStore

    from ._data_helpers import MockVendor

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\nuniverse: [AAPL, MSFT]\n"
        "session:\n  calendar: XNYS\n  timezone: America/New_York\n"
        "research_time_budget_minutes: 60\n"
        f"data:\n  lake_dir: {tmp_path}/lake\n  dataset: EQUS.MINI\n"
        f"state_dir: {tmp_path}/state/\n"
    )
    settings = load_settings(config_path=cfg)
    lake = MarketDataLake(tmp_path / "lake", MockVendor(), budget_usd=10_000.0, calendar="XNYS")
    lake.ensure_coverage(
        "EQUS.MINI", "ohlcv-1m", ["AAPL", "MSFT"], to_ns("2026-01-01"), to_ns("2026-12-31")
    )

    snapshots: list[dict] = []

    def recording_writer(run_dir: Path, record: dict) -> None:
        snapshots.append(json.loads(json.dumps(record)))
        write(run_dir, record)

    clock = FakeClock()
    store = _open(tmp_path / "runs", clock, writer=recording_writer)
    runtime = build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        reports_dir=str(tmp_path / "reports"),
        research_max_iters=2,
        sleeper_factory=lambda start: SimulatedSleeper(start),
        on_cycle_close=lambda result: store.checkpoint(counters=segment_counters(result)),
    )
    result = runtime.run(
        start=datetime(2027, 1, 4, 6, 0, tzinfo=ZoneInfo("America/New_York")), max_cycles=2
    )
    clock.advance(120)
    store.close(reason=result.stopped_reason, counters=segment_counters(result))

    assert result.cycles_completed == 2
    # opening write + one per CLOSE + the segment-close write
    assert len(snapshots) == 4
    # the opening write counts nothing yet (an honest empty), then one write per CLOSE
    assert [s["segments"][0]["counters"].get("cycles") for s in snapshots] == [None, 1, 2, 2]
    assert [s["run"]["complete"] for s in snapshots] == [False, False, False, True]
    final = _record(store.run_dir)
    assert final["segments"][0]["stopped_reason"] == "max_cycles"
    assert final["segments"][0]["counters"]["cycles"] == 2
    assert schema.validate(final) == []


# ── the CLI: `noctis runs` and `noctis run-record` (story #130) ────────────────────────────


def test_noctis_runs_lists_id_label_status_segments_and_headline_numbers(tmp_path):
    runs = _runs_dir(tmp_path)
    clock = FakeClock()
    first = _finished_run(runs, clock, label="nightly-momo", seconds=7200)
    second = _finished_run(runs, clock, seconds=1800)

    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if first.run_id in line]
    assert len(lines) == 1
    assert "nightly-momo" in lines[0]
    assert "stopped" in lines[0]
    assert "2h00m" in lines[0]  # the headline runtime, in a shape a human reads
    assert " 1 " in lines[0]  # one segment
    assert json.loads((runs / RUN_INDEX_NAME).read_text())["runs"][0]["run_id"] == second.run_id
    assert second.run_id in result.output


def test_noctis_runs_hides_short_runs_until_all_widens_the_filter(tmp_path):
    """The default listing is the operator's experiment board, so it hides the noise a startup
    failure or a mistyped command leaves behind — and says how many it hid."""
    runs = _runs_dir(tmp_path)
    clock = FakeClock()
    real = _finished_run(runs, clock, label="real-work", seconds=7200)
    aborted = _finished_run(runs, clock, label="aborted", seconds=2)
    cfg = _config(tmp_path)

    default = runner.invoke(app, ["runs", "--config", cfg])
    widened = runner.invoke(app, ["runs", "--all", "--config", cfg])

    assert default.exit_code == 0, default.output
    assert real.run_id in default.output
    assert aborted.run_id not in default.output
    assert "--all" in default.output  # the hidden ones are never silent
    assert widened.exit_code == 0, widened.output
    assert real.run_id in widened.output and aborted.run_id in widened.output


def test_noctis_runs_lists_an_unreadable_run_rather_than_crashing(tmp_path):
    runs = _runs_dir(tmp_path)
    good = _finished_run(runs, FakeClock(), seconds=7200)
    broken = runs / "20260102T000000Z-brokn0"
    broken.mkdir()
    (broken / RUN_RECORD_NAME).write_text("{ not json")

    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert good.run_id in result.output
    assert "20260102T000000Z-brokn0" in result.output
    assert "unreadable" in result.output


def test_noctis_runs_regenerates_the_index_from_the_records_on_disk(tmp_path):
    runs = _runs_dir(tmp_path)
    clock = FakeClock()
    _finished_run(runs, clock, label="alpha", seconds=7200)
    _finished_run(runs, clock, label="beta", seconds=7200)
    before = (runs / RUN_INDEX_NAME).read_bytes()
    (runs / RUN_INDEX_NAME).unlink()

    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (runs / RUN_INDEX_NAME).read_bytes() == before


def test_noctis_runs_with_no_runs_yet_says_so(tmp_path):
    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "No runs" in result.output


def test_noctis_run_record_prints_the_record_for_a_run(tmp_path):
    runs = _runs_dir(tmp_path)
    store = _finished_run(runs, FakeClock(), label="nightly-momo", seconds=7200)

    result = runner.invoke(app, ["run-record", store.run_id, "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    printed = json.loads(result.output)
    assert printed == _record(store.run_dir)
    assert schema.validate(printed) == []
    assert printed["engine"]["comparable_key"]


def test_noctis_run_record_on_an_unknown_id_exits_nonzero_naming_the_run_tree(tmp_path):
    runs = _runs_dir(tmp_path)
    _finished_run(runs, FakeClock(), seconds=7200)

    result = runner.invoke(
        app, ["run-record", "20260101T000000Z-nope00", "--config", _config(tmp_path)]
    )

    assert result.exit_code == 1
    assert "20260101T000000Z-nope00" in result.output


def test_noctis_run_record_on_an_unreadable_record_exits_nonzero_saying_why(tmp_path):
    runs = _runs_dir(tmp_path)
    broken = runs / "20260102T000000Z-brokn0"
    broken.mkdir(parents=True)
    (broken / RUN_RECORD_NAME).write_text("{ not json")

    result = runner.invoke(app, ["run-record", broken.name, "--config", _config(tmp_path)])

    assert result.exit_code == 1
    assert "unreadable" in result.output
