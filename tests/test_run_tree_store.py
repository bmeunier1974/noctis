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

import dataclasses
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from noctis.bootstrap import capture_environment
from noctis.cli import app
from noctis.observability.engine_id import ENGINE_VERSION
from noctis.observability.runid import RUN_ID_RE
from noctis.reporting import schema
from noctis.reporting.run_record import RunArtifacts
from noctis.reporting.run_tree import (
    RUN_INDEX_NAME,
    RUN_LOCK_NAME,
    RUN_RECORD_NAME,
    Evidence,
    RunLockedError,
    RunStore,
    open_run,
    read_artifacts,
    rebuild_index,
    write,
    write_index,
)

from ._run_tree_helpers import ENGINE, FakeClock, hold_lock, stamp, write_run

runner = CliRunner()


def _open(runs_dir: Path, clock: FakeClock, **kwargs):
    kwargs.setdefault("argv", ["run", "-v"])
    kwargs.setdefault("election_metric", "sharpe")
    return open_run(runs_dir, clock=clock, **kwargs)


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_RECORD_NAME).read_text())


def _lock(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_LOCK_NAME).read_text())


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


# ── the environment is per segment, never per run (story #139) ─────────────────────────────


def _machine(host: str, *, cores: int) -> dict:
    """One machine's environment block, handed to the store the way the composition root hands
    it the block its injected probes captured — no hardware is read from this test."""
    return {
        "hostname_hash": host,
        "os": {"system": "Linux", "release": "7.0.0-14-generic", "arch": "x86_64"},
        "container": False,
        "cpu": {
            "model": "AMD Ryzen 9 7950X",
            "cores_physical": cores // 2,
            "cores_logical": cores,
            "freq_max_mhz": 5881.0,
        },
        "memory_total_bytes": 67351248896,
        "disk_free_bytes": 412000000000,
        "python": "3.11.9",
        "noctis_version": "0.1.0",
        "git": {"commit": "a380d3a", "branch": "main", "dirty": False, "describe": "v0.1.0-42"},
        "lockfile_digest": "sha256:beef",
        "extras_present": {
            "llm": None,
            "data": None,
            "research": None,
            "engine": None,
            "hardware": None,
        },
        "degraded_seams": ["data", "engine", "hardware", "llm", "research"],
    }


def test_a_segment_records_the_machine_it_was_opened_on(tmp_path):
    store = _open(tmp_path / "runs", FakeClock(), environment=_machine("aaaa1111", cores=8))

    record = _record(store.run_dir)

    assert record["segments"][0]["environment"] == _machine("aaaa1111", cores=8)
    assert record["environment_latest"] == _machine("aaaa1111", cores=8)
    assert schema.validate(record) == []


def test_a_run_resumed_on_another_machine_keeps_both_segments_environments(tmp_path):
    """A run may migrate boxes mid-experiment. Research throughput is CPU-bound, so the first
    night's trials-per-hour must stay attributed to the first night's hardware."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, environment=_machine("aaaa1111", cores=8))
    clock.advance(3600)
    first.close(reason="time_limit")

    clock.advance(36000)
    second = _open(
        runs, clock, run_id=first.run_id, resume=True, environment=_machine("bbbb2222", cores=32)
    )
    second.close(reason="stop_requested")

    record = _record(second.run_dir)
    assert [s["environment"]["hostname_hash"] for s in record["segments"]] == [
        "aaaa1111",
        "bbbb2222",
    ]
    assert [s["environment"]["cpu"]["cores_logical"] for s in record["segments"]] == [8, 32]
    assert record["environment_latest"]["hostname_hash"] == "bbbb2222"
    assert schema.validate(record) == []


def test_a_segment_opened_without_an_environment_says_so_with_an_explicit_null(tmp_path):
    store = _open(tmp_path / "runs", FakeClock())

    record = _record(store.run_dir)

    assert record["segments"][0]["environment"] is None
    assert record["environment_latest"] is None
    assert "environment_latest" in record
    assert schema.validate(record) == []


# ── research-only segments and the trials they journal (story #137) ────────────────────────


def test_a_segment_records_the_command_it_was_opened_by(tmp_path):
    """One record, two kinds of night: the loop's, and a standalone research session's."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, argv=["run", "-v"])
    clock.advance(3600)
    first.close(reason="time_limit")

    second = _open(
        runs,
        clock,
        run_id=first.run_id,
        resume=True,
        command="research",
        argv=["research", "--resume", "latest"],
    )
    clock.advance(1800)
    second.close(reason="agent_done", phase_seconds={"RESEARCH": 1750.0})

    record = _record(first.run_dir)
    assert [segment["command"] for segment in record["segments"]] == ["run", "research"]
    assert record["segments"][1]["argv"] == ["research", "--resume", "latest"]
    assert record["segments"][1]["stopped_reason"] == "agent_done"
    assert record["run"]["cumulative_research_s"] == 1750.0
    assert schema.validate(record) == []


def test_a_run_killed_mid_segment_is_marked_interrupted_on_the_next_open(tmp_path, caplog):
    runs = tmp_path / "runs"
    clock = FakeClock()
    killed = _open(runs, clock)
    # At write time the record says exactly what was true: a segment is open. Nothing is guessed.
    assert _record(killed.run_dir)["segments"][0]["status"] == "running"
    # The kill: the process is gone, its lock left behind holding a pid that no longer exists.
    hold_lock(killed.run_dir, run_id=killed.run_id, pid=_dead_pid())

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
    hold_lock(killed.run_dir, run_id=killed.run_id, pid=_dead_pid())

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


# ── read_artifacts: the record half of the read, on its own (story #288) ───────────────────
#
# The two tests above drive a whole store over a broken record; these call the parse directly and
# pin the same note, byte for byte, so the text a run says about its own lost history cannot drift
# when the collectors are no longer part of the same function.


def _fresh(run_dir: Path):
    return read_artifacts(run_dir, current=ENGINE)


def test_a_run_dir_with_no_record_yet_reads_back_as_an_empty_run(tmp_path):
    """The normal case for a fresh run: nothing on disk, nothing to say about it."""
    run_dir = tmp_path / "runs" / "20260727T142233Z-a1b2c3"
    run_dir.mkdir(parents=True)

    artifacts = _fresh(run_dir)

    assert artifacts.run_id == "20260727T142233Z-a1b2c3"
    assert artifacts.created_utc is None
    assert artifacts.segments == ()
    assert artifacts.events == ()
    assert artifacts.engine is ENGINE
    assert artifacts.current_engine is ENGINE


def test_an_unreadable_record_reads_back_as_a_fresh_one_saying_so(tmp_path):
    run_dir = tmp_path / "runs" / "20260727T142233Z-a1b2c3"
    run_dir.mkdir(parents=True)
    (run_dir / RUN_RECORD_NAME).write_text('{"schema_version": 1, "run"')

    artifacts = _fresh(run_dir)

    assert [(e.kind, e.text) for e in artifacts.events] == [
        (
            "warn",
            "this run had an unreadable run.json (JSONDecodeError); "
            "a fresh record was started in its place",
        )
    ]


def test_a_record_of_a_foreign_shape_reads_back_as_a_fresh_one_saying_so(tmp_path):
    """Valid JSON, foreign shape: the parse raises, and the raise becomes the note."""
    run_dir = tmp_path / "runs" / "20260727T142233Z-a1b2c3"
    run_dir.mkdir(parents=True)
    (run_dir / RUN_RECORD_NAME).write_text('{"run": 5, "segments": "nope"}')

    artifacts = _fresh(run_dir)

    assert [(e.kind, e.text) for e in artifacts.events] == [
        (
            "warn",
            "this run had an unreadable run.json (TypeError); "
            "a fresh record was started in its place",
        )
    ]


def test_reading_a_record_derives_nothing_off_the_run_tree(tmp_path):
    """The seven derived fields stay at their defaults — an open reads the record, not the run."""
    clock = FakeClock()
    store = _open(tmp_path / "runs", clock)
    _journal_a_trial(store.run_dir)
    store.checkpoint()
    assert _record(store.run_dir)["run"]["cumulative_trials"] == 1  # it is on disk

    artifacts = _fresh(store.run_dir)

    assert artifacts.trials is None
    assert artifacts.spend is None
    assert artifacts.pricing_table_version is None
    assert artifacts.champions is None
    assert artifacts.strategies == ()
    assert artifacts.sessions == ()
    assert artifacts.benchmark is None
    assert artifacts.run_id == store.run_id  # the record itself came back in full


def _journal_a_trial(run_dir: Path) -> None:
    from noctis.research.journal import ExperimentJournal
    from tests.test_champions import make_scorecard

    ExperimentJournal(run_dir / "state").record_trial(
        "alpha",
        source="sweep",
        symbols=["AAPL"],
        params={"lookback": 10},
        window={},
        card=make_scorecard("alpha", test_metric=1.2, train_metric=1.4),
    )


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


def test_the_run_tree_pulls_no_optional_extra_and_reads_no_settings():
    """The record is written on the core install alone: no vendor seam, no LLM, no config.

    Import-time cost, measured on the whole package — including ``evidence``, which is the one
    module allowed to *name* a heavy package and does so only inside the bodies that read. Where
    such an import may be written at all is the other half of this guard, and it is pinned
    statically by ``tests/test_run_tree_boundary.py``.
    """
    code = (
        "import sys\n"
        "from noctis.reporting.run_tree import derive_evidence, open_run, read_artifacts, write\n"
        "assert 'noctis.config' not in sys.modules, 'the run store read config'\n"
        "heavy = {'nautilus_trader', 'vectorbt', 'optuna', 'quantstats', "
        "'databento', 'exchange_calendars', 'anthropic', 'litellm', 'pandas', "
        "'noctis.research', 'noctis.champions', 'noctis.broker'}\n"
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


# ── every field of a record is derived, store-owned or carried (story #289) ────────────────
#
# The classification below is the whole contract of ``RunStore``'s copies, written down once:
#
# * **derived** — the seven fields ``derive_evidence`` reads off the run's own durable artifacts
#   at every write, and that a store must therefore overwrite;
# * **store-owned** — what an open, a checkpoint and a close legitimately rewrite: the stamps, the
#   segment list, the label, the two engine identities, the event streams and ``complete``;
# * **carried** — everything else, which the record froze and every later segment must hand
#   forward verbatim.
#
# A field added to ``RunArtifacts`` without a classification lands in ``CARRIED`` with no sample
# and fails the coverage assertion; a carried field a copy forgets fails the round trip.

DERIVED = {f.name for f in dataclasses.fields(Evidence)}
STORE_OWNED = {
    "created_utc",
    "last_active_utc",
    "segments",
    "complete",
    "label",
    "engine",
    "current_engine",
    "events",
    "errors",
}
CARRIED = {f.name for f in dataclasses.fields(RunArtifacts)} - DERIVED - STORE_OWNED

CARRIED_RUN_ID = "20260727T142233Z-a1b2c3"
# One non-default, JSON-round-trippable value per carried field: a value the store's copies cannot
# reproduce by accident, so a dropped field comes back as its default and fails.
CARRIED_SAMPLES: dict[str, object] = {
    "run_id": CARRIED_RUN_ID,
    "completed_utc": "2026-07-27T16:00:00.000Z",
    "inputs": {"settings": {"resolved": {"mode": "paper"}}},
    "state_pruned": True,
}


def _store_over(run_dir: Path, **carried: object) -> RunStore:
    """A store opened over artifacts that already carry something — what ``open_run`` builds once
    it has parsed the record, minus the lock and the engine fingerprint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, object] = {
        "run_id": run_dir.name,
        "created_utc": None,
        "last_active_utc": None,
        "engine": ENGINE,
        "current_engine": ENGINE,
    }
    artifacts = RunArtifacts(**{**fields, **carried})  # type: ignore[arg-type]
    return RunStore(run_dir, artifacts=artifacts, clock=FakeClock(), argv=["run", "-v"])


def _carried_round_trip(runs_dir: Path, field: str, sample: object) -> object:
    """Open, checkpoint and close a store over artifacts carrying ``sample``, then read the value
    back off the ``run.json`` it left behind."""
    run_dir = runs_dir / CARRIED_RUN_ID
    store = _store_over(run_dir, **{field: sample})
    store.checkpoint(counters={"cycles": 1})
    store.close(reason="stopped")

    return getattr(read_artifacts(run_dir, current=ENGINE), field)


def test_every_artifact_field_is_carried_or_derived(tmp_path):
    """Every field of ``RunArtifacts`` is classified, and every carried one survives a segment."""
    assert DERIVED & STORE_OWNED == set()
    assert DERIVED | STORE_OWNED | CARRIED == {f.name for f in dataclasses.fields(RunArtifacts)}
    assert set(CARRIED_SAMPLES) == CARRIED  # every carried field has a sample to round-trip

    round_tripped = {
        field: _carried_round_trip(tmp_path / field, field, sample)
        for field, sample in sorted(CARRIED_SAMPLES.items())
    }

    assert round_tripped == CARRIED_SAMPLES


def test_a_pruned_runs_state_pruned_survives_the_open_that_appends_a_segment(tmp_path):
    """``state_pruned`` is carried forward verbatim, exactly as the record's own docstring says:
    it states that the heavy directories were deliberately removed, and no later write may un-say
    it because something recreated an empty ``state/``."""
    run_dir = tmp_path / "runs" / CARRIED_RUN_ID

    store = _store_over(run_dir, state_pruned=True)

    assert store.record()["run"]["state_pruned"] is True
    store.close(reason="stopped")
    assert _record(run_dir)["run"]["state_pruned"] is True


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
    """Snapshot the file a real run leaves on disk — two segments on two different machines, a
    kill in between, a stolen lock — so drift between the store and the record contract is visible
    in review."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _open(runs, clock, argv=["run", "-v"], environment=_machine("aaaa1111", cores=8))
    clock.advance(3600)
    first.checkpoint(counters={"cycles": 1, "research_iterations": 4, "trades": 2})
    clock.advance(3600)
    first.close(
        reason="time_limit",
        counters={"cycles": 2, "research_iterations": 9, "trades": 2},
        phase_seconds={"RESEARCH": 5400.0, "TRADING": 1800.0, "CLOSE": 30.0},
    )

    clock.advance(36000)
    hold_lock(
        first.run_dir,
        run_id=first.run_id,
        pid=999_999,
        hostname_hash="0" * 12,
        heartbeat_utc="2020-01-01T00:00:00.000Z",
    )
    second = _open(
        runs, clock, run_id=first.run_id, argv=["run"], environment=_machine("bbbb2222", cores=32)
    )
    clock.advance(1800)
    second.close(
        reason="stop_requested",
        counters={"cycles": 3},
        phase_seconds={"RESEARCH": 1700.0, "TRADING": 0.0, "CLOSE": 20.0},
    )

    record = _record(second.run_dir)
    assert schema.validate(record) == []
    assert _masked(record) == json.loads(GOLDEN.read_text())
    # The run-level phase totals are re-derived from both segments as they sit on disk — the
    # resumed process never held the first night's numbers in memory.
    assert record["run"]["cumulative_research_s"] == 7100.0
    assert record["run"]["cumulative_trading_s"] == 1800.0


# ── addressing and the index, where the subject is the STORE (story #287) ──────────────────
#
# The rules themselves live with the modules that own them: `tests/test_run_tree_address.py`
# resolves the four address forms over written records, `tests/test_run_tree_index.py` derives the
# roll-up from them. What stays here is the store's *use* of both — an open reached by an alias,
# and the listing the CLI renders after a real run.


def _finished_run(runs: Path, clock: FakeClock, *, seconds: float = 3600.0, **kwargs):
    """One run that actually did some work: opened, ran for a while, closed cleanly."""
    store = _open(runs, clock, **kwargs)
    clock.advance(seconds)
    store.close(reason="time_limit")
    clock.advance(60)
    return store


def test_a_run_opened_through_an_alias_is_locked_and_recorded_under_its_own_id(tmp_path):
    """An address is how you *reach* a run; the id is what it *is*. Everything the open writes —
    the lock, the record, the store's own id — names the id, never the alias it was reached by."""
    runs = tmp_path / "runs"
    clock = FakeClock()
    first = _finished_run(runs, clock, label="nightly-momo")

    resumed = _open(runs, clock, run_id="@nightly-momo", resume=True)
    lock = _lock(resumed.run_dir)
    resumed.close(reason="stopped")

    assert resumed.run_id == first.run_id
    assert lock["run_id"] == first.run_id
    assert _record(first.run_dir)["run"]["run_id"] == first.run_id
    assert len(_record(first.run_dir)["segments"]) == 2


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
    """A record is meant to be shared. Nothing the settings digest excludes may appear in one —
    not a value, and not a credential's *name*, in any section of the document.

    The value check is over the serialized file, so it covers every section the epic has grown
    since — the frozen settings, the resolved models, the environment, the strategies, the
    sessions and the honesty block. The **name** check is stronger than "not in these two
    sections": a credential may appear in exactly one place, ``inputs.settings``' tier lists,
    where naming it is the point (it says the key is live tier and therefore never restored).
    Remove those three lists and no section may mention it at all.
    """
    from noctis.bootstrap import _DIGEST_SECRET_FIELDS

    secrets = {field: f"sk-{field}-do-not-leak" for field in _DIGEST_SECRET_FIELDS}
    for field, value in secrets.items():
        monkeypatch.setenv(field.upper(), value)

    result = runner.invoke(app, ["run", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    record_path = next(_runs_dir(tmp_path).rglob(RUN_RECORD_NAME))
    text = record_path.read_text()
    for value in secrets.values():
        assert value not in text
    record = json.loads(text)
    assert record["inputs"]["models"]["research"] is not None  # the block IS populated
    assert record["assumptions"]["paper_only"] is True  # so is the honesty block
    for tier in ("frozen_keys", "live_keys", "refused_keys"):
        record["inputs"]["settings"].pop(tier)
    serialized = json.dumps(record)
    for field in _DIGEST_SECRET_FIELDS:
        assert field not in serialized, field


def test_a_real_run_records_the_machine_it_ran_on(tmp_path):
    """The composition root's probes, end to end: a bare ``noctis run`` on the core install
    leaves a schema-valid environment block on its segment."""
    result = runner.invoke(app, ["run", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    record = json.loads(next(_runs_dir(tmp_path).rglob(RUN_RECORD_NAME)).read_text())

    environment = record["segments"][0]["environment"]
    assert environment["python"] is not None
    assert environment["os"]["system"] is not None
    assert record["environment_latest"] == environment
    assert schema.validate(record) == []


def test_the_environments_hostname_hash_is_the_one_the_lock_writes(tmp_path):
    """Story #129 hashes the hostname into the lock; #139 hashes it into the record. One machine,
    one digest — otherwise two segments on one host would not be provably the same host."""
    store = _open(tmp_path / "runs", FakeClock(), environment=capture_environment())

    lock = _lock(store.run_dir)
    environment = _record(store.run_dir)["segments"][0]["environment"]

    assert environment["hostname_hash"] == lock["hostname_hash"]


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
#
# These verbs are the subject here, not the store: what the listing prints, which runs it hides,
# what an address prints. So the fixture is a record written through the pure builder — the same
# one the addressing and index tests use — and the last test in the file pins that the CLI cannot
# tell such a record from a real store's own.

LISTED = "20260727T142233Z-a1b2c3"
SECOND_LISTED = "20260727T152233Z-d4e5f6"
BROKEN = "20260102T000000Z-brokn0"


def test_noctis_runs_lists_id_label_status_segments_and_headline_numbers(tmp_path):
    runs = _runs_dir(tmp_path)
    first = write_run(runs, LISTED, label="nightly-momo", runtime_s=7200)
    second = write_run(runs, SECOND_LISTED, created_utc=stamp(7200), runtime_s=1800)

    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if first.name in line]
    assert len(lines) == 1
    assert "nightly-momo" in lines[0]
    assert "stopped" in lines[0]
    assert "2h00m" in lines[0]  # the headline runtime, in a shape a human reads
    assert " 1 " in lines[0]  # one segment
    assert json.loads((runs / RUN_INDEX_NAME).read_text())["runs"][0]["run_id"] == second.name
    assert second.name in result.output


def test_noctis_runs_hides_short_runs_until_all_widens_the_filter(tmp_path):
    """The default listing is the operator's experiment board, so it hides the noise a startup
    failure or a mistyped command leaves behind — and says how many it hid."""
    runs = _runs_dir(tmp_path)
    real = write_run(runs, LISTED, label="real-work", runtime_s=7200)
    aborted = write_run(runs, SECOND_LISTED, label="aborted", created_utc=stamp(7200), runtime_s=2)
    cfg = _config(tmp_path)

    default = runner.invoke(app, ["runs", "--config", cfg])
    widened = runner.invoke(app, ["runs", "--all", "--config", cfg])

    assert default.exit_code == 0, default.output
    assert real.name in default.output
    assert aborted.name not in default.output
    assert "--all" in default.output  # the hidden ones are never silent
    assert widened.exit_code == 0, widened.output
    assert real.name in widened.output and aborted.name in widened.output


def test_noctis_runs_says_everything_was_hidden_rather_than_crashing(tmp_path):
    """A workspace whose only run is a failed start has runs to count but none to show. The
    listing must still tell the operator that, and how to see them."""
    runs = _runs_dir(tmp_path)
    aborted = write_run(runs, LISTED, label="aborted", runtime_s=2)

    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert aborted.name not in result.output
    assert "--all" in result.output  # the hidden ones are never silent


def test_noctis_runs_lists_an_unreadable_run_rather_than_crashing(tmp_path):
    runs = _runs_dir(tmp_path)
    good = write_run(runs, LISTED, runtime_s=7200)
    broken = runs / BROKEN
    broken.mkdir()
    (broken / RUN_RECORD_NAME).write_text("{ not json")

    result = runner.invoke(app, ["runs", "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    assert good.name in result.output
    assert BROKEN in result.output
    assert "unreadable" in result.output


def test_noctis_runs_regenerates_the_index_from_the_records_on_disk(tmp_path):
    runs = _runs_dir(tmp_path)
    write_run(runs, LISTED, label="alpha", runtime_s=7200)
    write_run(runs, SECOND_LISTED, label="beta", created_utc=stamp(7200), runtime_s=7200)
    write_index(runs, rebuild_index(runs))
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
    run_dir = write_run(runs, LISTED, label="nightly-momo", runtime_s=7200)

    result = runner.invoke(app, ["run-record", run_dir.name, "--config", _config(tmp_path)])

    assert result.exit_code == 0, result.output
    printed = json.loads(result.output)
    assert printed == _record(run_dir)
    assert schema.validate(printed) == []
    assert printed["engine"]["comparable_key"]


def test_noctis_run_record_takes_the_same_address_forms_as_resume(tmp_path):
    """One resolver, one set of rules: a verb that addresses a run understands every form."""
    runs = _runs_dir(tmp_path)
    write_run(runs, LISTED, runtime_s=7200)
    momo = write_run(
        runs, SECOND_LISTED, label="nightly-momo", created_utc=stamp(7200), runtime_s=7200
    )
    cfg = _config(tmp_path)

    for address in ("@nightly-momo", "latest", str(momo / RUN_RECORD_NAME)):
        result = runner.invoke(app, ["run-record", address, "--config", cfg])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["run"]["run_id"] == momo.name


def test_noctis_run_record_on_an_unknown_id_exits_nonzero_naming_the_run_tree(tmp_path):
    runs = _runs_dir(tmp_path)
    write_run(runs, LISTED, runtime_s=7200)

    result = runner.invoke(
        app, ["run-record", "20260101T000000Z-nope00", "--config", _config(tmp_path)]
    )

    assert result.exit_code == 1
    assert "20260101T000000Z-nope00" in result.output


def test_noctis_run_record_on_an_unreadable_record_exits_nonzero_saying_why(tmp_path):
    runs = _runs_dir(tmp_path)
    broken = runs / BROKEN
    broken.mkdir(parents=True)
    (broken / RUN_RECORD_NAME).write_text("{ not json")

    result = runner.invoke(app, ["run-record", broken.name, "--config", _config(tmp_path)])

    assert result.exit_code == 1
    assert "unreadable" in result.output


def test_the_cli_cannot_tell_a_written_record_from_a_real_stores_own(tmp_path):
    """Why the addressing and index tests may drop the store: the fixture is the same artifact.

    One run opened, run and closed by a real store, one written by ``write_run`` with the same
    stamps, the same label and the same runtime — and ``noctis runs`` prints the same row for
    both, down to the comparable key. The one column a hand-written record cannot derive is the
    engine identity itself (a real fingerprint costs exactly what these fixtures exist to avoid),
    so the key is handed to the builder; everything else is what the record says.
    """
    runs = _runs_dir(tmp_path)
    real = _finished_run(runs, FakeClock(), label="twin")
    twin = write_run(
        runs,
        "20260101T000000Z-abcdef",
        label="twin",
        comparable_key=_record(real.run_dir)["engine"]["comparable_key"],
    )
    cfg = _config(tmp_path)

    listed = runner.invoke(app, ["runs", "--config", cfg])

    assert listed.exit_code == 0, listed.output
    (real_row,) = [line for line in listed.output.splitlines() if real.run_id in line]
    (twin_row,) = [line for line in listed.output.splitlines() if twin.name in line]
    assert twin_row.replace(twin.name, real.run_id) == real_row
    for run_dir in (real.run_dir, twin):
        printed = runner.invoke(app, ["run-record", run_dir.name, "--config", cfg])
        assert printed.exit_code == 0, printed.output
        assert json.loads(printed.output) == _record(run_dir)  # each prints its own, verbatim
        assert schema.validate(json.loads(printed.output)) == []
