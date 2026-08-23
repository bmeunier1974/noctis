"""The run lock — the one place a failure in the run tree is fatal (epic #284, story #286).

Two engines writing one run is corruption, not degradation, so a *live* lock is a hard refusal and
only a **stale** one (a dead pid on this host, or a heartbeat gone cold from a host we cannot
check) may be taken, loudly. That decision is answerable from the lock file alone, so every test
here writes one with ``hold_lock`` and calls a lock verb over it — no store is opened, no record is
read, no engine fingerprint is computed. The store's *use* of the lock (two opens refuse, close
releases, the heartbeat moves at each checkpoint) stays in ``tests/test_run_tree_store.py``.

Everything asserted is external: what the verb returned or raised, and what ``run.lock`` says
afterwards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from noctis.reporting.run_tree.lock import (
    RUN_LOCK_NAME,
    STALE_HEARTBEAT_S,
    RunLockedError,
    acquire_lock,
    assert_unlocked,
    release_lock,
    touch_lock,
)

from ._run_tree_helpers import START, hold_lock

RUN_ID = "20260727T142233Z-a1b2c3"

# A hash no host produces: the "another machine holds this" side of the stale rule.
OTHER_HOST = "0" * 12

CONSEQUENCE = "so it cannot be sealed from underneath it."


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    return run_dir


def _lock(run_dir: Path) -> dict:
    return json.loads((run_dir / RUN_LOCK_NAME).read_text())


def _our_host_hash() -> str:
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]


def _dead_pid() -> int:
    """A pid that is provably not running: a child we started and reaped."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


# ── the stale rules: which held lock may be taken ──────────────────────────────────────────


def test_a_live_lock_from_another_host_is_never_stolen(tmp_path):
    """A pid on another host tells you nothing, so only a cold heartbeat may condemn it."""
    run_dir = _run_dir(tmp_path)
    hold_lock(
        run_dir,
        run_id=RUN_ID,
        hostname_hash=OTHER_HOST,
        pid=999_999,
        heartbeat_utc="2026-07-27T14:22:33.418Z",
    )

    with pytest.raises(RunLockedError) as excinfo:
        acquire_lock(run_dir, run_id=RUN_ID, now=START)

    message = str(excinfo.value)
    assert RUN_ID in message
    assert "999999" in message  # names the holder
    assert str(run_dir / RUN_LOCK_NAME) in message  # …and where the evidence is
    assert _lock(run_dir)["pid"] == 999_999  # the holder's lock is left exactly as it was


def test_a_dead_pid_on_this_host_is_a_stale_lock_stolen_with_a_warning_and_an_event(
    tmp_path, caplog
):
    run_dir = _run_dir(tmp_path)
    dead = _dead_pid()
    hold_lock(run_dir, run_id=RUN_ID, pid=dead)

    with caplog.at_level(logging.WARNING):
        note = acquire_lock(run_dir, run_id=RUN_ID, now=START)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(dead) in warnings[0].getMessage()
    assert note is not None
    assert note.kind == "warn"
    assert note.text == (
        f"stole a stale run lock held by pid {dead}: pid {dead} is not running on this host"
    )
    assert _lock(run_dir)["pid"] == os.getpid()  # …and the lock is ours now


def test_a_cold_heartbeat_is_a_stale_lock_stolen_with_a_warning_and_an_event(tmp_path, caplog):
    run_dir = _run_dir(tmp_path)
    cold = START - timedelta(seconds=STALE_HEARTBEAT_S + 60)
    hold_lock(
        run_dir,
        run_id=RUN_ID,
        hostname_hash=OTHER_HOST,
        pid=999_999,
        heartbeat_utc=f"{cold:%Y-%m-%dT%H:%M:%S}.000Z",
    )

    with caplog.at_level(logging.WARNING):
        note = acquire_lock(run_dir, run_id=RUN_ID, now=START)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "heartbeat" in warnings[0].getMessage()
    assert note is not None
    assert "heartbeat" in note.text
    assert f"{int(STALE_HEARTBEAT_S)}s threshold" in note.text
    assert _lock(run_dir)["pid"] == os.getpid()


def test_an_unheld_run_is_locked_with_no_note_at_all(tmp_path):
    """The ordinary open: nothing was stolen, so the record gets nothing to say about it."""
    run_dir = _run_dir(tmp_path)

    note = acquire_lock(run_dir, run_id=RUN_ID, now=START)

    assert note is None
    assert _lock(run_dir)["run_id"] == RUN_ID


# ── assert_unlocked: the read-only half ────────────────────────────────────────────────────


def test_asserting_unlocked_refuses_a_live_lock_and_names_the_callers_consequence(tmp_path):
    run_dir = _run_dir(tmp_path)
    held = hold_lock(
        run_dir,
        run_id=RUN_ID,
        hostname_hash=OTHER_HOST,
        pid=999_999,
        heartbeat_utc="2026-07-27T14:22:33.418Z",
    )

    with pytest.raises(RunLockedError) as excinfo:
        assert_unlocked(
            run_dir,
            run_id=RUN_ID,
            now=START,
            stale_after_s=STALE_HEARTBEAT_S,
            consequence=CONSEQUENCE,
        )

    assert str(excinfo.value) == (
        f"run {RUN_ID} is open by pid 999999 on host {OTHER_HOST} "
        f"(heartbeat 2026-07-27T14:22:33.418Z), {CONSEQUENCE}"
    )
    assert _lock(run_dir) == held  # read-only: it never takes the lock it just refused


def test_asserting_unlocked_passes_over_no_lock_at_all(tmp_path):
    run_dir = _run_dir(tmp_path)

    assert_unlocked(
        run_dir, run_id=RUN_ID, now=START, stale_after_s=STALE_HEARTBEAT_S, consequence=CONSEQUENCE
    )

    assert not (run_dir / RUN_LOCK_NAME).exists()  # …and takes none either


def test_asserting_unlocked_passes_over_a_stale_lock_and_leaves_it_where_it_is(tmp_path):
    """A crashed run must never need manual cleanup — and a read-only check leaves no trace."""
    run_dir = _run_dir(tmp_path)
    held = hold_lock(run_dir, run_id=RUN_ID, pid=_dead_pid())

    assert_unlocked(
        run_dir, run_id=RUN_ID, now=START, stale_after_s=STALE_HEARTBEAT_S, consequence=CONSEQUENCE
    )

    assert _lock(run_dir) == held


# ── touch and release: what the store drives ───────────────────────────────────────────────


def test_touching_the_lock_writes_this_pid_a_hashed_host_and_the_heartbeat(tmp_path):
    """The record and its lock are meant to be shareable: a machine name is not."""
    run_dir = _run_dir(tmp_path)

    touch_lock(run_dir, run_id=RUN_ID, now=START)

    lock = _lock(run_dir)
    assert lock == {
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "hostname_hash": _our_host_hash(),
        "started_utc": "2026-07-27T14:22:33.418Z",
        "heartbeat_utc": "2026-07-27T14:22:33.418Z",
    }
    assert len(lock["hostname_hash"]) == 12
    assert socket.gethostname() not in (run_dir / RUN_LOCK_NAME).read_text()


def test_touching_the_lock_again_moves_the_heartbeat_forward(tmp_path):
    run_dir = _run_dir(tmp_path)
    touch_lock(run_dir, run_id=RUN_ID, now=START)

    touch_lock(run_dir, run_id=RUN_ID, now=START + timedelta(seconds=600))

    assert _lock(run_dir)["heartbeat_utc"] == "2026-07-27T14:32:33.418Z"


def test_releasing_the_lock_removes_the_file_and_asks_nothing_of_the_caller(tmp_path):
    """Best effort, always attempted: a lock nobody holds must never block the next invocation."""
    run_dir = _run_dir(tmp_path)
    touch_lock(run_dir, run_id=RUN_ID, now=START)

    release_lock(run_dir)
    release_lock(run_dir)  # idempotent: a released lock is released

    assert not (run_dir / RUN_LOCK_NAME).exists()
    assert acquire_lock(run_dir, run_id=RUN_ID, now=START) is None  # no steal, no refusal


def test_a_lock_that_is_not_readable_json_is_no_lock_at_all(tmp_path):
    """Evidence, not a crash: an unparseable lock file cannot name a holder, so it holds nothing."""
    run_dir = _run_dir(tmp_path)
    (run_dir / RUN_LOCK_NAME).write_text("{not json")

    note = acquire_lock(run_dir, run_id=RUN_ID, now=datetime.now(UTC))

    assert note is None
    assert _lock(run_dir)["pid"] == os.getpid()
