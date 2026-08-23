"""The run's liveness lock — the one fatal failure in the whole run tree (story #286, epic #284).

Everything else in this package is latched (a reporting artifact must never take down a
multi-week run). A lock is different in kind: two engines writing one run's record — and one
champion registry, one paper account — is *corruption*, not degradation. So a live lock is a hard,
informative refusal, never a silent downgrade to "write anyway". A **stale** lock is stealable,
because a crashed run must not need manual cleanup before it can be resumed: stale means a dead
pid on *this* host (a pid on another host tells you nothing about whether it is alive) or a
heartbeat gone colder than :data:`STALE_HEARTBEAT_S`. A steal is loud — one warning here and an
event on the record — because it is the one moment a run's history could be attributed to the
wrong process.

The module imports nothing from :mod:`noctis.reporting.run_tree`: a lock is a file with a pid, a
hashed host and a heartbeat, and deciding whether it may be taken needs no record, no index and no
addressing. Its four verbs are what the store drives — :func:`acquire_lock` at an open,
:func:`assert_unlocked` for the read-only refusals (sealing, pruning), :func:`touch_lock` at each
checkpoint and :func:`release_lock` at close.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from noctis.reporting.run_record import RecordEvent, utc_iso

logger = logging.getLogger(__name__)

RUN_LOCK_NAME = "run.lock"

# How cold a heartbeat must be before a lock we cannot otherwise check counts as abandoned.
# Deliberately generous — a week. The heartbeat is touched at each CLOSE, i.e. roughly once per
# trading day, and a live engine can sit in RESEARCH right through a long weekend; anything
# tighter would steal the lock from a running process, which is the exact corruption the refusal
# exists to prevent. The same-host dead-pid check is what catches the common crash promptly.
STALE_HEARTBEAT_S = 7 * 24 * 3600.0


class RunLockedError(RuntimeError):
    """Another engine holds this run. The one hard refusal — see the module docstring."""


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


def assert_unlocked(
    run_dir: Path, *, run_id: str, now: datetime, stale_after_s: float, consequence: str
) -> None:
    """Refuse when another engine is live on this run — the read-only half of :func:`acquire_lock`.

    Deliberately does not *take* the lock: sealing (and pruning) is one write, and a lock file left
    behind by a command that started nothing would be friction the next invocation has to reason
    about. A stale lock (a dead pid here, a heartbeat gone cold) is no obstacle at all, for the same
    reason a resume may steal one — a crashed run must never need manual cleanup.

    ``consequence`` is the caller's half of the sentence: the holder is named identically whatever
    was asked for, and what would have gone wrong differs.
    """
    held = _read_lock(run_dir / RUN_LOCK_NAME)
    if held is None or _stale_reason(held, now=now, stale_after_s=stale_after_s) is not None:
        return
    raise RunLockedError(
        f"run {run_id} is open by pid {held.get('pid')} on host {held.get('hostname_hash')} "
        f"(heartbeat {held.get('heartbeat_utc')}), {consequence}"
    )


def touch_lock(run_dir: Path, *, run_id: str, now: datetime) -> None:
    """Re-stamp the lock this process already holds, so the heartbeat says it is still alive.

    Driven at each checkpoint, which is roughly once per trading day: the heartbeat is the only
    evidence a *different host* can ever have about this process, and a lock that stops being
    touched is what :data:`STALE_HEARTBEAT_S` eventually condemns.
    """
    _write_lock(run_dir / RUN_LOCK_NAME, run_id=run_id, started=now, heartbeat=now)


def release_lock(run_dir: Path) -> None:
    """Best effort, always attempted: a stale lock file is friction for the next invocation,
    never a reason to fail this one."""
    try:
        (run_dir / RUN_LOCK_NAME).unlink(missing_ok=True)
    except OSError:  # pragma: no cover - a lock we cannot remove is the next open's problem
        pass


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
    are still provably the same host, without publishing a machine name. The hashing itself lives
    in ``observability.environment`` (story #139), so the lock and the record's per-segment
    environment block cannot drift into two different answers for one machine.
    """
    from noctis.observability.environment import hostname_hash

    return hostname_hash(socket.gethostname())


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
