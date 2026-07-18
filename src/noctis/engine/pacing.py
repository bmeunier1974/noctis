"""The clock-pacing seam — every "wait in wall-clock time" concern in one module.

A :class:`Sleeper` is the time source plus ``sleep_until``: in production :class:`RealSleeper`
blocks against the wall clock; in tests and replay :class:`SimulatedSleeper` advances a
compressed clock instantly, so the same loops run either at market speed or at CPU speed with
no code change. The live TRADING driver paces each feed poll with a bare sleeper; the run loop
paces *between phases* through :class:`BoundedWaiter`, which layers a stop-request poll and the
run's time-limit deadline over the same seam. :class:`StopFlag` (the stop-signal adapter) and
:class:`StallGuard` (the frozen-clock safety net) complete the run loop's pacing bookkeeping,
kept here so a pacing bug has exactly one home.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol

# Longest a real-time wall-clock wait blocks before re-checking the stop flag. A SIGINT/SIGTERM
# handler only *sets* the stop flag (it never raises), and ``time.sleep`` resumes after the
# handler returns (PEP 475), so a single long ``sleep_until`` — e.g. parked over a weekend until
# the next open — would ignore the flag until it elapsed. Polling in chunks bounds Ctrl+C
# latency to this.
_STOP_POLL_SECONDS = 1.0


class Sleeper(Protocol):
    """A time source plus a ``sleep_until`` the driver paces each poll with.

    ``wall_clock`` says whether ``now`` advances on its own (real time) or only when
    ``sleep_until`` is called (a compressed clock). The run loop reads it to decide whether
    a closed market can be filled with real work or must be jumped over.
    """

    wall_clock: bool

    def now(self) -> datetime: ...

    def sleep_until(self, t: datetime) -> None: ...


class RealSleeper:
    """Production pacing: reads wall-clock time and blocks until ``t``."""

    # ``now`` advances with the wall clock whether or not anyone calls ``sleep_until``.
    wall_clock = True

    def __init__(self):
        self._sleep = time.sleep

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep_until(self, t: datetime) -> None:
        remaining = (t - self.now()).total_seconds()
        if remaining > 0:
            self._sleep(remaining)


class SimulatedSleeper:
    """Test/replay pacing: a compressed clock that jumps forward instantly, never blocks."""

    # ``now`` is frozen until ``sleep_until``/``advance`` moves it, so closed stretches must
    # be jumped over — there is no real time in which to do work.
    wall_clock = False

    def __init__(self, start: datetime):
        self._t = start if start.tzinfo is not None else start.replace(tzinfo=UTC)

    def now(self) -> datetime:
        return self._t

    def sleep_until(self, t: datetime) -> None:
        target = t if t.tzinfo is not None else t.replace(tzinfo=UTC)
        if target > self._t:
            self._t = target

    def advance(self, seconds: float) -> None:
        self._t = self._t + timedelta(seconds=seconds)


class BoundedWaiter:
    """Paces the run loop to a target time — but never past a stop request or the deadline.

    The one home for the "bounded wait" the phase loop needs between work:

    * **Deadline clamp.** The machine's time-up check runs only after a wait, so without the
      clamp a run started while the market is closed (e.g. a weekend) would block in
      ``RealSleeper.sleep_until`` until the next open before an already-elapsed
      ``--time-limit-hours`` could take effect.
    * **Stop polling.** ``RealSleeper.sleep_until`` is a single ``time.sleep``; a SIGINT/SIGTERM
      handler only *sets* the stop flag (it never raises), and the sleep resumes after the
      handler returns (PEP 475) — so a Ctrl+C mid-weekend would appear to hang until the open.
      Under wall-clock pacing the wait runs in ``poll_seconds`` chunks and re-checks ``stop``
      between them, bounding stop latency to one chunk.
    * **Simulated single jump.** A non-wall-clock sleeper advances instantly, so it keeps a
      single ``sleep_until`` call — chunking it would loop thousands of times over a long
      closed stretch for no gain.
    """

    def __init__(
        self,
        sleeper: Sleeper,
        *,
        stop: Callable[[], bool],
        deadline: datetime | None = None,
        poll_seconds: float = _STOP_POLL_SECONDS,
    ):
        self._sleeper = sleeper
        self._stop = stop
        self._deadline = deadline
        self._poll = timedelta(seconds=poll_seconds)

    @property
    def wall_clock(self) -> bool:
        """Whether the underlying clock advances on its own (the sleeper's flag, defaulted)."""
        return bool(getattr(self._sleeper, "wall_clock", False))

    def wait_until(self, target: datetime) -> None:
        """Block until ``target``, clamped to the deadline, waking promptly on a stop."""
        if self._stop():
            return
        if self._deadline is not None and self._deadline < target:
            target = self._deadline
        if not self.wall_clock:
            if self._sleeper.now() < target:
                self._sleeper.sleep_until(target)
            return
        while not self._stop() and self._sleeper.now() < target:
            self._sleeper.sleep_until(min(target, self._sleeper.now() + self._poll))


class StopFlag:
    """Adapts a bool-returning callable to the ``is_set()`` protocol stop-events expose.

    The runtime holds one ``_stop`` bool that a signal handler flips; the research and
    trading loops poll an event-like object (``StopEvent``). This is the one adapter
    between the two views of that signal.
    """

    def __init__(self, fn: Callable[[], bool]):
        self._fn = fn

    def is_set(self) -> bool:
        return bool(self._fn())


class StallGuard:
    """Safety net against an unexpected *non-advancing* clock in the run loop.

    Counts only consecutive observations in which the clock did not move forward, so a
    healthy real-time run that fills a closed market with many back-to-back research
    sessions never trips it — only a genuinely stuck clock (e.g. next_open <= now) does.
    """

    def __init__(self, limit: int = 5000):
        self._limit = limit
        self._count = 0
        self._last: datetime | None = None

    def stalled(self, now: datetime) -> bool:
        """Observe the clock once; True after ``limit`` consecutive frozen observations."""
        if self._last is None or now > self._last:
            self._last = now
            self._count = 0
            return False
        self._count += 1
        return self._count > self._limit
