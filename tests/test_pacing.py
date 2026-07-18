"""Unit tests for the run loop's pacing seam — BoundedWaiter, StallGuard, StopFlag.

The waiter is the one home for "wait, but never past a stop request or the run's time
limit"; these tests drive that interface directly against recording clocks — no Runtime,
no lake, no settings. The runtime-level proofs (an expired limit never waits out a closed
weekend or a session close) stay in test_smoke_cycle.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from noctis.engine.pacing import BoundedWaiter, StallGuard, StopFlag

ET = ZoneInfo("America/New_York")

SATURDAY = datetime(2027, 1, 2, 6, 0, tzinfo=ET)
MONDAY_OPEN = datetime(2027, 1, 4, 9, 30, tzinfo=ET)  # ~51h later


class _RecordingSleeper:
    """A simulated clock that records every ``sleep_until`` target. Deliberately has no
    ``wall_clock`` attribute — the waiter must default a bare sleeper to simulated pacing."""

    def __init__(self, start):
        self._t = start
        self.waits: list = []

    def now(self):
        return self._t

    def sleep_until(self, t):
        self.waits.append(t)
        if t > self._t:
            self._t = t


class _RecordingWallClock(_RecordingSleeper):
    """A controllable clock that *reports* as real-time pacing, with a hook fired after each
    wait so a test can trip a stop mid-wait exactly as a SIGINT handler would."""

    wall_clock = True

    def __init__(self, start):
        super().__init__(start)
        self.on_wait = lambda: None

    def sleep_until(self, t):
        super().sleep_until(t)
        self.on_wait()


class _Stop:
    def __init__(self, value: bool = False):
        self.value = value

    def __call__(self) -> bool:
        return self.value


def test_wall_clock_wait_chunks_and_honors_stop_mid_wait():
    """A real-time wait parked over a long closed stretch polls the stop flag in short chunks,
    so a Ctrl+C mid-wait (which only *sets* the flag, never raises) returns within one chunk
    instead of sleeping straight through to the next open — the RealSleeper weekend hang."""
    sleeper = _RecordingWallClock(SATURDAY)
    stop = _Stop()
    waiter = BoundedWaiter(sleeper, stop=stop)

    # Trip the stop after the 3rd poll chunk, as a signal handler would fire mid-wait.
    def _maybe_stop():
        if len(sleeper.waits) >= 3:
            stop.value = True

    sleeper.on_wait = _maybe_stop
    waiter.wait_until(MONDAY_OPEN)

    assert len(sleeper.waits) == 3  # woke after 3 chunks, not one 51-hour sleep
    assert sleeper.now() < MONDAY_OPEN  # never reached the open
    # each chunk is a short poll interval, never a single jump to the far target
    assert all((w - SATURDAY).total_seconds() <= 3600 for w in sleeper.waits)


def test_simulated_wait_stays_a_single_jump():
    """Contrast: a non-wall-clock (simulated/replay) sleeper still jumps to the target in one
    call — chunking it would loop thousands of times over a long closed stretch for no gain."""
    sleeper = _RecordingSleeper(SATURDAY)  # no wall_clock attr
    waiter = BoundedWaiter(sleeper, stop=_Stop())
    waiter.wait_until(MONDAY_OPEN)

    assert sleeper.waits == [MONDAY_OPEN]  # one jump straight to the target
    assert sleeper.now() == MONDAY_OPEN


def test_wait_clamps_to_the_deadline():
    """A target past the run's deadline is clamped: the waiter paces only to the deadline,
    so an elapsed --time-limit-hours never parks the run against the market calendar."""
    deadline = SATURDAY + timedelta(minutes=3)
    sleeper = _RecordingSleeper(SATURDAY)
    waiter = BoundedWaiter(sleeper, stop=_Stop(), deadline=deadline)
    waiter.wait_until(MONDAY_OPEN)

    assert sleeper.waits == [deadline]
    assert sleeper.now() == deadline


def test_stop_already_set_skips_the_wait_entirely():
    sleeper = _RecordingWallClock(SATURDAY)
    waiter = BoundedWaiter(sleeper, stop=_Stop(True))
    waiter.wait_until(MONDAY_OPEN)
    assert sleeper.waits == []


def test_wait_for_a_past_target_never_sleeps():
    sleeper = _RecordingSleeper(MONDAY_OPEN)
    waiter = BoundedWaiter(sleeper, stop=_Stop())
    waiter.wait_until(SATURDAY)
    assert sleeper.waits == []
    assert sleeper.now() == MONDAY_OPEN


def test_stall_guard_trips_only_on_a_frozen_clock():
    guard = StallGuard(limit=3)
    assert not guard.stalled(SATURDAY)  # first observation primes it
    for _ in range(3):
        assert not guard.stalled(SATURDAY)  # counting toward the limit
    assert guard.stalled(SATURDAY)  # one past the limit trips

    # any forward movement resets the count — back-to-back research never trips it
    assert not guard.stalled(SATURDAY + timedelta(seconds=1))
    assert not guard.stalled(SATURDAY + timedelta(seconds=1))


def test_stop_flag_is_a_live_view_of_the_callable():
    stop = _Stop()
    flag = StopFlag(stop)
    assert not flag.is_set()
    stop.value = True
    assert flag.is_set()
