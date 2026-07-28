"""The trading state machine — RESEARCH ↔ TRADING → CLOSE → RESEARCH, plus STOPPED.

Transitions are driven by the market clock: the machine researches while the market is
closed, trades while it is open, runs the close phase once when the market closes, then
returns to research. Every transition — including the terminal one — goes through the
machine's own surface: the clock drives ``tick``, the global time limit stops from any
state, and an external stop request (operator signal, a test's max-cycles cap) arrives
via ``stop()``. The machine is tick-driven for deterministic testing; the runtime drives
the ticks in wall-clock time and observes each entry through the ``on_enter`` callback.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from noctis.engine.clock import MarketClock

# The two ceilings that can stop the loop, and the ``stopped_reason`` each one writes. They differ
# only in what they bound — this *process* (how long tonight lasts) versus the whole *run* across
# every stop/resume (the experiment's compute) — and they stop through exactly the same move, so a
# run-level cap behaves like the per-process limit an operator already trusts.
TIME_LIMIT = "time_limit"
RUN_LIMIT = "run_limit"


class Phase(StrEnum):
    RESEARCH = "RESEARCH"
    TRADING = "TRADING"
    CLOSE = "CLOSE"
    STOPPED = "STOPPED"


class TradingMachine:
    """Market-clock-driven state machine."""

    def __init__(
        self,
        clock: MarketClock,
        on_enter: Callable[[Phase], None] | None = None,
        time_limit_hours: float | None = None,
        run_limit_hours: float | None = None,
        prior_runtime_s: float = 0.0,
    ):
        self.clock = clock
        # Invoked once per phase entry (the runtime's phase banner). None on a bare run.
        self.on_enter = on_enter
        self.time_limit_hours = time_limit_hours
        # The run-level compute cap (story #136) and the runtime the run's earlier segments already
        # spent against it. Both default to "no cap", so a machine that was never given one answers
        # every question about time exactly as it always did.
        self.run_limit_hours = run_limit_hours
        self.prior_runtime_s = prior_runtime_s
        self.state: Phase = Phase.STOPPED
        self.start_time: datetime | None = None
        self.history: list[Phase] = []

    def initial_state(self, now: datetime) -> Phase:
        return Phase.TRADING if self.clock.is_open(now) else Phase.RESEARCH

    def start(self, now: datetime | None = None) -> Phase:
        now = now or self.clock.now()
        self.start_time = now
        self.state = self.initial_state(now)
        self.history = [self.state]
        if self.on_enter:
            self.on_enter(self.state)
        return self.state

    def time_up(self, now: datetime) -> bool:
        if self.time_limit_hours is None or self.start_time is None:
            return False
        return (now - self.start_time).total_seconds() >= self.time_limit_hours * 3600.0

    def run_limit_reached(self, now: datetime) -> bool:
        """Whether the whole **run** — this segment plus every earlier one — has spent its cap.

        The per-run twin of :meth:`time_up`, and deliberately the same shape: the run's accumulated
        runtime is just this segment's elapsed time added to what the record says the earlier
        segments used, so "100 hours, then stop" survives being stopped each morning and resumed
        each night.
        """
        if self.run_limit_hours is None or self.start_time is None:
            return False
        elapsed = self.prior_runtime_s + (now - self.start_time).total_seconds()
        return elapsed >= self.run_limit_hours * 3600.0

    def limit_hit(self, now: datetime) -> str | None:
        """Which ceiling has been reached — ``"time_limit"``, ``"run_limit"`` — or ``None``.

        One place both are asked, so both stop through the same transition and the loop reports
        which one fired without deciding anything itself. The per-process limit is checked first,
        so a process that would have stopped tonight anyway keeps saying exactly what it always
        said when both ceilings land together.
        """
        if self.time_up(now):
            return TIME_LIMIT
        if self.run_limit_reached(now):
            return RUN_LIMIT
        return None

    def deadline(self) -> datetime | None:
        """The wall-clock moment this machine must have stopped by, or ``None`` when unbounded.

        The earlier of the two ceilings, so the run loop's bounded waits (a weekend, a session
        close) can never park past *either* — the same clamp the per-process limit has always had,
        now honest about both.
        """
        if self.start_time is None:
            return None
        deadlines = []
        if self.time_limit_hours is not None:
            deadlines.append(self.start_time + timedelta(hours=self.time_limit_hours))
        if self.run_limit_hours is not None:
            remaining = max(0.0, self.run_limit_hours * 3600.0 - self.prior_runtime_s)
            deadlines.append(self.start_time + timedelta(seconds=remaining))
        return min(deadlines) if deadlines else None

    def tick(self, now: datetime) -> Phase:
        """Advance the machine given the current time. Returns the (possibly new) state."""
        if self.state is Phase.STOPPED:
            return self.state
        if self.limit_hit(now) is not None:
            return self.stop()

        if self.state is Phase.RESEARCH and self.clock.is_open(now):
            self._go(Phase.TRADING)
        elif self.state is Phase.TRADING and not self.clock.is_open(now):
            self._go(Phase.CLOSE)
        elif self.state is Phase.CLOSE:
            self._go(Phase.RESEARCH)
        return self.state

    def stop(self) -> Phase:
        """Transition to STOPPED from any state (idempotent) — the one terminal move.

        External stop reasons (an operator signal, a max-cycles cap) come through here,
        so the terminal transition records history and fires ``on_enter`` like any other.
        """
        if self.state is not Phase.STOPPED:
            self._go(Phase.STOPPED)
        return self.state

    def _go(self, new: Phase) -> None:
        self.state = new
        self.history.append(new)
        if self.on_enter:
            self.on_enter(new)


def initial_phase_for(clock: MarketClock, now: datetime | None = None) -> Phase:
    """Convenience: the phase ``run`` would start in for the current time."""
    now = now or clock.now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return Phase.TRADING if clock.is_open(now) else Phase.RESEARCH
