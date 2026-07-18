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
from datetime import UTC, datetime
from enum import StrEnum

from noctis.engine.clock import MarketClock


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
    ):
        self.clock = clock
        # Invoked once per phase entry (the runtime's phase banner). None on a bare run.
        self.on_enter = on_enter
        self.time_limit_hours = time_limit_hours
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

    def tick(self, now: datetime) -> Phase:
        """Advance the machine given the current time. Returns the (possibly new) state."""
        if self.state is Phase.STOPPED:
            return self.state
        if self.time_up(now):
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
