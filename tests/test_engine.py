"""Market clock and the trading state machine."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from noctis.engine import MarketClock, Phase, TradingMachine, initial_phase_for

ET = ZoneInfo("America/New_York")


def et(y, mo, d, h, mi) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


# --- clock -------------------------------------------------------------------------------


def test_clock_is_open_within_session():
    clock = MarketClock("XNYS", "America/New_York")
    assert clock.is_open(et(2026, 1, 5, 10, 0)) is True  # Monday 10:00 ET
    assert clock.is_open(et(2026, 1, 5, 9, 0)) is False  # before open
    assert clock.is_open(et(2026, 1, 5, 16, 0)) is False  # at close (exclusive)
    assert clock.is_open(et(2026, 1, 3, 10, 0)) is False  # Saturday


def test_clock_next_open_and_close():
    clock = MarketClock("XNYS", "America/New_York")
    nxt_open = clock.next_open(et(2026, 1, 5, 6, 0)).astimezone(ET)
    assert (nxt_open.hour, nxt_open.minute) == (9, 30)
    assert nxt_open.date().isoformat() == "2026-01-05"

    nxt_close = clock.next_close(et(2026, 1, 5, 10, 0)).astimezone(ET)
    assert (nxt_close.hour, nxt_close.minute) == (16, 0)


def test_clock_next_open_rolls_over_weekend():
    clock = MarketClock("XNYS", "America/New_York")
    # Friday after close → next open is Monday.
    nxt = clock.next_open(et(2026, 1, 2, 17, 0)).astimezone(ET)
    assert nxt.weekday() == 0  # Monday


# --- machine transitions -----------------------------------------------------------------


def test_initial_state_selection():
    clock = MarketClock("XNYS", "America/New_York")
    assert initial_phase_for(clock, et(2026, 1, 5, 10, 0)) is Phase.TRADING
    assert initial_phase_for(clock, et(2026, 1, 5, 6, 0)) is Phase.RESEARCH


def test_machine_walks_research_trading_close_research():
    clock = MarketClock("XNYS", "America/New_York")
    entered: list[Phase] = []
    machine = TradingMachine(clock, on_enter=entered.append)

    machine.start(et(2026, 1, 5, 6, 0))  # pre-open Monday → RESEARCH
    machine.tick(et(2026, 1, 5, 10, 0))  # open → TRADING
    machine.tick(et(2026, 1, 5, 12, 0))  # still open → stays TRADING
    machine.tick(et(2026, 1, 5, 17, 0))  # closed → CLOSE
    machine.tick(et(2026, 1, 5, 18, 0))  # CLOSE is transient → RESEARCH

    assert machine.history == [Phase.RESEARCH, Phase.TRADING, Phase.CLOSE, Phase.RESEARCH]
    assert entered == machine.history


def test_stop_is_the_one_terminal_move_and_idempotent():
    """External stops (operator signal, max-cycles) go through ``stop()``: the terminal
    transition records history and fires ``on_enter`` like any other, exactly once."""
    clock = MarketClock("XNYS", "America/New_York")
    entered: list[Phase] = []
    machine = TradingMachine(clock, on_enter=entered.append)
    machine.start(et(2026, 1, 5, 6, 0))  # RESEARCH

    assert machine.stop() is Phase.STOPPED
    assert machine.history == [Phase.RESEARCH, Phase.STOPPED]
    assert entered == machine.history
    machine.stop()  # idempotent: no duplicate history entry, no re-fired hook
    assert machine.history == [Phase.RESEARCH, Phase.STOPPED]
    machine.tick(et(2026, 1, 5, 10, 0))  # a stopped machine never restarts on tick
    assert machine.state is Phase.STOPPED


def test_time_limit_stops_from_research():
    clock = MarketClock("XNYS", "America/New_York")
    machine = TradingMachine(clock, time_limit_hours=1.0)
    machine.start(et(2026, 1, 5, 6, 0))  # RESEARCH
    machine.tick(et(2026, 1, 5, 7, 30))  # 1.5h > 1h limit
    assert machine.state is Phase.STOPPED


def test_time_limit_stops_from_trading():
    clock = MarketClock("XNYS", "America/New_York")
    machine = TradingMachine(clock, time_limit_hours=1.0)
    machine.start(et(2026, 1, 5, 10, 0))  # TRADING (open)
    machine.tick(et(2026, 1, 5, 12, 0))  # 2h > 1h limit
    assert machine.state is Phase.STOPPED
