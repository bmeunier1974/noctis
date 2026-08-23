"""Account continuity: one continuous paper account across sessions (live-holdout plan 2).

``state/paper_account.json`` carries equity AND open positions across TRADING sessions —
loaded at TRADING entry, saved after each session's finalize — so the rolling live-holdout
accumulates one genuine multi-day equity curve. Covers the AccountStore (inception,
round trip, corrupt → hard error, reset/archive), the TRADING phase (carried
equity + positions, overnight gap in day-2 P&L, daily-loss anchor on carried equity,
corrupt-file refusal, no fill double-counting), and the carried-position hold at the
session open.
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from noctis.broker.persistence import AccountStore
from noctis.broker.seam import Order, Side
from noctis.engine.sessions import SessionLedger

from ._session_helpers import (
    ET,
    _account_path,
    _bars_local,
    _concat,
    _FakeLake,
    _ledger_path,
    _make_runtime,
    _run_phase,
    _uptrend,
)


def _at(trade) -> datetime:
    """When a report trade filled — the row's own UTC stamp, parsed back."""
    return datetime.fromisoformat(trade.ts.replace("Z", "+00:00")).astimezone(ET)


# --- AccountStore --------------------------------------------------------------------------


def test_store_inception_then_round_trip(tmp_path):
    path = tmp_path / "state" / "paper_account.json"
    store = AccountStore(path)
    broker = store.load()
    assert broker.equity() == 100_000.0  # absent file → fresh account
    assert not path.exists()  # the file appears at first save, not at load

    broker.set_price("AAPL", 100.0, ts_event=7)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    store.save(broker, date(2026, 7, 6))
    assert not path.with_suffix(".json.tmp").exists()  # tmp replaced, not left behind

    fresh = AccountStore(path)  # a restart
    loaded = fresh.load()
    assert loaded.equity() == pytest.approx(broker.equity())
    assert loaded.position("AAPL").quantity == 10
    assert loaded.fills == []  # fills are per-session report material, never persisted
    assert fresh.opened == "2026-07-06"
    assert fresh.last_session == "2026-07-06"


def test_store_opened_is_inception_and_survives_later_sessions(tmp_path):
    store = AccountStore(tmp_path / "paper_account.json")
    broker = store.load()
    store.save(broker, date(2026, 7, 6))
    store.save(broker, date(2026, 7, 7))
    reread = AccountStore(store.path)
    reread.load()
    assert reread.opened == "2026-07-06"  # inception sticks
    assert reread.last_session == "2026-07-07"


@pytest.mark.parametrize(
    "text",
    [
        "{not json",  # unreadable
        '{"version": 99, "cash": 1.0}',  # future/unknown version
        '{"version": 1, "cash": 1.0}',  # missing required fields
        '{"version": 1, "starting_cash": "x"}',  # wrong types
    ],
)
def test_store_corrupt_file_is_a_hard_error_and_left_untouched(tmp_path, text):
    path = tmp_path / "paper_account.json"
    path.write_text(text)
    with pytest.raises(RuntimeError, match="corrupt paper account"):
        AccountStore(path).load()
    assert path.read_text() == text  # never mutated by the failed load


def test_store_reset_archives_and_starts_fresh(tmp_path):
    store = AccountStore(tmp_path / "paper_account.json")
    broker = store.load()
    broker.set_price("AAPL", 100.0)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    store.save(broker, date(2026, 7, 6))

    archive = store.reset()
    assert archive is not None and archive.is_file()
    assert archive.name.startswith("paper_account.") and archive.suffix == ".json"
    assert not store.path.exists()
    assert store.load().equity() == 100_000.0  # next session starts fresh
    assert store.reset() is None  # nothing left to reset


def test_store_summary_reads_without_side_effects(tmp_path):
    store = AccountStore(tmp_path / "paper_account.json")
    assert store.summary() is None
    broker = store.load()
    broker.set_price("AAPL", 100.0)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    broker.set_price("AAPL", 120.0)
    store.save(broker, date(2026, 7, 6))

    summary = AccountStore(store.path).summary()
    assert summary is not None
    assert summary.starting_cash == 100_000.0
    assert summary.equity == pytest.approx(broker.equity())
    assert summary.cumulative_pnl == pytest.approx(broker.equity() - 100_000.0)
    assert summary.open_positions == 1
    assert summary.opened == "2026-07-06"


def test_store_summarize_is_the_summary_of_one_load(tmp_path):
    """``summary()`` IS ``summarize(load())``: the view is pure over an already-loaded broker
    plus the store's own provenance, so a caller that needs both the broker and the summary
    parses the file once (story #266)."""
    store = AccountStore(tmp_path / "paper_account.json")
    broker = store.load()
    broker.set_price("AAPL", 100.0)
    broker.submit_order(Order("AAPL", Side.BUY, 10))
    broker.set_price("AAPL", 120.0)
    store.save(broker, date(2026, 7, 6))
    store.save(broker, date(2026, 7, 7))

    reread = AccountStore(store.path)
    summarized = reread.summarize(reread.load())

    assert summarized == AccountStore(store.path).summary()  # same numbers, same provenance
    assert summarized.equity == pytest.approx(broker.equity())
    assert summarized.cumulative_pnl == pytest.approx(broker.equity() - 100_000.0)
    assert summarized.open_positions == 1
    assert summarized.opened == "2026-07-06" and summarized.last_session == "2026-07-07"


def test_store_summarize_reads_the_broker_it_is_given(tmp_path):
    """The summary describes the broker handed in — not a re-read of the file — so the close
    can summarize the account it already loaded."""
    store = AccountStore(tmp_path / "paper_account.json")
    store.save(store.load(), date(2026, 7, 6))
    other = AccountStore(store.path).load()
    other.set_price("AAPL", 100.0)
    other.submit_order(Order("AAPL", Side.BUY, 10))
    other.set_price("AAPL", 150.0)

    summarized = store.summarize(other)

    assert summarized.equity == pytest.approx(other.equity())
    assert summarized.open_positions == 1
    assert summarized.opened == "2026-07-06"  # provenance still the store's


# --- runtime wiring ------------------------------------------------------------------------


def test_two_sessions_carry_equity_and_positions_and_gap_hits_day2(tmp_path):
    """Session 2 starts exactly where session 1 ended (the phase's one carried broker, at
    the persisted prior-close marks), the open day-1 position is still on the book, and the
    overnight gap lands in day-2 P&L."""
    d1, d2 = date(2026, 3, 4), date(2026, 3, 5)
    day1 = _uptrend(30)  # closes 100 → 158; the champion ends long
    day2 = _uptrend(30, start=200.0)  # gaps up 158 → 200 overnight, then keeps trending
    lake = _FakeLake({"AAPL": _concat(_bars_local(d1, day1), _bars_local(d2, day2))})
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(d1 - pd.Timedelta(days=1))  # both unseen

    outcome = _run_phase(runtime)

    assert len(outcome.sessions) == 2
    s1, s2 = outcome.sessions
    qty1 = s1.positions.get("AAPL", 0.0)
    assert qty1 > 0  # day 1 ended long — the position rides overnight
    assert s2.start_equity == pytest.approx(s1.final_equity)
    # Day-2 P&L includes the overnight gap on the carried quantity (the day itself also
    # trends up, so the session P&L is at least the gap minus rebalancing costs).
    gap_pnl = qty1 * (day2[0] - day1[-1])
    assert s2.final_equity - s2.start_equity > gap_pnl * 0.9

    # Restart continuity: the persisted account picks up where the run left off.
    data = json.loads(_account_path(runtime).read_text())
    assert data["opened"] == d1.isoformat() and data["last_session"] == d2.isoformat()
    carried = AccountStore(_account_path(runtime)).load()
    assert carried.equity() == pytest.approx(s2.final_equity)
    assert carried.position("AAPL").quantity == pytest.approx(s2.positions["AAPL"])


def test_carried_position_is_not_flattened_at_the_next_open(tmp_path):
    """A fresh _TradingSession seeds its pending target from the carried position's sign,
    so the first bars of day 2 (before the strategy's first decision) hold the position
    instead of force-flattening it — turnover the strategy never chose."""
    d1, d2 = date(2026, 3, 4), date(2026, 3, 5)
    lake = _FakeLake(
        {"AAPL": _concat(_bars_local(d1, _uptrend(30)), _bars_local(d2, _uptrend(30, 160.0)))}
    )
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(d1 - pd.Timedelta(days=1))

    outcome = _run_phase(runtime)

    s1, s2 = outcome.sessions
    qty1 = s1.positions["AAPL"]
    day2_trades = [t for t in outcome.trades if _at(t).date() == s2.session]
    # No day-2 fill closes the whole carried position (resizes are fine; a flatten-to-zero
    # sell of the full quantity is exactly the forced turnover this guards against).
    assert all(not (t.side == Side.SELL and t.quantity >= qty1 * 0.999) for t in day2_trades), (
        day2_trades
    )
    assert s2.positions["AAPL"] > 0  # still long at day-2 close


def test_daily_loss_anchors_to_carried_equity(tmp_path):
    """`RiskManager(limits, broker.equity())` with a carried broker anchors the day's loss
    floor to that day's carried starting equity — 90k after a losing streak, never a
    fictional fresh 100k."""
    day = date(2026, 3, 5)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend())})
    runtime = _make_runtime(tmp_path, lake)
    # A prior losing run left the account at 90k cash, flat.
    store = AccountStore(_account_path(runtime))
    broker = store.load()
    broker.cash = 90_000.0
    store.save(broker, day - pd.Timedelta(days=1))

    outcome = _run_phase(runtime)

    assert outcome.sessions[0].start_equity == pytest.approx(90_000.0)
    # And the sizing follows the carried equity: 10% max_position_pct of 90k, not 100k.
    buys = [t for t in outcome.trades if t.side == Side.BUY]
    assert buys and buys[0].quantity * buys[0].price <= 9_000.0 * 1.01


def test_corrupt_account_refuses_trading_with_event_and_file_untouched(tmp_path, caplog):
    day = date(2026, 3, 5)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend())})
    runtime = _make_runtime(tmp_path, lake)
    path = _account_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{torn write")

    with caplog.at_level("ERROR", logger="noctis.runtime"):
        outcome = _run_phase(runtime)

    assert outcome.sessions == []  # no session ran, nothing traded
    assert "corrupt paper account" in caplog.text
    assert any(e.startswith("Trading refused — corrupt paper account") for e in outcome.events)
    assert path.read_text() == "{torn write"  # evidence preserved for the operator
    with pytest.raises(RuntimeError, match="corrupt paper account"):
        AccountStore(path).load()  # still refuses — nothing was rewritten over the evidence
    assert SessionLedger(_ledger_path(runtime)).load() is None  # high-water mark untouched


def test_inception_file_created_after_first_session(tmp_path):
    day = date(2026, 3, 4)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend())})
    runtime = _make_runtime(tmp_path, lake)
    assert not _account_path(runtime).exists()

    outcome = _run_phase(runtime)

    assert outcome.sessions[0].start_equity == 100_000.0  # inception: fresh 100k
    data = json.loads(_account_path(runtime).read_text())
    assert data["version"] == 1
    assert data["opened"] == day.isoformat() == data["last_session"]


def test_no_double_count_of_prior_session_fills(tmp_path):
    """Across a two-session catch-up on one carried broker, each session's report trades
    and fill count cover only its own fills."""
    d1, d2 = date(2026, 3, 4), date(2026, 3, 5)
    lake = _FakeLake(
        {"AAPL": _concat(_bars_local(d1, _uptrend(30)), _bars_local(d2, _uptrend(30, 160.0)))}
    )
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(d1 - pd.Timedelta(days=1))

    outcome = _run_phase(runtime)

    s1, s2 = outcome.sessions
    assert s1.fills > 0 and s2.fills > 0
    # Each fill is reported exactly once: the per-session counts partition the phase's trades.
    assert s1.fills + s2.fills == len(outcome.trades)


def test_crash_between_account_and_ledger_saves_retraces_safely(tmp_path, monkeypatch):
    """The account is saved before the high-water mark advances, so a crash between the
    two writes re-trades the session on restart (ledger behind account) — never skips it."""
    day = date(2026, 3, 4)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend())})
    runtime = _make_runtime(tmp_path, lake)

    real_save = SessionLedger.save

    def crashy_save(self, last_traded):
        raise RuntimeError("boom between account and ledger")

    monkeypatch.setattr(SessionLedger, "save", crashy_save)
    with pytest.raises(RuntimeError, match="boom between account and ledger"):
        _run_phase(runtime)
    monkeypatch.setattr(SessionLedger, "save", real_save)

    assert _account_path(runtime).exists()  # account committed first…
    assert SessionLedger(_ledger_path(runtime)).load() is None  # …ledger still behind
    outcome = _run_phase(runtime)  # restart
    assert len(outcome.sessions) == 1  # the session is re-traded, not skipped
    assert SessionLedger(_ledger_path(runtime)).load() == day
