"""TRADING-phase session slicing: the rolling live-holdout (live-holdout plan 1).

Each day the replay trades only the newest lake session(s) past the high-water mark in
``state/trading_sessions.json`` — one risk-managed session per session date — instead of
the whole catalog. Covers the pure slicing functions, the ledger, and the TRADING phase's
replay driver (first run, catch-up, cap, no-new-data, daily risk reset).
"""

from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

import noctis.engine.trading_day as trading_day_mod
from noctis.data.types import empty_bars
from noctis.engine.sessions import (
    SessionLedger,
    session_date,
    sessions_present,
    slice_session,
    unseen_sessions,
)

from ._session_helpers import (
    ET,
    _bars_local,
    _concat,
    _FakeLake,
    _ledger_path,
    _make_runtime,
    _run_phase,
    _traded_dates,
    _uptrend,
)

# --- slicing units -------------------------------------------------------------------------


def test_session_date_converts_utc_ns_to_exchange_local_date():
    # 2026-03-05 01:00 UTC is 2026-03-04 20:00 in New York (EST, UTC-5): the bar belongs
    # to the 03-04 session even though its UTC date is 03-05.
    ns = pd.Timestamp("2026-03-05 01:00:00", tz="UTC").value
    assert session_date(ns, ET) == date(2026, 3, 4)
    # A mid-session bar keeps its own date.
    ns = pd.Timestamp("2026-03-04 14:00:00", tz="UTC").value  # 09:00 ET
    assert session_date(ns, ET) == date(2026, 3, 4)


def test_sessions_present_and_slice_across_three_dates():
    d1, d2, d3 = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)
    # d2 is a half-day (fewer bars); the tz guard: one d3 bar at 20:00 ET lands past
    # midnight UTC yet must stay in the d3 session.
    aapl = _concat(
        _bars_local(d1, _uptrend(10)),
        _bars_local(d2, _uptrend(4)),
        _bars_local(d3, _uptrend(9)),
        _bars_local(d3, [130.0], start=(20, 0)),  # 01:00 UTC on 03-05
    )
    msft = _concat(_bars_local(d1, _uptrend(10)), _bars_local(d2, _uptrend(4)))  # misses d3
    bars = {"AAPL": aapl, "MSFT": msft}

    assert sessions_present(bars, ET) == [d1, d2, d3]

    sliced = slice_session(bars, d3, ET)
    assert len(sliced["AAPL"]) == 10  # 9 morning bars + the evening tz-guard bar
    assert len(sliced["MSFT"]) == 0  # symbol missing the newest date → empty frame
    assert list(sliced["AAPL"].columns) == list(aapl.columns)
    # Every sliced bar really is on the d3 local date.
    assert all(session_date(int(ts), ET) == d3 for ts in sliced["AAPL"]["ts_event"])

    half = slice_session(bars, d2, ET)
    assert len(half["AAPL"]) == 4  # the half-day session is whatever bars the lake has


def test_slice_session_handles_empty_frames():
    sliced = slice_session({"AAPL": empty_bars()}, date(2026, 3, 2), ET)
    assert len(sliced["AAPL"]) == 0


def test_unseen_sessions_rules():
    d = [date(2026, 3, i) for i in range(2, 7)]
    # First run ever: only the newest, never all of history.
    assert unseen_sessions(d, None, cap=5) == ([d[-1]], 0)
    # Steady state: everything after the high-water mark, chronological.
    assert unseen_sessions(d, d[1], cap=5) == (d[2:], 0)
    # Cap truncates from the oldest side, keeping the newest.
    assert unseen_sessions(d, d[0], cap=2) == (d[3:], 2)
    # Nothing new / nothing at all.
    assert unseen_sessions(d, d[-1], cap=5) == ([], 0)
    assert unseen_sessions([], None, cap=5) == ([], 0)


# --- ledger --------------------------------------------------------------------------------


def test_ledger_round_trip_and_atomic_tmp_cleanup(tmp_path):
    path = tmp_path / "state" / "trading_sessions.json"
    ledger = SessionLedger(path)
    assert ledger.load() is None  # absent file → never traded
    ledger.save(date(2026, 7, 3))
    assert ledger.load() == date(2026, 7, 3)
    assert not path.with_suffix(".json.tmp").exists()  # tmp file replaced, not left behind
    # A fresh instance (restart) reads the same mark.
    assert SessionLedger(path).load() == date(2026, 7, 3)


@pytest.mark.parametrize("text", ["{not json", '{"version": 1}', '{"last_traded_session": 42}'])
def test_ledger_corrupt_file_is_a_hard_error(tmp_path, text):
    path = tmp_path / "trading_sessions.json"
    path.write_text(text)
    with pytest.raises(RuntimeError, match="corrupt trading-session ledger"):
        SessionLedger(path).load()


# --- runtime wiring ------------------------------------------------------------------------


def test_first_run_trades_only_newest_session(tmp_path):
    days = [date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)]
    lake = _FakeLake({"AAPL": _concat(*(_bars_local(d, _uptrend()) for d in days))})
    runtime = _make_runtime(tmp_path, lake)

    outcome = _run_phase(runtime)

    assert len(outcome.sessions) == 1  # a multi-month catalog still trades exactly one session
    assert _traded_dates(outcome.sessions[0]) == {days[-1]}
    assert SessionLedger(_ledger_path(runtime)).load() == days[-1]  # high-water mark seeded
    # Non-regression: slicing is trading-only — research still sees the full history.
    assert len(runtime.research_panel["AAPL"]) == 90


def test_bar_refresh_at_trading_entry_sees_new_lake_sessions(tmp_path):
    """Steady state: the T+1 sync lands one new session between cycles; the next TRADING
    entry drives the phase with its freshly loaded bars and only that slice trades (the
    runtime feeds ``run`` the reload's return value, so it can never hand stale bars)."""
    d_old, d_new = date(2026, 3, 4), date(2026, 3, 5)
    lake = _FakeLake({"AAPL": _bars_local(d_old, _uptrend())})
    runtime = _make_runtime(tmp_path, lake)

    _run_phase(runtime)  # cycle 1: seeds the mark at d_old
    # CLOSE-phase sync adds the next session to the lake (bars loaded at startup are stale).
    lake.bars["AAPL"] = _concat(lake.bars["AAPL"], _bars_local(d_new, _uptrend()))
    outcome = _run_phase(runtime, bars=dict(lake.bars))  # cycle 2: the refreshed view

    assert len(outcome.sessions) == 1
    record = outcome.sessions[0]
    assert _traded_dates(record) == {d_new}  # only the new bars, none re-traded
    assert sum(len(df) for df in record.bars.values()) == 30
    # No fill predates the new session (nothing leaked from already-traded history).
    day_start_ns = pd.Timestamp(datetime.combine(d_new, time(0, 0), tzinfo=ET)).value
    fills = outcome.broker.fills
    assert fills and all(f.ts_event >= day_start_ns for f in fills)
    assert SessionLedger(_ledger_path(runtime)).load() == d_new


def test_restart_catches_up_in_chronological_order(tmp_path):
    """Ledger says day N, the lake has N+1..N+3 → three sessions traded oldest-first,
    each in its own _TradingSession, all on the one continuous account."""
    days = [date(2026, 3, d) for d in (2, 3, 4, 5)]
    lake = _FakeLake({"AAPL": _concat(*(_bars_local(d, _uptrend()) for d in days))})
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(days[0])  # already traded through day N

    outcome = _run_phase(runtime)

    assert [_traded_dates(r) for r in outcome.sessions] == [{days[1]}, {days[2]}, {days[3]}]
    # The carried account spans the catch-up: each session picks up exactly where the
    # previous one settled.
    for prev, nxt in zip(outcome.sessions, outcome.sessions[1:], strict=False):
        assert nxt.summary.start_equity == pytest.approx(prev.summary.final_equity)
    assert SessionLedger(_ledger_path(runtime)).load() == days[-1]


def test_catchup_cap_trades_newest_and_reports_truncation(tmp_path):
    days = [date(2026, 3, 2 + i) for i in range(10)]
    lake = _FakeLake({"AAPL": _concat(*(_bars_local(d, _uptrend(10)) for d in days))})
    runtime = _make_runtime(tmp_path, lake)
    assert runtime.settings.trading.max_catchup_sessions == 5  # the shipped default
    SessionLedger(_ledger_path(runtime)).save(days[0] - pd.Timedelta(days=1))

    outcome = _run_phase(runtime)

    # Newest 5, in order.
    assert [_traded_dates(r) for r in outcome.sessions] == [{d} for d in days[-5:]]
    assert any(
        e.startswith(f"Skipped 5 stale sessions older than {days[5]}") for e in outcome.events
    )


def test_crash_mid_catchup_resumes_at_the_right_date(tmp_path, monkeypatch):
    """The ledger advances after each completed session, so a crash on session 2 of 3
    re-trades session 2 on restart rather than skipping it. (Fault injection: the crash
    has to land between two settles, which no interface input can produce.)"""
    days = [date(2026, 3, d) for d in (2, 3, 4, 5)]
    lake = _FakeLake({"AAPL": _concat(*(_bars_local(d, _uptrend(10)) for d in days))})
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(days[0])

    real = trading_day_mod.TradingDay.run
    sessions_run = [0]

    def crashy(self, **kwargs):
        if sessions_run[0] == 1:
            raise RuntimeError("boom mid-catch-up")
        sessions_run[0] += 1
        return real(self, **kwargs)

    monkeypatch.setattr(trading_day_mod.TradingDay, "run", crashy)
    with pytest.raises(RuntimeError, match="boom"):
        _run_phase(runtime)

    assert SessionLedger(_ledger_path(runtime)).load() == days[1]  # only session 1 committed


def test_no_new_data_day_skips_trading_with_an_explicit_event(tmp_path):
    day = date(2026, 3, 4)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend())})
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(day)  # lake newest == last traded
    before = _ledger_path(runtime).read_text()

    outcome = _run_phase(runtime)

    assert outcome.sessions == []  # no session ran, nothing submitted
    assert outcome.orders_submitted == 0 and outcome.trades == []
    assert (
        f"Trading skipped — no new session data (newest in lake {day}, last traded {day})"
        in outcome.events
    )
    assert _ledger_path(runtime).read_text() == before  # high-water mark untouched


def test_daily_loss_latch_resets_across_session_dates(tmp_path):
    """Regression for the latched halt: session 1 breaches max_daily_loss_pct and refuses
    every later entry; session 2 (next date) starts a fresh RiskManager and trades again
    with zero refusals. Under the old full-catalog replay the latch persisted for months."""
    d1, d2 = date(2026, 3, 4), date(2026, 3, 5)
    # Session 1: get long on an uptrend, crash >30% (≈>3% of equity at 10% sizing) so the
    # strategy flattens, then trend up again so it *tries* to re-enter — refused by the latch.
    session1 = _uptrend(12, start=100.0) + [60, 50, 45, 42, 40, 40] + _uptrend(20, start=40.0)
    session2 = _uptrend(30, start=100.0)
    lake = _FakeLake({"AAPL": _concat(_bars_local(d1, session1), _bars_local(d2, session2))})
    runtime = _make_runtime(tmp_path, lake)
    SessionLedger(_ledger_path(runtime)).save(d1 - pd.Timedelta(days=1))  # both unseen

    outcome = _run_phase(runtime)

    assert len(outcome.sessions) == 2
    s1, s2 = outcome.sessions[0].summary, outcome.sessions[1].summary
    assert s1.orders_refused > 0  # the latch engaged on day 1
    assert s2.orders_submitted > 0  # day 2 trades again…
    assert s2.orders_refused == 0  # …with a refusal count starting at 0
