"""Champion-orphan position flattening (the last open trading-replay item).

A carried position whose symbol NO current champion is eligible to trade is unmanaged — its
opener was displaced from the board (or its symbol set changed) and nothing will ever decide
it again. Each session detects those orphans at entry and flattens them at their first
tradable bar through the NORMAL risk/broker path, so fills, report trades, and the forward
ledger stay honest. The two design decisions under test:

* a symbol REASSIGNED to a *different* champion is **inherited**, never flattened — the new
  assignee re-decides from the carried position (the shipped pending-sign seeding), and both
  realized and unrealized attribution already follow the inheritor;
* the closing fill of an orphan flatten is credited to the champion that OPENED the position
  via the forward ledger's persisted holder map — not to nobody, not to any current champion.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from noctis.broker.paper import PaperBroker
from noctis.broker.persistence import AccountStore
from noctis.broker.seam import Order, Side
from noctis.config import load_settings
from noctis.data.types import NS_PER_SECOND
from noctis.engine import ForwardLedger, SimulatedSleeper, build_runtime
from noctis.engine.trading_day import TradingDay
from noctis.live import RiskLimits, SessionConfig, run_trading, run_trading_day
from noctis.memory import MemoryStore
from noctis.strategies import FamilyRegistry
from noctis.strategies.base import Bar

from ._data_helpers import make_ohlcv
from ._session_helpers import (
    _account_path,
    _bars_local,
    _FakeLake,
    _run_phase,
    _uptrend,
)

# --- local fixtures ------------------------------------------------------------------------


class _Entry:
    """A champion entry with a controllable symbol set / identity."""

    def __init__(self, family="sma_crossover", crowned_at="t1", live_symbols=None):
        self.family = family
        self.params = {"fast": 3, "slow": 8}
        self.crowned_at = crowned_at
        self.live_symbols = live_symbols
        self.test_metric = 1.0


class _Registry:
    def __init__(self, entries):
        self._entries = entries

    def list(self):
        return self._entries


def _runtime_with(tmp_path, lake, entries, universe=("AAPL",)):
    """`_make_runtime`, but with a caller-chosen champion board."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "mode: paper\n"
        f"universe: [{', '.join(universe)}]\n"
        f"state_dir: {tmp_path}/state/\n"
        f"strategies_dir: {tmp_path}/strategies/\n"
    )
    settings = load_settings(config_path=cfg)
    return build_runtime(
        settings,
        market_lake=lake,
        memory=MemoryStore(tmp_path / "MEMORY.md"),
        registry=_Registry(entries),
        reports_dir=str(tmp_path / "reports"),
    )


def _seed_position(runtime, symbol, qty, price, day):
    """A prior session left an open position on the persisted continuous account."""
    store = AccountStore(_account_path(runtime))
    broker = store.load()
    broker.set_price(symbol, price)
    broker.submit_order(Order(symbol, Side.BUY, qty))
    store.save(broker, day)


def _forward_path(runtime) -> Path:
    return Path(runtime.settings.state_dir) / "forward_ledger.json"


def _seed_holder(runtime, symbol, key, family):
    fl = ForwardLedger(_forward_path(runtime))
    fl.load()
    fl.holders[symbol] = {"key": key, "family": family}
    fl.save()


class _AlwaysLong:
    def on_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx, bar) -> None:
        ctx.set_target(1)


class _Candidate:
    def __init__(self, strat):
        self._strat = strat

    def build(self, families):
        return self._strat


class _ScriptedFeed:
    """A minimal fake live feed: one pre-built minute group per poll (clock-bounded)."""

    def __init__(self, symbols, groups):
        self.symbols = list(symbols)
        self._groups = list(groups)
        self._i = 0
        self.degraded = False
        self.exhausted = False

    def poll_once(self):
        group = self._groups[self._i] if self._i < len(self._groups) else {}
        self._i += 1
        return group

    def flush(self):
        return {}


def _bar(row) -> Bar:
    return Bar(
        ts_event=int(row["ts_event"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def _groups(frames: dict[str, pd.DataFrame]):
    """Cross-symbol minute groups from per-symbol frames (aligned on row index)."""
    n = max(len(df) for df in frames.values())
    out = []
    for i in range(n):
        group = {s: _bar(df.iloc[i]) for s, df in frames.items() if i < len(df)}
        out.append(group)
    return out


# --- 1) replay driver: detect, flatten, report, attribute ----------------------------------


def test_replay_flattens_orphan_and_attributes_to_recorded_holder(tmp_path):
    """The full path on the replay driver: a carried AAPL position with no eligible champion
    is flattened at its first bar, the event names the displaced opener, the report trade is
    labeled honestly, and the closing realized P&L lands on the opener's ledger entry
    without counting a traded session for it."""
    day = date(2026, 3, 9)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend()), "MSFT": _bars_local(day, _uptrend())})
    # The one champion is pinned to MSFT — nothing on the board is eligible for AAPL.
    runtime = _runtime_with(
        tmp_path, lake, [_Entry(live_symbols=["MSFT"])], universe=("AAPL", "MSFT")
    )
    _seed_position(runtime, "AAPL", 10.0, 100.0, day - timedelta(days=1))
    _seed_holder(runtime, "AAPL", "old_champ@2026-01-01", "old_champ")

    outcome = _run_phase(runtime)

    summary = outcome.sessions[0].summary
    assert summary.orphans_flattened == ["AAPL"]
    flatten_lines = [e for e in summary.events if "Orphaned position flattened: AAPL" in e]
    assert flatten_lines and "opened by old_champ@2026-01-01" in flatten_lines[0]
    assert flatten_lines[0] in outcome.events  # reaches the CLOSE report verbatim

    # The position is genuinely closed on the persisted account.
    assert AccountStore(_account_path(runtime)).load().position("AAPL").quantity == 0.0
    # The closing fill is labeled honestly in the report's trades (not "champion signal").
    assert ("AAPL", "orphan flatten") in {(t.symbol, t.rationale) for t in outcome.trades}

    # Ledger: the realized delta is the opener's, no session is claimed, the holder drops,
    # and the still-open MSFT position now records the current champion as holder.
    fl = ForwardLedger(_forward_path(runtime))
    fl.load()
    broker = outcome.broker
    entry = fl.entries["old_champ@2026-01-01"]
    assert entry.realized_pnl == pytest.approx(broker.realized_pnl_by_symbol["AAPL"])
    assert entry.symbols == {"AAPL": pytest.approx(broker.realized_pnl_by_symbol["AAPL"])}
    assert entry.sessions_traded == 0
    assert "AAPL" not in fl.holders
    assert fl.holders["MSFT"]["key"] == "sma_crossover@t1"


def test_live_driver_flattens_orphan_identically():
    """The clock-bounded (live) driver funnels through the same session core: the orphan is
    flattened at its first polled bar with the same event."""
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    sleeper = SimulatedSleeper(start)
    broker = PaperBroker()
    broker.set_price("ORPH", 100.0)
    broker.submit_order(Order("ORPH", Side.BUY, 10))
    frames = {
        "AAPL": make_ohlcv([100.0 + i * 0.5 for i in range(8)]),
        "ORPH": make_ohlcv([100.0] * 8),
    }
    config = SessionConfig(
        candidates=[_Candidate(_AlwaysLong())],
        live_symbols=[{"AAPL"}],
        scores=[1.0],
        broker=broker,
        position_holders={"ORPH": "old_champ@t0"},
    )
    result = run_trading_day(
        config,
        _ScriptedFeed(["AAPL", "ORPH"], _groups(frames)),
        session_start=start,
        session_end=start + timedelta(seconds=12),
        now=sleeper.now,
        sleeper=sleeper,
        poll_interval_s=1.0,
    )
    assert result.summary.orphans_flattened == ["ORPH"]
    assert broker.position("ORPH").quantity == 0.0
    assert any("opened by old_champ@t0" in e for e in result.summary.events)


# --- 2) reassigned symbol: inherited, never flattened ---------------------------------------


def test_reassigned_symbol_is_inherited_not_flattened(tmp_path):
    """A carried position whose symbol got a NEW champion is not an orphan: the inheritor
    starts from the carried position (no forced flatten at the open) and becomes the
    recorded holder at settle."""
    day = date(2026, 3, 9)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend(30))})
    runtime = _runtime_with(tmp_path, lake, [_Entry(crowned_at="new", live_symbols=["AAPL"])])
    _seed_position(runtime, "AAPL", 5.0, 100.0, day - timedelta(days=1))
    _seed_holder(runtime, "AAPL", "old_champ@t0", "old_champ")

    outcome = _run_phase(runtime)

    summary = outcome.sessions[0].summary
    assert summary.orphans_flattened == []
    assert not any("Orphaned" in e for e in summary.events)
    # The inheritor's first action is its own re-true UP from the carried 5 shares (a buy),
    # not a forced dump at the open; later resizes are the strategy's own choices.
    broker = outcome.broker
    assert broker.fills[0].side is Side.BUY
    assert all(t.rationale == "champion signal" for t in outcome.trades)
    assert summary.positions["AAPL"] > 0  # still long at the close (uptrend hold)
    # Settle re-derives the holder: the inheritor owns the open position now.
    fl = ForwardLedger(_forward_path(runtime))
    fl.load()
    assert fl.holders["AAPL"]["key"] == "sma_crossover@new"


def test_no_orphans_under_a_legacy_whole_universe_champion(tmp_path):
    """A legacy champion (live_symbols None) is eligible everywhere — a carried position is
    never an orphan, and the session is byte-identical to the pre-flatten loop."""
    day = date(2026, 3, 9)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend(30))})
    runtime = _runtime_with(tmp_path, lake, [_Entry(live_symbols=None)])
    _seed_position(runtime, "AAPL", 5.0, 100.0, day - timedelta(days=1))

    outcome = _run_phase(runtime)

    summary = outcome.sessions[0].summary
    assert summary.orphans_flattened == []
    assert not any("Orphaned" in e for e in summary.events)
    assert all(t.rationale == "champion signal" for t in outcome.trades)


# --- 3) settle attribution + holder upkeep (TradingDay units) -------------------------------


def _trading_day(tmp_path, entries, broker):
    return TradingDay(
        broker=broker,
        store=None,  # units below never touch persistence
        ledger=None,
        forward=ForwardLedger(tmp_path / "forward_ledger.json"),
        registry=_Registry(entries),
        families=FamilyRegistry(),
        limits=RiskLimits(),
    )


def test_attribute_credits_orphan_delta_to_recorded_holder(tmp_path):
    broker = PaperBroker()
    day = _trading_day(tmp_path, [_Entry(family="B", crowned_at="t2", live_symbols=["T"])], broker)
    day.forward.holders["S"] = {"key": "A@t1", "family": "A"}

    broker.realized_pnl_by_symbol = {"S": 25.0, "T": 5.0}
    day._attribute(["S", "T"], {}, date(2026, 7, 6))

    a, b = day.forward.entries["A@t1"], day.forward.entries["B@t2"]
    assert a.realized_pnl == pytest.approx(25.0) and a.symbols == {"S": 25.0}
    assert a.sessions_traded == 0  # its position closed; it made no decision that day
    assert a.family == "A"
    assert b.realized_pnl == pytest.approx(5.0) and b.sessions_traded == 1


def test_attribute_orphan_without_recorded_holder_warns_and_skips(tmp_path, caplog):
    broker = PaperBroker()
    day = _trading_day(tmp_path, [_Entry(family="B", crowned_at="t2", live_symbols=["T"])], broker)
    broker.realized_pnl_by_symbol = {"S": 25.0}

    with caplog.at_level(logging.WARNING, logger="noctis.runtime"):
        day._attribute(["S"], {}, date(2026, 7, 6))

    assert "no recorded holder" in caplog.text
    assert day.forward.entries == {}  # unattributed — never guessed onto a current champion


def test_update_holders_inherits_drops_and_keeps(tmp_path):
    """Assigned open position → inheritor recorded; closed symbol → dropped; orphaned but
    still-open symbol → recorded holder kept for the eventual flatten."""
    broker = PaperBroker()
    for sym in ("S", "Q"):
        broker.set_price(sym, 100.0)
        broker.submit_order(Order(sym, Side.BUY, 10))
    day = _trading_day(tmp_path, [_Entry(family="B", crowned_at="t2", live_symbols=["S"])], broker)
    day.forward.holders = {
        "S": {"key": "A@t1", "family": "A"},  # open, reassigned to B
        "Q": {"key": "A@t1", "family": "A"},  # open, orphaned (B not eligible)
        "X": {"key": "A@t1", "family": "A"},  # no longer held
    }

    day._update_holders()

    assert day.forward.holders["S"] == {"key": "B@t2", "family": "B"}
    assert day.forward.holders["Q"] == {"key": "A@t1", "family": "A"}
    assert "X" not in day.forward.holders


# --- 4) risk interplay: halt latch and degraded feed ----------------------------------------


def test_flatten_executes_while_daily_loss_latch_is_tripped():
    """The AAPL crash trips the daily-loss latch BEFORE the orphan's first bar arrives; the
    flatten still fills. That is by design, not a bypass: a flatten is risk-reducing, and the
    risk policy (live/risk.py) refuses only exposure-INCREASING orders while latched —
    "flattening is allowed". The orphan flatten routes through the same ``risk.target`` seam,
    so it inherits exactly that policy."""
    broker = PaperBroker()
    broker.set_price("ORPH", 100.0)
    broker.submit_order(Order("ORPH", Side.BUY, 50))
    aapl = make_ohlcv([100.0, 100.0, 60.0, 60.0, 60.0, 60.0])  # crash at bar 2 → latch
    orph = make_ohlcv([100.0, 100.0, 100.0])
    orph["ts_event"] = orph["ts_event"] + 3 * 60 * NS_PER_SECOND  # first bar after the crash

    summary = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol={"AAPL": aapl, "ORPH": orph},
        broker=broker,
        live_symbols=[{"AAPL"}],
        scores=[1.0],
        limits=RiskLimits(
            max_position_pct=50.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=3.0
        ),
    )

    assert summary.halt_latched is True  # the latch really tripped…
    assert summary.orphans_flattened == ["ORPH"]  # …and the flatten still executed
    assert broker.position("ORPH").quantity == 0.0


def test_degraded_feed_defers_flatten_and_reports_it():
    """Degraded quotes halt ALL order emission, flattens included: the orphan stays open,
    the summary says so, and the next session retries (detection re-runs at entry)."""
    broker = PaperBroker()
    broker.set_price("ORPH", 100.0)
    broker.submit_order(Order("ORPH", Side.BUY, 10))
    bars = {
        "AAPL": make_ohlcv([100.0 + i for i in range(5)]),
        "ORPH": make_ohlcv([100.0] * 5),
    }
    summary = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        broker=broker,
        live_symbols=[{"AAPL"}],
        scores=[1.0],
        is_degraded=lambda: True,
    )
    assert summary.orphans_flattened == []
    assert broker.position("ORPH").quantity == 10.0
    assert any("Orphaned position(s) still open" in e and "ORPH" in e for e in summary.events)


# --- 5) holder persistence -------------------------------------------------------------------


def test_holders_round_trip_and_pre_holder_files_load_clean(tmp_path):
    path = tmp_path / "forward_ledger.json"
    fl = ForwardLedger(path)
    fl.record("A@t1", "A", date(2026, 7, 6), {"S": 30.0})
    fl.holders["S"] = {"key": "A@t1", "family": "A"}
    fl.save()

    reloaded = ForwardLedger(path)
    reloaded.load()
    assert reloaded.holders == {"S": {"key": "A@t1", "family": "A"}}

    # A pre-holder file (no "holders" key) loads with an empty map, never an error.
    path.write_text('{"version": 1, "champions": {}}')
    old = ForwardLedger(path)
    old.load()
    assert old.corrupt is False and old.holders == {}


def test_champion_opening_a_position_records_itself_as_holder(tmp_path):
    """The map is self-maintaining: a normal session that ends with an open position records
    the assigned champion as its holder — the identity a later flatten will credit."""
    day = date(2026, 3, 9)
    lake = _FakeLake({"AAPL": _bars_local(day, _uptrend(30))})
    runtime = _runtime_with(tmp_path, lake, [_Entry(crowned_at="c1")])

    outcome = _run_phase(runtime)

    assert outcome.sessions[0].summary.positions["AAPL"] > 0  # the champion ended long
    fl = ForwardLedger(_forward_path(runtime))
    fl.load()
    assert fl.holders["AAPL"] == {"key": "sma_crossover@c1", "family": "sma_crossover"}
