"""Live-holdout plan 5: the per-champion forward track record.

Attribution folds each session's realized P&L into the champion that held the symbol that
session; the ledger is derived evidence, so a corrupt file is omitted from display and never
blocks trading; display adds current unrealized on top of ledger realized.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from noctis.broker import FeeModel, Order, PaperBroker, Side, SlippageModel
from noctis.engine import ForwardLedger, champion_key, forward_records
from noctis.engine.trading_day import TradingDay
from noctis.live import RiskLimits
from noctis.strategies import FamilyRegistry

from ._session_helpers import (
    _bars_local,
    _FakeLake,
    _make_runtime,
    _run_phase,
    _uptrend,
)


class _Entry:
    """A champion entry stand-in with just what attribution/display read."""

    def __init__(self, family, crowned_at, live_symbols):
        self.family = family
        self.params = {}
        self.crowned_at = crowned_at
        self.live_symbols = live_symbols
        self.test_metric = 1.0


class _Registry:
    def __init__(self, entries):
        self._entries = entries

    def list(self):
        return self._entries


# ── ledger record + persistence ─────────────────────────────────────────────────────────────
def test_ledger_folds_sessions_and_round_trips(tmp_path):
    path = tmp_path / "forward_ledger.json"
    fl = ForwardLedger(path)
    fl.record("A@t1", "A", date(2026, 7, 6), {"S": 30.0, "T": 5.0})
    fl.record("A@t1", "A", date(2026, 7, 7), {"S": 20.0})  # traded again
    fl.save()

    reloaded = ForwardLedger(path)
    reloaded.load()
    e = reloaded.entries["A@t1"]
    assert e.realized_pnl == pytest.approx(55.0)
    assert e.symbols["S"] == pytest.approx(50.0)
    assert e.symbols["T"] == pytest.approx(5.0)
    assert e.sessions_traded == 2
    assert e.opened_session == "2026-07-06"
    assert e.last_session == "2026-07-07"


def test_ledger_corrupt_load_is_graceful_never_raises(tmp_path):
    path = tmp_path / "forward_ledger.json"
    path.write_text("}{ not json")
    fl = ForwardLedger(path)
    fl.load()  # must not raise
    assert fl.corrupt is True
    assert fl.entries == {}


# ── display: realized + current unrealized ───────────────────────────────────────────────────
def test_forward_records_add_current_unrealized_and_sort_best_first(tmp_path):
    fl = ForwardLedger(tmp_path / "forward_ledger.json")
    fl.record("A@t1", "A", date(2026, 7, 6), {"S": 30.0})
    fl.record("B@t2", "B", date(2026, 7, 6), {"T": 10.0})
    entries = [_Entry("A", "t1", ["S"]), _Entry("B", "t2", ["T"])]

    broker = PaperBroker(fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0))
    broker.set_price("S", 100.0)
    broker.submit_order(Order("S", Side.BUY, 10))
    broker.set_price("S", 115.0)  # +150 unrealized, attributed to A (holds S)

    records = forward_records(fl, entries, broker)
    by_family = {r.family: r for r in records}
    assert by_family["A"].realized_pnl == pytest.approx(30.0)
    assert by_family["A"].unrealized_pnl == pytest.approx(150.0)
    assert by_family["A"].forward_pnl == pytest.approx(180.0)
    assert by_family["B"].unrealized_pnl == pytest.approx(0.0)  # holds nothing now
    assert [r.family for r in records] == ["A", "B"]  # 180 before 10


def test_forward_records_without_broker_show_realized_only(tmp_path):
    fl = ForwardLedger(tmp_path / "forward_ledger.json")
    fl.record("A@t1", "A", date(2026, 7, 6), {"S": 30.0})
    records = forward_records(fl, [_Entry("A", "t1", ["S"])], broker=None)
    assert records[0].realized_pnl == pytest.approx(30.0)
    assert records[0].unrealized_pnl == pytest.approx(0.0)


# ── session attribution (TradingDay) ─────────────────────────────────────────────────────────
def test_attribute_forward_credits_each_symbol_to_its_champion(tmp_path):
    # Two champions pinned to distinct symbols: S's realized goes to A, T's to B; a second
    # session with only S moving still increments B's sessions_traded (it was assigned T).
    entries = [_Entry("A", "t1", ["S"]), _Entry("B", "t2", ["T"])]
    fl = ForwardLedger(tmp_path / "forward_ledger.json")
    broker = PaperBroker()
    day = TradingDay(
        broker=broker,
        store=None,  # _attribute never touches persistence
        ledger=None,
        forward=fl,
        registry=_Registry(entries),
        families=FamilyRegistry(),
        limits=RiskLimits(),
    )

    broker.realized_pnl_by_symbol = {"S": 30.0, "T": 12.0}
    day._attribute(["S", "T"], {}, date(2026, 7, 6))
    assert fl.entries[champion_key(entries[0])].realized_pnl == pytest.approx(30.0)
    assert fl.entries[champion_key(entries[1])].realized_pnl == pytest.approx(12.0)

    broker.realized_pnl_by_symbol = {"S": 50.0, "T": 12.0}  # only S moved this session
    day._attribute(["S", "T"], {"S": 30.0, "T": 12.0}, date(2026, 7, 7))
    a = fl.entries[champion_key(entries[0])]
    b = fl.entries[champion_key(entries[1])]
    assert a.realized_pnl == pytest.approx(50.0) and a.sessions_traded == 2
    assert b.realized_pnl == pytest.approx(12.0) and b.sessions_traded == 2  # traded, +0


def test_replay_writes_the_forward_ledger(tmp_path):
    runtime = _make_runtime(
        tmp_path, _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    )
    _run_phase(runtime)
    fl = ForwardLedger(Path(runtime.settings.state_dir) / "forward_ledger.json")
    fl.load()
    assert not fl.corrupt
    assert "sma_crossover@" in fl.entries  # the fake champion has no crowned_at → "@" suffix
    assert fl.entries["sma_crossover@"].sessions_traded == 1
    assert "AAPL" in fl.entries["sma_crossover@"].symbols


def test_corrupt_forward_ledger_does_not_block_trading(tmp_path):
    runtime = _make_runtime(
        tmp_path, _FakeLake({"AAPL": _bars_local(date(2026, 3, 9), _uptrend())})
    )
    fl_path = Path(runtime.settings.state_dir) / "forward_ledger.json"
    fl_path.parent.mkdir(parents=True, exist_ok=True)
    fl_path.write_text("{ corrupt")
    _run_phase(runtime)  # must not raise
    # Trading still happened: the account advanced despite the bad ledger.
    assert (Path(runtime.settings.state_dir) / "paper_account.json").is_file()


def test_exit_fill_credits_the_opener_and_carries_its_reason(tmp_path):
    """A stop-out closes the position through the normal path: the realized loss lands on
    the champion that opened it, and the fill's reason rides the session outcome the report
    reads — the operator can see WHY the position closed."""
    from dataclasses import dataclass

    from noctis.broker.persistence import AccountStore
    from noctis.engine.sessions import SessionLedger
    from noctis.live.feed import ReplayBarFeed
    from noctis.strategies.base import ExitRules, TraderStrategy

    @dataclass(frozen=True)
    class _NoParams:
        pass

    class _StopProbe(TraderStrategy):
        name = "stop_probe"
        params_cls = _NoParams
        timeframe = "1m"

        def on_start(self, ctx):
            pass

        def on_bar(self, ctx, bar):
            ctx.set_target(1, exits=ExitRules(stop_pct=0.10))

        @classmethod
        def param_space(cls):
            return []

    families = FamilyRegistry()
    families.register(_StopProbe)
    entries = [_Entry("stop_probe", "t1", ["S"])]
    fl = ForwardLedger(tmp_path / "forward_ledger.json")
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )
    day = TradingDay(
        broker=broker,
        store=AccountStore(tmp_path / "paper_account.json"),
        ledger=SessionLedger(tmp_path / "trading_sessions.json"),
        forward=fl,
        registry=_Registry(entries),
        families=families,
        limits=RiskLimits(
            max_position_pct=95.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=100.0
        ),
    )

    import pandas as pd

    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 101.0, 100.0, 101.0),  # entry: 950 units at 100
        (100.0, 100.0, 88.0, 92.0),  # stop fills at 90 → realized −9500
        (91.0, 92.0, 90.0, 91.0),
    ]
    tape = pd.DataFrame(
        {
            "ts_event": [i * 60 * 1_000_000_000 for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )

    outcome = day.run(ReplayBarFeed({"S": tape}), date(2026, 7, 6))

    stop_fills = [f for f in outcome.fills if f.reason == "stop"]
    assert len(stop_fills) == 1 and stop_fills[0].price == 90.0
    key = champion_key(entries[0])
    assert fl.entries[key].realized_pnl == pytest.approx(-9500.0)
    assert fl.entries[key].symbols == {"S": pytest.approx(-9500.0)}
    assert any("protective-exit fill(s): stop ×1" in e for e in outcome.summary.events)
