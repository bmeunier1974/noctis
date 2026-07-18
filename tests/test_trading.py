"""TRADING loop: risk limits and champions emitting paper orders on replay data."""

from __future__ import annotations

import pandas as pd

from noctis.broker import FeeModel, PaperBroker, SlippageModel, simulate
from noctis.live import RiskLimits, RiskManager, run_trading
from noctis.strategies import Candidate
from noctis.strategies.base import ExitRules

from ._data_helpers import make_ohlcv

# --- risk manager units ------------------------------------------------------------------


def test_risk_halts_below_daily_loss_floor():
    rm = RiskManager(RiskLimits(max_daily_loss_pct=3.0), start_equity=100_000.0)
    assert rm.is_halted(98_000.0) is False
    assert rm.is_halted(97_000.0) is True  # exactly at the 3% floor
    assert rm.is_halted(96_000.0) is True


def test_risk_halt_latches_for_the_session():
    rm = RiskManager(RiskLimits(max_daily_loss_pct=3.0), start_equity=100_000.0)
    assert rm.is_halted(96_000.0) is True
    # Equity recovering above the floor does not un-halt the session.
    assert rm.is_halted(99_000.0) is True
    d = rm.target(
        "AAPL", 1, price=100.0, equity=99_000.0, positions={"AAPL": 0.0}, marks={"AAPL": 100.0}
    )
    assert d.refused is True and "daily loss" in d.reason


def test_risk_target_respects_position_cap():
    rm = RiskManager(RiskLimits(max_position_pct=10.0, max_gross_exposure_pct=100.0), 100_000.0)
    d = rm.target("AAPL", desired_sign=1, price=200.0, equity=100_000.0, positions={}, marks={})
    # 10% of 100k = 10k notional / 200 = 50 shares.
    assert d.refused is False
    assert d.target_qty == 50.0


def test_risk_target_respects_gross_exposure():
    rm = RiskManager(RiskLimits(max_position_pct=100.0, max_gross_exposure_pct=50.0), 100_000.0)
    # Another position already uses 40k of gross exposure.
    positions = {"MSFT": 100.0}
    marks = {"MSFT": 400.0}  # 40k
    d = rm.target("AAPL", 1, price=100.0, equity=100_000.0, positions=positions, marks=marks)
    # Gross cap 50k − 40k used = 10k room → 100 shares at $100.
    assert d.target_qty == 100.0


def test_risk_refuses_new_exposure_when_halted():
    rm = RiskManager(RiskLimits(max_daily_loss_pct=3.0), start_equity=100_000.0)
    d = rm.target(
        "AAPL", 1, price=100.0, equity=95_000.0, positions={"AAPL": 0.0}, marks={"AAPL": 100.0}
    )
    assert d.refused is True
    assert "daily loss" in d.reason


# --- champion→symbol assignment ----------------------------------------------------------


def test_assign_legacy_round_robin_preserved():
    """No symbol sets, no scores → the original round-robin mapping, exactly."""
    from noctis.live.node import _assign

    cands = [
        Candidate("sma_crossover", {"fast": 3, "slow": 8}),
        Candidate("donchian_breakout", {"channel": 15}),
    ]
    symbols = ["AAPL", "MSFT", "NVDA", "SPY"]
    got = _assign(cands, symbols)
    assert got == {sym: cands[i % len(cands)] for i, sym in enumerate(symbols)}


def test_assign_best_scoring_eligible_champion_wins():
    """Each symbol goes to the best-scoring champion whose live set contains it; a legacy
    None entry is eligible everywhere and picks up the symbols nobody else claims."""
    from noctis.live.node import _assign

    legacy = Candidate("sma_crossover", {"fast": 3, "slow": 8})
    narrow = Candidate("donchian_breakout", {"channel": 15})
    strong = Candidate("rsi_meanrev", {"period": 5, "low": 30, "high": 70})
    got = _assign(
        [legacy, narrow, strong],
        ["AAPL", "MSFT", "JPM"],
        live_symbols=[None, {"AAPL", "MSFT"}, {"MSFT"}],
        scores=[0.1, 0.5, 0.9],
    )
    assert got["AAPL"] is narrow  # 0.5 beats the legacy 0.1; strong is not eligible
    assert got["MSFT"] is strong  # highest score among all three eligible
    assert got["JPM"] is legacy  # only the legacy champion may trade it


def test_assign_leaves_symbol_without_eligible_champion_unassigned():
    from noctis.live.node import _assign

    only = Candidate("sma_crossover", {"fast": 3, "slow": 8})
    got = _assign([only], ["AAPL", "MSFT"], live_symbols=[{"AAPL"}], scores=[1.0])
    assert got == {"AAPL": only}  # MSFT has no eligible champion → nothing trades it


# --- trading loop on replay --------------------------------------------------------------


def _uptrend(n=120):
    return make_ohlcv([100.0 + i * 0.5 for i in range(n)])


def test_champions_emit_paper_orders_on_replay():
    bars = {"AAPL": _uptrend(), "MSFT": _uptrend()}
    last_close = 100.0 + (120 - 1) * 0.5  # final mark price of the uptrend
    candidates = [Candidate("sma_crossover", {"fast": 3, "slow": 8})]
    summary = run_trading(
        candidates=candidates,
        bars_by_symbol=bars,
        limits=RiskLimits(max_position_pct=10.0, max_gross_exposure_pct=100.0),
    )
    assert summary.orders_submitted > 0
    assert summary.fills > 0
    # No single position exceeds the 10% notional cap (allow a little for slippage/rounding).
    for sym, qty in summary.positions.items():
        assert abs(qty) * last_close <= summary.final_equity * 0.10 + 50.0, sym


def test_gross_cap_holds_when_multiple_symbols_enter_same_bar():
    # Two symbols going long the same bar must share the gross cap, not each get the
    # full room from a stale pre-minute snapshot (60% + 60% > 100% cap).
    bars = {"AAPL": _uptrend(), "MSFT": _uptrend()}
    last_close = 100.0 + (120 - 1) * 0.5
    candidates = [Candidate("sma_crossover", {"fast": 3, "slow": 8})]
    summary = run_trading(
        candidates=candidates,
        bars_by_symbol=bars,
        limits=RiskLimits(max_position_pct=60.0, max_gross_exposure_pct=100.0),
    )
    gross = sum(abs(qty) * last_close for qty in summary.positions.values())
    assert gross <= summary.final_equity * 1.0 + 100.0  # small slippage/rounding allowance


class _ProbeCandidate:
    def __init__(self, probe):
        self._probe = probe

    def build(self, families):
        return self._probe


class _TimeframeProbe:
    """A 5m strategy that counts its on_bar calls and records the bars it sees."""

    timeframe = "5m"

    def __init__(self):
        self.bars_seen: list = []

    def on_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx, bar) -> None:
        self.bars_seen.append(bar)
        ctx.set_target(1)


def test_timeframe_proxy_feeds_aggregated_bars_in_trading_loop():
    # 12 minute bars → 5m buckets [0-4][5-9][10-11]; the strategy must decide only when
    # a bucket completes (bars 5 and 10 arriving), never on raw minutes or the partial tail.
    probe = _TimeframeProbe()
    minutes = _uptrend(12)
    summary = run_trading(
        candidates=[_ProbeCandidate(probe)],
        bars_by_symbol={"AAPL": minutes},
    )
    assert summary.bars_processed == 12  # execution/marks still run per minute
    assert len(probe.bars_seen) == 2
    first = probe.bars_seen[0]
    assert first.open == float(minutes["open"].iloc[0])
    assert first.close == float(minutes["close"].iloc[4])
    assert first.high == float(minutes["high"].iloc[:5].max())
    assert first.volume == float(minutes["volume"].iloc[:5].sum())
    assert summary.fills > 0  # the long decision still executes on later minutes


def test_degraded_feed_halts_order_emission():
    bars = {"AAPL": _uptrend()}
    candidates = [Candidate("sma_crossover", {"fast": 3, "slow": 8})]
    summary = run_trading(
        candidates=candidates,
        bars_by_symbol=bars,
        is_degraded=lambda: True,  # feed is delayed → never act
    )
    assert summary.orders_submitted == 0
    assert summary.fills == 0
    assert summary.halted_for_degraded > 0


def test_zero_position_cap_refuses_all_orders():
    bars = {"AAPL": _uptrend()}
    candidates = [Candidate("sma_crossover", {"fast": 3, "slow": 8})]
    summary = run_trading(
        candidates=candidates,
        bars_by_symbol=bars,
        limits=RiskLimits(max_position_pct=0.0, max_gross_exposure_pct=100.0),
    )
    assert summary.orders_submitted == 0
    assert summary.orders_refused > 0


# --- protective exits in live: same engine as the simulator, same fills ---------------------


class _ScriptedExitStub:
    """A 1m scripted-target strategy that re-declares the same exit rules every bar."""

    timeframe = "1m"

    def __init__(self, script, exits):
        self._script = list(script)
        self._exits = exits
        self._i = 0

    def on_start(self, ctx):
        self._i = 0

    def on_bar(self, ctx, bar):
        idx = min(self._i, len(self._script) - 1)
        ctx.set_target(self._script[idx], exits=self._exits)
        self._i += 1


def _exit_tape():
    """Entry at 100 → 10% stop breached intrabar → target cycles 0 → re-entry at 90.5."""
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 101.0, 100.0, 101.0),  # +1 fills at the open: 950 units at 100
        (100.0, 100.0, 88.0, 92.0),  # low breaches 90 → stop fill at 90, latch on
        (91.0, 92.0, 90.0, 91.0),  # raw target flips to 0 → un-latch, still flat
        (90.0, 91.0, 89.0, 90.0),  # raw target back to +1 → decision to re-enter
        (90.5, 91.0, 90.0, 91.0),  # re-entry fills at the open
    ]
    return pd.DataFrame(
        {
            "ts_event": [i * 60 * 1_000_000_000 for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1000.0] * len(rows),
        }
    )


def test_live_and_simulate_produce_the_same_fill_sequence_with_exits():
    """The Phase-4 parity assertion: the live driver and the simulator run the SAME exit
    engine on the same tape and cannot disagree on a single fill."""
    script, rules = [1, 1, 1, 0, 1, 1], ExitRules(stop_pct=0.10)
    tape = _exit_tape()

    sim_result = simulate(
        _ScriptedExitStub(script, rules),
        tape,
        PaperBroker(
            starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
        ),
        symbol="AAPL",
        alloc=0.95,
    )

    live_broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )
    summary = run_trading(
        candidates=[_ProbeCandidate(_ScriptedExitStub(script, rules))],
        bars_by_symbol={"AAPL": tape},
        broker=live_broker,
        # 95% position cap = the simulator's alloc; loss floor at 100% never halts the drive.
        limits=RiskLimits(
            max_position_pct=95.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=100.0
        ),
    )

    def fill_seq(fills):
        return [(f.side.value, f.quantity, f.price, f.reason) for f in fills]

    assert fill_seq(live_broker.fills) == fill_seq(sim_result.fills)
    assert [f.reason for f in live_broker.fills] == ["target", "stop", "target"]
    assert summary.exit_fills == {"stop": 1}


def test_halted_session_skips_exit_evaluation():
    """Precedence: the daily-loss halt owns a halted session — armed exit rules do not
    evaluate while halted (the existing risk path, not the exit engine, is in charge)."""
    script, rules = [1, 1, 1, 1, 1], ExitRules(stop_pct=0.10)
    rows = [
        (100.0, 101.0, 99.0, 100.0),  # decide +1
        (100.0, 101.0, 99.5, 100.0),  # entry: 950 units at 100; stop level 90
        (98.0, 99.0, 97.0, 97.0),  # drifting down, still above the 3% halt floor
        (96.0, 97.0, 89.0, 92.0),  # halt latches at the open; low 89 would breach the stop
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
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )

    summary = run_trading(
        candidates=[_ProbeCandidate(_ScriptedExitStub(script, rules))],
        bars_by_symbol={"AAPL": tape},
        broker=broker,
        limits=RiskLimits(
            max_position_pct=95.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=3.0
        ),
    )

    assert summary.halt_latched is True
    assert all(f.reason == "target" for f in broker.fills)  # the stop never fired
    assert summary.exit_fills == {}
    assert broker.position("AAPL").quantity > 900.0  # still held (modulo per-bar re-trues)


def test_exits_evaluate_only_on_completed_declared_timeframe_bars():
    """A 5m strategy's stop is breached by a minute-7 low, but the exit fires only when
    the 5m bucket completes (minute 10 arriving) — never on a raw sub-timeframe minute."""

    class _FiveMinuteStop:
        timeframe = "5m"

        def on_start(self, ctx):
            pass

        def on_bar(self, ctx, bar):
            ctx.set_target(1, exits=ExitRules(stop_pct=0.10))

    rows = [(100.0, 100.5, 99.5, 100.0)] * 12
    rows[7] = (100.0, 100.5, 88.0, 100.0)  # sub-timeframe breach of the eventual 90 level
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
    broker = PaperBroker(
        starting_cash=100_000.0, fee_model=FeeModel(0.0), slippage_model=SlippageModel(0.0)
    )

    run_trading(
        candidates=[_ProbeCandidate(_FiveMinuteStop())],
        bars_by_symbol={"AAPL": tape},
        broker=broker,
        limits=RiskLimits(
            max_position_pct=95.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=100.0
        ),
    )

    assert [f.reason for f in broker.fills] == ["target", "stop"]
    stop_fill = broker.fills[-1]
    assert stop_fill.price == 90.0  # evaluated against the completed 5m bar's low
    # Fired while processing minute 10 (the bucket-completing minute), not minute 7.
    assert stop_fill.ts_event == 10 * 60 * 1_000_000_000
