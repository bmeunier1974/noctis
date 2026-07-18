"""Live-holdout plan 3: the replay loop's rebalance dead-band and collapsed refusal log.

Two quality-of-life quiets that touch only the live/replay session (never the backtest fills
path, so no scorecard or gate moves):

* the **dead-band** stops a held, same-direction position from re-truing into a sub-share fill
  nearly every bar, while opens/exits/flips always execute;
* the **refusal collapse** turns a per-bar INFO flood after the daily-loss latch into one WARNING,
  keeping ``summary.orders_refused`` the honest full count.
"""

from __future__ import annotations

import logging

from noctis.live import RiskLimits, run_trading

from ._data_helpers import make_ohlcv


def _uptrend(n: int = 20):
    return make_ohlcv([100.0 + i * 0.5 for i in range(n)])


class _AlwaysLong:
    """A native-1m strategy that targets long every bar (maximum rebalance dust)."""

    def on_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx, bar) -> None:
        ctx.set_target(1)


class _ExitAfter:
    """Long until ``exit_at`` bars in, then flat — a real exit the band must not suppress."""

    def __init__(self, exit_at: int):
        self._exit_at = exit_at
        self._i = 0

    def on_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx, bar) -> None:
        ctx.set_target(1 if self._i < self._exit_at else 0)
        self._i += 1


class _Flip:
    """Long, then short — a real reversal the band must not suppress."""

    def __init__(self, flip_at: int):
        self._flip_at = flip_at
        self._i = 0

    def on_start(self, ctx) -> None:
        pass

    def on_bar(self, ctx, bar) -> None:
        ctx.set_target(1 if self._i < self._flip_at else -1)
        self._i += 1


class _Candidate:
    """A minimal Candidate: ``build`` hands the driver a ready strategy instance."""

    def __init__(self, strat):
        self._strat = strat

    def build(self, families):
        return self._strat


# --- 2a. rebalance dead-band -------------------------------------------------------------


def test_dead_band_suppresses_held_position_dust():
    # A held long re-trues nearly every bar with the band off (equity/price wobble), but the
    # band holds it after the single entry fill.
    bars = {"AAPL": _uptrend(20)}
    off = run_trading(candidates=[_Candidate(_AlwaysLong())], bars_by_symbol=bars)
    on = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        rebalance_band_pct=50.0,
    )
    assert off.fills > 5  # a sub-share fill nearly every bar
    assert on.fills == 1  # just the entry from flat; the held long then holds
    assert on.positions.get("AAPL", 0.0) > 0.0  # still long — held, not exited


def test_dead_band_never_suppresses_exit_or_flip():
    # Even with both thresholds set absurdly high, a target → 0 (exit) and a sign change (flip)
    # must execute; the band only guards same-direction re-truing.
    bars = {"AAPL": _uptrend(20)}
    band = {"rebalance_band_pct": 50.0, "min_order_notional": 1e12}
    ex = run_trading(candidates=[_Candidate(_ExitAfter(9))], bars_by_symbol=bars, **band)
    assert ex.positions.get("AAPL", 0.0) == 0.0  # exited flat, not pinned long by the band
    fl = run_trading(candidates=[_Candidate(_Flip(9))], bars_by_symbol=bars, **band)
    assert fl.positions["AAPL"] < 0.0  # reversed to short despite the band


def test_dead_band_default_is_a_no_op():
    # 0.0/0.0 must be byte-identical to the pre-band loop (regression guard for fill-count tests).
    bars = {"AAPL": _uptrend(20)}
    baseline = run_trading(candidates=[_Candidate(_AlwaysLong())], bars_by_symbol=bars)
    explicit = run_trading(
        candidates=[_Candidate(_AlwaysLong())],
        bars_by_symbol=bars,
        min_order_notional=0.0,
        rebalance_band_pct=0.0,
    )
    assert explicit.fills == baseline.fills
    assert explicit.orders_submitted == baseline.orders_submitted


# --- 2b. collapsed refusal logging -------------------------------------------------------


def test_refusal_collapse_logs_one_warning_not_per_bar(caplog):
    # A long position at a 50% cap, then a price crash: equity falls below the 3% floor and the
    # daily-loss latch trips. Because only half the account is invested, the notional cap re-sizes
    # the desired position ABOVE the held shares each low bar → an exposure-increasing order that
    # is refused every bar. That per-bar INFO flood must collapse to exactly one WARNING.
    bars = {"AAPL": make_ohlcv([100.0, 100.0, 60.0, 60.0, 60.0, 60.0])}
    with caplog.at_level(logging.INFO, logger="noctis.trading"):
        summary = run_trading(
            candidates=[_Candidate(_AlwaysLong())],
            bars_by_symbol=bars,
            limits=RiskLimits(
                max_position_pct=50.0, max_gross_exposure_pct=100.0, max_daily_loss_pct=3.0
            ),
        )
    assert summary.halt_latched is True
    assert summary.halt_floor == 97_000.0  # 3% below the 100k session start
    assert summary.halt_equity <= summary.halt_floor  # tripped at/under the floor
    assert summary.orders_refused >= 2  # refused every low-price bar, not merely once

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "daily-loss halt latched" in warnings[0].getMessage()
    # The old code logged an INFO line per refused bar; captured at INFO, none survive (they
    # dropped to DEBUG), so the flood is gone while the honest count stays on the summary.
    assert [r for r in caplog.records if "order refused" in r.getMessage()] == []
