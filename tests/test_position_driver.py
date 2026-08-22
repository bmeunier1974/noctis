"""The position driver's contract — one symbol walked through bars under the fill model.

Tape style of the exit-engine tests: hand-built bars, a scripted strategy, and a
:class:`PaperBroker` with fees and slippage switched off, so every asserted price is the
tape's own price. These tests pin the step order the backtest simulator and the live
trading day both delegate to — decide on bar *t*, fill at bar *t+1*'s open — and the
seeding of a driver from whatever position the account already carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from noctis.broker.paper import PaperBroker
from noctis.broker.position_driver import PositionDriver
from noctis.broker.seam import FeeModel, SlippageModel
from noctis.strategies.base import Bar, ExitRules, TargetContext, TraderStrategy

SYMBOL = "SYM"


def _broker() -> PaperBroker:
    """A frictionless paper account: fills land exactly on the tape's price."""
    return PaperBroker(fee_model=FeeModel(bps=0.0), slippage_model=SlippageModel(bps=0.0))


def _bar(open_: float, high: float, low: float, close: float, ts: int = 0) -> Bar:
    return Bar(ts_event=ts, open=open_, high=high, low=low, close=close, volume=1.0)


def _tape(*bars: Bar) -> list[Bar]:
    """Re-stamp a tape with increasing timestamps, so a mark's ts is unambiguous."""
    return [
        Bar(ts_event=i + 1, open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume)
        for i, b in enumerate(bars)
    ]


def _scripted(targets: list[int], exits: ExitRules | None = None) -> TraderStrategy:
    """A strategy whose per-bar target is dictated by bar index, ignoring the tape."""

    class _Scripted(TraderStrategy):
        name = "scripted"

        @dataclass(frozen=True)
        class Params:
            pass

        params_cls = Params

        def on_start(self, ctx) -> None:
            self._i = -1

        def on_bar(self, ctx, bar) -> None:
            self._i += 1
            target = targets[self._i] if self._i < len(targets) else targets[-1]
            ctx.set_target(target, exits=exits)

        @classmethod
        def param_space(cls):
            return []

    return _Scripted.default()


def _sizer(scale: list[float]):
    """Ten units per unit of target, at any price — the scale is mutable for re-trues."""

    def sizer(symbol: str, target: int, price: float) -> float | None:
        return float(target) * scale[0]

    return sizer


def _driver(broker: PaperBroker, strategy: TraderStrategy, sizer) -> PositionDriver:
    """Build a driver the way its callers do — the strategy is started by the caller."""
    ctx = TargetContext()
    strategy.on_start(ctx)
    return PositionDriver.from_position(broker, SYMBOL, strategy, ctx, sizer)


def test_decision_on_a_bar_fills_at_the_next_bars_open():
    """The fill model, end to end: decided on bar t, executed at bar t+1's open."""
    broker = _broker()
    driver = _driver(broker, _scripted([1]), _sizer([10.0]))
    first, second = _tape(_bar(100.0, 101.0, 99.0, 100.0), _bar(110.0, 112.0, 109.0, 111.0))

    assert driver.at_open(first).fill is None  # nothing was decided yet
    close = driver.at_close(first)
    assert close.target == 1
    assert broker.fills == []  # the decision does not trade on its own bar

    opened = driver.at_open(second)

    assert opened.skipped is False
    assert opened.fill is not None
    assert opened.fill.quantity == 10.0
    assert opened.fill.price == 110.0  # bar t+1's OPEN, never its close
    assert opened.fill.reason == "target"
    assert broker.position(SYMBOL).quantity == 10.0


def test_from_position_on_a_flat_account_seeds_a_flat_pending_target():
    broker = _broker()
    driver = _driver(broker, _scripted([1]), _sizer([10.0]))

    assert driver.pending_target == 0
    assert driver.pending_exits is None
    assert driver.latched is False

    opened = driver.at_open(_bar(100.0, 101.0, 99.0, 100.0, ts=1))

    assert opened.fill is None
    assert broker.fills == []


def test_at_close_before_at_open_raises():
    driver = _driver(_broker(), _scripted([1]), _sizer([10.0]))
    first, second = _tape(_bar(100.0, 101.0, 99.0, 100.0), _bar(101.0, 102.0, 100.0, 101.0))

    with pytest.raises(RuntimeError):
        driver.at_close(first)

    driver.at_open(first)
    driver.at_close(first)
    with pytest.raises(RuntimeError):  # a second close without an open between
        driver.at_close(second)
