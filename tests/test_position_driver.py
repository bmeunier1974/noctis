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

from noctis.broker.exits import ExitState
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


def test_a_none_sizer_answer_skips_the_bar_and_holds_the_pending_target():
    """A refusal to trade this bar is not a change of mind: the target stays pending."""
    broker = _broker()
    scale = [10.0]
    refuse = [True]

    def sizer(symbol: str, target: int, price: float) -> float | None:
        return None if refuse[0] else float(target) * scale[0]

    driver = _driver(broker, _scripted([1]), sizer)
    first, second, third = _tape(
        _bar(100.0, 101.0, 99.0, 100.0),
        _bar(110.0, 112.0, 109.0, 111.0),
        _bar(120.0, 122.0, 119.0, 121.0),
    )

    driver.at_open(first)
    driver.at_close(first)

    skipped = driver.at_open(second)

    assert skipped.skipped is True
    assert skipped.fill is None
    assert broker.fills == []
    assert driver.pending_target == 1  # still owed, unchanged by the refusal
    driver.at_close(second)

    refuse[0] = False
    opened = driver.at_open(third)

    assert opened.skipped is False
    assert opened.fill is not None
    assert opened.fill.quantity == 10.0
    assert opened.fill.price == 120.0
    assert broker.position(SYMBOL).quantity == 10.0


def test_from_position_on_a_carried_long_holds_it_until_the_strategy_decides():
    """A carried position keeps its direction and anchors exits at its average price."""
    broker = _broker()
    broker.set_price(SYMBOL, 100.0, 0)
    broker.rebalance_to(SYMBOL, 10.0)  # an earlier session's long, avg price 100
    scale = [10.0]

    driver = _driver(broker, _scripted([1]), _sizer(scale))

    assert driver.pending_target == 1
    assert driver.exit_state == ExitState(direction=1, entry_price=100.0, best=100.0)

    first, second = _tape(_bar(110.0, 112.0, 109.0, 111.0), _bar(120.0, 122.0, 119.0, 121.0))
    held = driver.at_open(first)

    assert held.fill is None  # the same size at a new price: the position simply holds
    assert held.skipped is False
    assert broker.position(SYMBOL).quantity == 10.0

    driver.at_close(first)
    scale[0] = 20.0  # the sizer now wants a bigger position
    re_trued = driver.at_open(second)

    assert re_trued.fill is not None
    assert re_trued.fill.quantity == 10.0  # the increment, not the whole position
    assert re_trued.fill.price == 120.0
    assert re_trued.fill.reason == "target"
    assert broker.position(SYMBOL).quantity == 20.0


def test_from_position_on_a_carried_short_seeds_a_short_pending_target():
    broker = _broker()
    broker.set_price(SYMBOL, 50.0, 0)
    broker.rebalance_to(SYMBOL, -4.0)

    driver = _driver(broker, _scripted([0]), _sizer([10.0]))

    assert driver.pending_target == -1
    assert driver.exit_state == ExitState(direction=-1, entry_price=50.0, best=50.0)
