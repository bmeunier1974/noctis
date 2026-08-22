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

from noctis.broker.exits import ExitState, ExitTrigger
from noctis.broker.paper import PaperBroker
from noctis.broker.position_driver import PositionDriver
from noctis.broker.seam import FeeModel, Side, SlippageModel
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


def test_execute_false_at_open_marks_the_bar_without_trading():
    """A suppressed open still prices the account honestly; the target stays owed."""
    broker = _broker()
    broker.set_price(SYMBOL, 100.0, 0)
    broker.rebalance_to(SYMBOL, 10.0)  # 10 units at 100 → cash 99_000, equity 100_000
    driver = _driver(broker, _scripted([1]), _sizer([20.0]))
    carried_fills = len(broker.fills)  # the seeding trade above, not this session's
    (first,) = _tape(_bar(110.0, 112.0, 109.0, 111.0))

    opened = driver.at_open(first, execute=False)

    assert opened.fill is None
    assert opened.skipped is False  # nothing was refused — nothing was offered
    assert broker.marks()[SYMBOL] == 110.0
    assert broker.equity() == 100_100.0  # 99_000 cash + 10 units marked at the open
    assert len(broker.fills) == carried_fills
    assert driver.pending_target == 1


def test_execute_false_at_close_still_decides_carries_and_marks():
    broker = _broker()
    driver = _driver(broker, _scripted([-1]), _sizer([10.0]))
    (first,) = _tape(_bar(100.0, 101.0, 99.0, 100.5))

    driver.at_open(first, execute=False)
    close = driver.at_close(first, execute=False)

    assert close.target == -1
    assert driver.pending_target == -1  # carried into the next open
    assert broker.marks()[SYMBOL] == 100.5
    assert broker.fills == []


def test_an_unlatched_close_executes_the_strategys_raw_target_and_carries_its_rules():
    rules = ExitRules(stop_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([1], exits=rules), _sizer([10.0]))
    (first,) = _tape(_bar(100.0, 101.0, 99.0, 100.0))

    driver.at_open(first)
    close = driver.at_close(first)

    assert close.raw_target == 1
    assert close.target == close.raw_target  # nothing suppresses it while unlatched
    assert close.exit_fill is None
    assert close.trigger is None
    assert driver.pending_target == 1
    assert driver.pending_exits == rules
    assert driver.latched is False


def test_a_long_stop_touched_intrabar_exits_at_the_stop_level_and_latches():
    """The armed rules are enforced against the bar the position was opened on."""
    rules = ExitRules(stop_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([1], exits=rules), _sizer([10.0]))
    first, second = _tape(_bar(99.0, 100.0, 98.0, 99.0), _bar(100.0, 101.0, 94.0, 96.0))

    driver.at_open(first)
    driver.at_close(first)  # decides +1 and arms the stop
    driver.at_open(second)  # long 10 units at 100 — the anchor
    close = driver.at_close(second)

    assert close.trigger == ExitTrigger(price=95.0, reason="stop")
    assert close.exit_fill is not None
    assert close.exit_fill.price == 95.0  # 100 * (1 - 0.05)
    assert close.exit_fill.reason == "stop"
    assert close.exit_fill.side is Side.SELL
    assert close.exit_fill.quantity == 10.0
    assert broker.position(SYMBOL).quantity == 0.0
    assert len(broker.fills) == 2  # the entry and the exit
    assert driver.latched is True
    assert close.raw_target == 1  # the strategy still wants the long...
    assert close.target == 0  # ...and the latch holds it flat anyway


def test_a_short_stop_mirrors_the_long_case():
    rules = ExitRules(stop_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([-1], exits=rules), _sizer([10.0]))
    first, second = _tape(_bar(99.0, 100.0, 98.0, 99.0), _bar(100.0, 106.0, 99.0, 104.0))

    driver.at_open(first)
    driver.at_close(first)
    driver.at_open(second)  # short 10 units at 100
    close = driver.at_close(second)

    assert close.trigger == ExitTrigger(price=105.0, reason="stop")
    assert close.exit_fill is not None
    assert close.exit_fill.price == 105.0  # 100 * (1 + 0.05), adverse for a short
    assert close.exit_fill.reason == "stop"
    assert close.exit_fill.side is Side.BUY
    assert broker.position(SYMBOL).quantity == 0.0
    assert driver.latched is True
    assert close.target == 0


def test_the_trail_measures_from_the_prior_bars_extreme_not_this_bars():
    """Evaluate before ratchet: a bar that makes a new high exits at the OLD trail level.

    The high that would raise the mark may print after the low that breaches the level,
    so ratcheting off the same bar the trail is judged on is intrabar lookahead.
    """
    rules = ExitRules(trail_pct=0.10)
    broker = _broker()
    driver = _driver(broker, _scripted([1], exits=rules), _sizer([10.0]))
    first, second, third = _tape(
        _bar(100.0, 101.0, 99.0, 100.0),
        _bar(100.0, 120.0, 99.0, 119.0),  # runs up to 120 without breaching 100 * 0.9
        _bar(118.0, 125.0, 107.0, 108.0),  # a new high AND a breach of the prior level
    )

    driver.at_open(first)
    driver.at_close(first)
    driver.at_open(second)  # long 10 units at 100
    held = driver.at_close(second)

    assert held.trigger is None  # 90 is far below this bar's low
    assert held.exit_fill is None

    driver.at_open(third)
    close = driver.at_close(third)

    assert close.trigger is not None
    assert close.trigger.reason == "trail"
    assert close.exit_fill is not None
    assert close.exit_fill.price == 108.0  # 120 (the PRIOR bar's high) * 0.9, not 125 * 0.9
    assert broker.position(SYMBOL).quantity == 0.0
    assert driver.latched is True


def test_a_take_profit_gapped_through_at_the_open_banks_the_better_price():
    rules = ExitRules(take_profit_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([1], exits=rules), _sizer([10.0]))
    first, second, third = _tape(
        _bar(99.0, 100.0, 98.0, 99.0),
        _bar(100.0, 101.0, 99.0, 100.0),  # long 10 units at 100; 105 is out of reach
        _bar(108.0, 110.0, 107.0, 109.0),  # gaps straight past the take-profit level
    )

    driver.at_open(first)
    driver.at_close(first)
    driver.at_open(second)
    assert driver.at_close(second).trigger is None

    driver.at_open(third)
    close = driver.at_close(third)

    assert close.trigger == ExitTrigger(price=108.0, reason="take_profit")
    assert close.exit_fill is not None
    assert close.exit_fill.price == 108.0  # the open, better than the 105 level
    assert close.exit_fill.reason == "take_profit"
    assert broker.position(SYMBOL).quantity == 0.0


def test_a_bar_that_touches_both_levels_resolves_to_the_stop():
    """The worst case the OHLC cannot disprove — never the flattering one."""
    rules = ExitRules(stop_pct=0.05, take_profit_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([1], exits=rules), _sizer([10.0]))
    first, second = _tape(_bar(99.0, 100.0, 98.0, 99.0), _bar(100.0, 106.0, 94.0, 100.0))

    driver.at_open(first)
    driver.at_close(first)
    driver.at_open(second)  # long 10 units at 100
    close = driver.at_close(second)

    assert close.trigger == ExitTrigger(price=95.0, reason="stop")
    assert close.exit_fill is not None
    assert close.exit_fill.price == 95.0
    assert close.exit_fill.reason == "stop"
    assert broker.position(SYMBOL).quantity == 0.0


def test_the_latch_holds_flat_until_the_strategy_asks_for_something_new():
    """An exit is not a suggestion: re-asking for the same stance changes nothing."""
    rules = ExitRules(stop_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([1, 1, 1, 0, 1], exits=rules), _sizer([10.0]))
    b1, b2, b3, b4, b5, b6 = _tape(
        _bar(99.0, 100.0, 98.0, 99.0),
        _bar(100.0, 101.0, 94.0, 96.0),  # long at 100, stopped out at 95
        _bar(96.0, 97.0, 95.0, 96.0),  # the strategy re-asks for +1 — refused
        _bar(96.0, 97.0, 95.0, 96.0),  # it changes its mind to 0 — the latch lifts
        _bar(100.0, 101.0, 99.0, 100.0),
        _bar(110.0, 111.0, 109.0, 110.0),
    )

    driver.at_open(b1)
    driver.at_close(b1)
    driver.at_open(b2)
    assert driver.at_close(b2).exit_fill is not None
    fills_after_the_exit = len(broker.fills)

    held = driver.at_open(b3)
    still_latched = driver.at_close(b3)

    assert held.fill is None
    assert still_latched.exit_fill is None
    assert still_latched.raw_target == 1
    assert still_latched.target == 0
    assert driver.latched is True
    assert len(broker.fills) == fills_after_the_exit

    driver.at_open(b4)
    unlatched = driver.at_close(b4)

    assert driver.latched is False
    assert unlatched.target == 0  # the new value is 0 — nothing to trade yet

    driver.at_open(b5)
    re_entry = driver.at_close(b5)

    assert re_entry.target == 1  # +1 is a change of mind now, and executes normally
    opened = driver.at_open(b6)

    assert opened.fill is not None
    assert opened.fill.price == 110.0
    assert opened.fill.quantity == 10.0
    assert opened.fill.reason == "target"
    assert broker.position(SYMBOL).quantity == 10.0
    assert driver.latched is False


def test_a_carried_position_is_protected_from_the_first_rules_it_is_given():
    """Rules arm at the close that declares them — never retroactively on that bar."""
    broker = _broker()
    broker.set_price(SYMBOL, 100.0, 0)
    broker.rebalance_to(SYMBOL, 10.0)  # an earlier session's long, avg price 100
    seeding_fills = len(broker.fills)
    driver = _driver(broker, _scripted([1], exits=ExitRules(stop_pct=0.05)), _sizer([10.0]))
    first, second = _tape(
        _bar(110.0, 112.0, 90.0, 111.0),  # would have breached 95 — nothing armed yet
        _bar(110.0, 111.0, 94.0, 96.0),
    )

    driver.at_open(first)
    unarmed = driver.at_close(first)  # declares the stop for the first time

    assert unarmed.exit_fill is None
    assert unarmed.trigger is None
    assert len(broker.fills) == seeding_fills
    assert broker.position(SYMBOL).quantity == 10.0

    driver.at_open(second)
    close = driver.at_close(second)

    assert close.trigger == ExitTrigger(price=95.0, reason="stop")
    assert close.exit_fill is not None
    assert close.exit_fill.price == 95.0  # the carried avg price 100 * 0.95, not 110 * 0.95
    assert broker.position(SYMBOL).quantity == 0.0
    assert driver.latched is True


def test_exit_tracking_re_anchors_at_the_fill_price_and_clears_when_flat():
    """The anchor is the price actually paid — at an open, and again at a flip."""
    rules = ExitRules(stop_pct=0.05)
    broker = _broker()
    driver = _driver(broker, _scripted([1, 1, -1, 1, 1, 1], exits=rules), _sizer([10.0]))
    b1, b2, b3, b4, b5, b6 = _tape(
        _bar(90.0, 91.0, 89.0, 90.0),  # decides +1 with the close at 90
        _bar(100.0, 101.0, 94.0, 96.0),  # fills at the OPEN, 100 → stop 95, not 85.5
        _bar(96.0, 97.0, 95.0, 96.0),  # flat: the cleared anchor's level goes unnoticed
        _bar(100.0, 101.0, 99.0, 100.0),  # re-opens short at 100
        _bar(120.0, 121.0, 119.0, 120.0),  # flips long at 120 → the new anchor
        _bar(118.0, 119.0, 113.0, 114.0),  # 120 * 0.95 = 114
    )

    driver.at_open(b1)
    driver.at_close(b1)
    driver.at_open(b2)
    opened = driver.at_close(b2)

    assert opened.exit_fill is not None
    assert opened.exit_fill.price == 95.0  # anchored at the fill, not the decision bar

    flat = driver.at_open(b3)
    cleared = driver.at_close(b3)

    assert flat.fill is None
    assert cleared.exit_fill is None  # the low touches 95, but nothing is anchored
    assert cleared.trigger is None
    assert broker.position(SYMBOL).quantity == 0.0

    driver.at_open(b4)
    driver.at_close(b4)
    flipped = driver.at_open(b5)
    driver.at_close(b5)

    assert flipped.fill is not None
    assert flipped.fill.quantity == 20.0  # short 10 → long 10, through flat
    assert flipped.fill.price == 120.0
    assert broker.position(SYMBOL).quantity == 10.0

    driver.at_open(b6)
    close = driver.at_close(b6)

    assert close.trigger == ExitTrigger(price=114.0, reason="stop")
    assert close.exit_fill is not None
    assert close.exit_fill.price == 114.0  # the FLIP fill price * 0.95
    assert broker.position(SYMBOL).quantity == 0.0


def test_execute_false_at_close_skips_the_exit_step_ratchet_included():
    """A bar it could not act on must not advance the trail it will be judged by."""
    rules = ExitRules(trail_pct=0.10)
    broker = _broker()
    driver = _driver(broker, _scripted([1], exits=rules), _sizer([10.0]))
    b1, b2, b3, b4 = _tape(
        _bar(100.0, 101.0, 99.0, 100.0),
        _bar(100.0, 120.0, 99.0, 119.0),  # long at 100; the best ratchets to 120
        _bar(118.0, 125.0, 107.0, 108.0),  # would breach 108 and mark 125 — suppressed
        _bar(110.0, 111.0, 107.0, 108.0),
    )

    driver.at_open(b1)
    driver.at_close(b1)
    driver.at_open(b2)
    driver.at_close(b2)
    entry_fills = len(broker.fills)

    driver.at_open(b3)
    suppressed = driver.at_close(b3, execute=False)

    assert suppressed.exit_fill is None
    assert suppressed.trigger is None
    assert len(broker.fills) == entry_fills
    assert broker.position(SYMBOL).quantity == 10.0  # still held
    assert driver.latched is False
    assert suppressed.target == 1  # on_bar still ran...
    assert driver.pending_target == 1  # ...and its decision still carried
    assert driver.pending_exits == rules
    assert broker.marks()[SYMBOL] == 108.0  # the close was still marked

    driver.at_open(b4)
    close = driver.at_close(b4)

    assert close.exit_fill is not None
    assert close.exit_fill.price == 108.0  # 120 * 0.9 — the suppressed bar's 125 never landed
    assert close.exit_fill.reason == "trail"
