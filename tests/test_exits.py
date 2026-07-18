"""The exit engine's contract table — pure evaluate/ratchet over value types.

These tests ARE the conservative-policy contract from the fill-model section of
docs/architecture.md: gap-through fills at the open, stop beats take-profit on a same-bar
touch, the trail measures from the prior bar's extreme, and shorts mirror longs symmetrically.
No driver is involved — the same functions are consumed by the simulator and the live driver.
"""

from __future__ import annotations

import pytest

from noctis.broker.exits import ExitState, evaluate, ratchet
from noctis.strategies.base import Bar, ExitRules


def _bar(open_: float, high: float, low: float, close: float) -> Bar:
    return Bar(ts_event=0, open=open_, high=high, low=low, close=close, volume=1.0)


def test_long_stop_touched_intrabar_fills_at_the_stop_level():
    rules = ExitRules(stop_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(99.0, 100.0, 94.0, 96.0))

    assert trigger is not None
    assert trigger.price == 95.0  # 100 * (1 - 0.05)
    assert trigger.reason == "stop"


def test_long_stop_gapped_through_at_open_fills_at_the_open():
    """An overnight gap past the stop is priced honestly — at the open, not the level."""
    rules = ExitRules(stop_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(90.0, 92.0, 89.0, 91.0))

    assert trigger is not None
    assert trigger.price == 90.0
    assert trigger.reason == "stop"


def test_long_take_profit_touched_intrabar_fills_at_the_level():
    rules = ExitRules(take_profit_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(101.0, 106.0, 100.0, 104.0))

    assert trigger is not None
    assert trigger.price == 105.0  # 100 * (1 + 0.05)
    assert trigger.reason == "take_profit"


def test_long_take_profit_gapped_through_at_open_fills_at_the_open():
    """A favorable gap past the take-profit banks the better open price."""
    rules = ExitRules(take_profit_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(108.0, 110.0, 107.0, 109.0))

    assert trigger is not None
    assert trigger.price == 108.0
    assert trigger.reason == "take_profit"


def test_stop_and_take_profit_both_touched_same_bar_assumes_the_stop_fired():
    """The intrabar path is unknowable from OHLC — ambiguity resolves to the worst case."""
    rules = ExitRules(stop_pct=0.05, take_profit_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(100.0, 106.0, 94.0, 100.0))

    assert trigger is not None
    assert trigger.price == 95.0
    assert trigger.reason == "stop"


def test_long_trail_measures_drawdown_from_best_not_entry():
    """A winner gives back at most trail_pct from its best favorable extreme."""
    rules = ExitRules(trail_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=120.0)

    trigger = evaluate(rules, state, _bar(116.0, 117.0, 113.0, 115.0))

    assert trigger is not None
    assert trigger.price == 114.0  # 120 * (1 - 0.05); entry-measured would never fire here
    assert trigger.reason == "trail"


def test_long_trail_tighter_than_stop_binds_first():
    """When both protective levels are armed, the tighter one is the one that fires."""
    rules = ExitRules(stop_pct=0.10, trail_pct=0.02)
    state = ExitState(direction=1, entry_price=100.0, best=110.0)

    # low touches the trail level (107.8) but stays far above the fixed stop (90).
    trigger = evaluate(rules, state, _bar(109.0, 109.5, 107.0, 108.0))

    assert trigger is not None
    assert trigger.price == pytest.approx(107.8)
    assert trigger.reason == "trail"


def test_ratchet_advances_the_long_high_water_mark_after_evaluate():
    state = ExitState(direction=1, entry_price=100.0, best=100.0)

    advanced = ratchet(state, _bar(101.0, 110.0, 100.0, 108.0))

    assert advanced.best == 110.0
    assert advanced.entry_price == 100.0
    assert advanced.direction == 1
    assert ratchet(advanced, _bar(108.0, 109.0, 107.0, 108.0)).best == 110.0  # never regresses


def test_trail_measures_from_the_prior_bar_extreme_not_the_same_bar():
    """The ordering case: this bar's high sets a new best AND its low would breach the
    trail measured from that new best — but not from the prior best. Ratcheting before
    evaluating would manufacture an intrabar-lookahead exit here."""
    rules = ExitRules(trail_pct=0.05)
    state = ExitState(direction=1, entry_price=100.0, best=100.0)
    bar = _bar(100.0, 110.0, 104.4, 105.0)  # from new best 110 the level is 104.5

    assert evaluate(rules, state, bar) is None  # prior best 100 → level 95, untouched

    state = ratchet(state, bar)
    trigger = evaluate(rules, state, _bar(105.0, 106.0, 104.0, 104.5))

    assert trigger is not None
    assert trigger.price == pytest.approx(104.5)  # 110 * (1 - 0.05), now armed
    assert trigger.reason == "trail"


# --- the short mirror: stop above, take-profit below, best is the low-water mark ----------


def test_short_stop_touched_intrabar_fills_at_the_stop_level():
    rules = ExitRules(stop_pct=0.05)
    state = ExitState(direction=-1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(101.0, 106.0, 100.0, 104.0))

    assert trigger is not None
    assert trigger.price == 105.0  # 100 * (1 + 0.05) — above entry for a short
    assert trigger.reason == "stop"


def test_short_stop_gapped_through_at_open_fills_at_the_open():
    rules = ExitRules(stop_pct=0.05)
    state = ExitState(direction=-1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(110.0, 112.0, 109.0, 111.0))

    assert trigger is not None
    assert trigger.price == 110.0
    assert trigger.reason == "stop"


def test_short_take_profit_touched_intrabar_fills_at_the_level():
    rules = ExitRules(take_profit_pct=0.05)
    state = ExitState(direction=-1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(99.0, 100.0, 94.0, 96.0))

    assert trigger is not None
    assert trigger.price == 95.0  # 100 * (1 - 0.05) — below entry for a short
    assert trigger.reason == "take_profit"


def test_short_take_profit_gapped_through_at_open_fills_at_the_open():
    rules = ExitRules(take_profit_pct=0.05)
    state = ExitState(direction=-1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(92.0, 94.0, 91.0, 93.0))

    assert trigger is not None
    assert trigger.price == 92.0
    assert trigger.reason == "take_profit"


def test_short_stop_and_take_profit_both_touched_same_bar_assumes_the_stop_fired():
    rules = ExitRules(stop_pct=0.05, take_profit_pct=0.05)
    state = ExitState(direction=-1, entry_price=100.0, best=100.0)

    trigger = evaluate(rules, state, _bar(100.0, 106.0, 94.0, 100.0))

    assert trigger is not None
    assert trigger.price == 105.0
    assert trigger.reason == "stop"


def test_short_trail_measures_rally_from_low_water_mark():
    rules = ExitRules(trail_pct=0.05)
    state = ExitState(direction=-1, entry_price=100.0, best=80.0)

    trigger = evaluate(rules, state, _bar(83.0, 85.0, 82.0, 84.5))

    assert trigger is not None
    assert trigger.price == 84.0  # 80 * (1 + 0.05); entry-measured would never fire here
    assert trigger.reason == "trail"


def test_ratchet_advances_the_short_low_water_mark():
    state = ExitState(direction=-1, entry_price=100.0, best=100.0)

    advanced = ratchet(state, _bar(99.0, 100.0, 90.0, 92.0))

    assert advanced.best == 90.0
    assert ratchet(advanced, _bar(92.0, 95.0, 91.0, 94.0)).best == 90.0  # never regresses


# --- boundary rows: exactly-at-level counts as touched; nothing armed never fires ----------


@pytest.mark.parametrize("direction", [1, -1])
def test_exactly_at_level_counts_as_touched(direction: int):
    """'Adverse move ≥ this fraction' is inclusive — the extreme landing on the level fires."""
    rules = ExitRules(stop_pct=0.05, take_profit_pct=0.10)
    state = ExitState(direction=direction, entry_price=100.0, best=100.0)
    stop_level = 100.0 * (1.0 - direction * 0.05)
    tp_level = 100.0 * (1.0 + direction * 0.10)

    stop_bar = (
        _bar(99.0, 100.0, stop_level, 96.0)
        if direction > 0
        else _bar(101.0, stop_level, 100.0, 104.0)
    )
    stopped = evaluate(rules, state, stop_bar)
    assert stopped is not None and stopped.price == stop_level and stopped.reason == "stop"

    tp_bar = (
        _bar(101.0, tp_level, 100.0, 104.0) if direction > 0 else _bar(99.0, 100.0, tp_level, 96.0)
    )
    banked = evaluate(rules, state, tp_bar)
    assert banked is not None and banked.price == tp_level and banked.reason == "take_profit"


@pytest.mark.parametrize("direction", [1, -1])
def test_no_rules_armed_never_fires(direction: int):
    """Empty rules are the no-position/warmup semantics at this layer: nothing can fire."""
    state = ExitState(direction=direction, entry_price=100.0, best=100.0)

    crash_bar = _bar(50.0, 150.0, 25.0, 60.0)  # wildly beyond any level in both directions
    assert evaluate(ExitRules(), state, crash_bar) is None
