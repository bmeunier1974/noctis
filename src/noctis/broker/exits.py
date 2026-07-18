"""The exit engine — pure evaluation of protective-exit rules against one bar.

One implementation, two drivers: the backtest simulator and the live trading day call the
same ``evaluate``/``ratchet`` on the strategy-timeframe bars, which is what structurally
prevents backtest/live drift (the same shape as ``TargetContext`` and
``StreamingAggregator``). Everything here is a pure function over frozen value types — no
I/O, no globals, no randomness. The conservative intrabar policy is locked in the
fill-model section of docs/architecture.md; the tests in ``tests/test_exits.py`` are its
contract table.
"""

from __future__ import annotations

from dataclasses import dataclass

from noctis.strategies.base import Bar, ExitRules

__all__ = ["ExitState", "ExitTrigger", "evaluate", "ratchet"]


@dataclass(frozen=True)
class ExitState:
    """Exit tracking for one open position, re-anchored on every open/flip."""

    direction: int  # +1 long / -1 short
    entry_price: float
    best: float  # best favorable extreme since entry (prior-bar ratchet)


@dataclass(frozen=True)
class ExitTrigger:
    """An exit that fired: the fill price per the conservative policy, and why."""

    price: float
    reason: str  # "stop" | "take_profit" | "trail"


def evaluate(rules: ExitRules, state: ExitState, bar: Bar) -> ExitTrigger | None:
    """Evaluate armed rules against one bar's OHLC; ``None`` when nothing fired.

    One signed implementation covers both directions (``d = ±1``): a long's stop sits
    below entry and triggers on the low; a short mirrors symmetrically (stop above,
    take-profit below, ``best`` is the low-water mark). ``d * a <= d * b`` reads as
    "``a`` is at or beyond ``b`` in the adverse direction".
    """
    d = 1 if state.direction > 0 else -1

    # The binding protective level is the tightest of fixed stop and trail (a tie stays
    # labeled "stop"); checked before take-profit, so a bar touching both resolves to
    # the stop — the worst case the OHLC cannot disprove.
    protective: tuple[float, str] | None = None
    if rules.stop_pct is not None:
        protective = (state.entry_price * (1.0 - d * rules.stop_pct), "stop")
    if rules.trail_pct is not None:
        trail_level = state.best * (1.0 - d * rules.trail_pct)
        if protective is None or d * trail_level > d * protective[0]:
            protective = (trail_level, "trail")
    if protective is not None:
        level, reason = protective
        if d * bar.open <= d * level:  # gapped through → fill at the open, never the level
            return ExitTrigger(price=bar.open, reason=reason)
        adverse_extreme = bar.low if d > 0 else bar.high
        if d * adverse_extreme <= d * level:
            return ExitTrigger(price=level, reason=reason)

    if rules.take_profit_pct is not None:
        level = state.entry_price * (1.0 + d * rules.take_profit_pct)
        if d * bar.open >= d * level:  # favorable gap banks the better open price
            return ExitTrigger(price=bar.open, reason="take_profit")
        favorable_extreme = bar.high if d > 0 else bar.low
        if d * favorable_extreme >= d * level:
            return ExitTrigger(price=level, reason="take_profit")
    return None


def ratchet(state: ExitState, bar: Bar) -> ExitState:
    """Advance the favorable extreme after ``evaluate`` — never before it.

    Running after ``evaluate`` is what makes ``state.best`` the *prior* bar's extreme when
    the trail is measured: the high that sets a new mark may occur after the low that would
    hit it, so ratcheting and triggering off the same bar would be intrabar lookahead.
    """
    if state.direction > 0:
        best = max(state.best, bar.high)
    else:
        best = min(state.best, bar.low)
    if best == state.best:
        return state
    return ExitState(direction=state.direction, entry_price=state.entry_price, best=best)
