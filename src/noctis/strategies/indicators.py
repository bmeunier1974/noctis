"""Indicator helpers for authored (one-file) strategies — keep ``on_bar`` code short.

Two styles, both O(period) per bar so a replayed backtest stays O(n·period):

* **Tail functions** over a plain ``list``/``deque`` of floats the strategy accumulates:
  ``sma``, ``ema``, ``rsi``, ``atr``, ``stdev``, ``zscore``, ``bollinger``, ``roc``, ``wma``,
  ``highest``, ``lowest``, ``stoch_k``, ``cci``, ``bars_since``, plus the crossing
  predicates ``cross_above`` / ``cross_below``. Each looks at the most recent ``period``
  values only and returns ``None`` during warmup **and** on degenerate windows (zero
  deviation for ``zscore``, flat range for ``stoch_k``/``cci``, zero base for ``roc``),
  so always guard with ``if value is not None``. Docstrings state the exact Pine/Wilder
  variant each one matches.
* **Stateful classes** (re-exported from the golden-tested spec interpreter library) for
  Wilder-smoothed / seeded / cumulative indicators whose exact values need full history:
  ``SmaState``, ``EmaState``, ``RsiState``, ``AtrState``, ``MacdState``, ``VwapState``,
  ``AdxState`` (``.plus_di``/``.minus_di`` properties), ``ObvState``, ``StochState``
  (%K/%D dict), ``SupertrendState`` (st/dir dict), ``ZScoreState``, ``RollingExtremeState``.
  Create them in ``on_start`` and call ``.update(bar)`` each bar (returns ``nan`` during
  warmup). Their vectorised twins (``adx_vector``, ``obv_vector``, ``stoch_vector``,
  ``supertrend_vector``) are re-exported too for ``signals()`` overrides.

The tail ``rsi`` uses simple-average gains/losses over the window (Cutler's RSI — the same
math as the seed ``rsi_meanrev`` family), not Wilder smoothing; use ``RsiState`` when you
want the Wilder variant.

**Deliberately not provided** (don't burn a session asking): *Ichimoku* — five lines of
config-heavy convention that is rarely a thesis in itself; *valuewhen / pivothigh-style
bar-indexing built-ins* — Python authors express "value at the bar where X happened" more
clearly with their own deque + ``bars_since``; *volume-profile indicators* — they need
intrabar data the 1-minute lake does not carry.
"""

from __future__ import annotations

from collections.abc import Sequence

from noctis.data.aggregate import NATIVE_TIMEFRAME, StreamingAggregator, validate_timeframe
from noctis.strategies.base import Bar
from noctis.strategies.spec.indicators import (  # noqa: F401 — re-exported for authors
    AdxState,
    AtrState,
    EmaState,
    MacdState,
    ObvState,
    RollingExtremeState,
    RsiState,
    SmaState,
    StochState,
    SupertrendState,
    VwapState,
    ZScoreState,
    adx_vector,
    obv_vector,
    stoch_vector,
    supertrend_vector,
)

__all__ = [
    "sma",
    "ema",
    "rsi",
    "atr",
    "stdev",
    "zscore",
    "bollinger",
    "roc",
    "wma",
    "highest",
    "lowest",
    "stoch_k",
    "cci",
    "bars_since",
    "cross_above",
    "cross_below",
    "SmaState",
    "AdxState",
    "EmaState",
    "RsiState",
    "AtrState",
    "MacdState",
    "ObvState",
    "StochState",
    "SupertrendState",
    "VwapState",
    "ZScoreState",
    "RollingExtremeState",
    "adx_vector",
    "obv_vector",
    "stoch_vector",
    "supertrend_vector",
    "HtfBars",
]


class HtfBars:
    """Higher-timeframe bar aggregator you own like any other state (Pine ``request.security``).

    Construct it in ``on_start`` with a timeframe strictly coarser than the strategy's own,
    feed every base bar in ``on_bar``, and receive a completed higher-timeframe :class:`Bar`
    or ``None``::

        def on_start(self, ctx):
            self.htf = ind.HtfBars("1h")
            self.htf_ema = ind.EmaState(self.params.trend_period)

        def on_bar(self, ctx, bar):
            done = self.htf.add(bar)          # completed 1h bar or None
            if done is not None:
                self.trend = self.htf_ema.update(done.close)

    Lookahead-free by construction: it wraps :class:`~noctis.data.StreamingAggregator`, which
    emits a bucket only once the first bar of the *next* bucket arrives — so a higher-timeframe
    bar is never visible before its bucket has fully closed, and the session-final partial
    bucket is never emitted. ``on_bar`` still receives only base-timeframe bars, so the
    walk-forward splitter, write-gate replay, and live loop are all untouched.

    The higher timeframe must be a multiple of the strategy's declared ``timeframe`` for the
    ``ts // bucket`` bucketing to be meaningful — the wrapper cannot see the declaring class,
    so that is a documented convention. ``"1m"`` (the aggregator's pass-through case, which
    would defeat the wrapper) and any unsupported timeframe are rejected at construction.
    """

    def __init__(self, timeframe: str):
        validate_timeframe(timeframe)
        if timeframe == NATIVE_TIMEFRAME:
            raise ValueError(
                f"HtfBars needs a timeframe coarser than the base bars, not {timeframe!r} "
                "(the native pass-through would defeat the wrapper); use e.g. '5m', '1h', '1d'."
            )
        self.timeframe = timeframe
        self._agg = StreamingAggregator(timeframe)

    def add(self, bar: Bar) -> Bar | None:
        """Feed one base bar; return the just-completed higher-timeframe bar, or ``None``."""
        return self._agg.add(bar)


def sma(values: Sequence[float], period: int) -> float | None:
    """Mean of the last ``period`` values; ``None`` until enough history."""
    period = int(period)
    if period < 1 or len(values) < period:
        return None
    return sum(list(values)[-period:]) / period


def ema(values: Sequence[float], period: int) -> float | None:
    """EMA of the last ``4 × period`` values (SMA-seeded); ``None`` until enough history.

    Bounded-window approximation: after 3 extra periods the truncation error is < 1%, and
    the cost stays O(period) per bar. Use :class:`EmaState` for the exact full-history EMA.
    """
    period = int(period)
    if period < 1 or len(values) < period:
        return None
    window = list(values)[-4 * period :]
    mult = 2.0 / (period + 1.0)
    out = sum(window[:period]) / period
    for v in window[period:]:
        out = (v - out) * mult + out
    return out


def rsi(values: Sequence[float], period: int) -> float | None:
    """Cutler's RSI (simple-average gains/losses) over the last ``period`` deltas.

    Needs ``period + 1`` values; ``None`` during warmup. 0–100.
    """
    period = int(period)
    if period < 1 or len(values) < period + 1:
        return None
    window = list(values)[-(period + 1) :]
    gains = losses = 0.0
    for prev, cur in zip(window[:-1], window[1:], strict=True):
        delta = cur - prev
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0.0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> float | None:
    """Simple-average true range over the last ``period`` bars; ``None`` during warmup.

    Needs ``period + 1`` bars (true range uses the prior close). Use :class:`AtrState` for
    the Wilder-smoothed variant.
    """
    period = int(period)
    n = min(len(highs), len(lows), len(closes))
    if period < 1 or n < period + 1:
        return None
    k = period + 1
    hs, ls, cs = list(highs)[-k:], list(lows)[-k:], list(closes)[-k:]
    total = 0.0
    for i in range(1, period + 1):
        total += max(hs[i] - ls[i], abs(hs[i] - cs[i - 1]), abs(ls[i] - cs[i - 1]))
    return total / period


def _mean_sigma(values: Sequence[float], period: int) -> tuple[float, float] | None:
    """Mean and population sigma of the last ``period`` values; ``None`` during warmup."""
    period = int(period)
    if period < 1 or len(values) < period:
        return None
    window = list(values)[-period:]
    mean = sum(window) / period
    var = sum((v - mean) ** 2 for v in window) / period
    return mean, var**0.5


def stdev(values: Sequence[float], period: int) -> float | None:
    """Population standard deviation of the last ``period`` values; ``None`` during warmup.

    Population sigma (divide by ``period``), matching Pine ``ta.stdev`` with its default
    biased estimator. A constant window returns ``0.0`` — a real value, not warmup.
    """
    ms = _mean_sigma(values, period)
    return None if ms is None else ms[1]


def zscore(values: Sequence[float], period: int) -> float | None:
    """Z-score of the latest value against the last ``period`` values' mean/sigma.

    ``(values[-1] − sma) / stdev`` with population sigma. ``None`` during warmup **and**
    when the window deviation is zero (flat tape) — guard with ``is not None`` instead of
    special-casing division by zero.
    """
    ms = _mean_sigma(values, period)
    if ms is None or ms[1] == 0.0:
        return None
    return (values[-1] - ms[0]) / ms[1]


def bollinger(
    values: Sequence[float], period: int, mult: float = 2.0
) -> tuple[float, float, float] | None:
    """Bollinger bands ``(upper, mid, lower)`` over the last ``period`` values.

    ``mid`` is the SMA, bands sit ``mult`` population sigmas away (Pine ``ta.bb``).
    ``None`` during warmup; a flat window collapses all three onto ``mid``.
    """
    ms = _mean_sigma(values, period)
    if ms is None:
        return None
    mid, sigma = ms
    return (mid + mult * sigma, mid, mid - mult * sigma)


def roc(values: Sequence[float], period: int) -> float | None:
    """Percent rate-of-change vs ``period`` bars ago (Pine ``ta.roc``).

    ``100 × (values[-1] − values[-1-period]) / values[-1-period]``. Needs ``period + 1``
    values; ``None`` during warmup and when the base value is ``0``. At ``period=1`` this
    is the one-bar percent change — the role ``ta.change`` plays in entry conditions (for
    the absolute difference, subtract directly).
    """
    period = int(period)
    if period < 1 or len(values) < period + 1:
        return None
    base = values[-1 - period]
    if base == 0.0:
        return None
    return (values[-1] - base) / base * 100.0


def wma(values: Sequence[float], period: int) -> float | None:
    """Linear-weighted mean of the last ``period`` values (Pine ``ta.wma``).

    Weights ``1..period`` with the most recent value heaviest; ``None`` during warmup.
    """
    period = int(period)
    if period < 1 or len(values) < period:
        return None
    window = list(values)[-period:]
    num = sum(v * (i + 1) for i, v in enumerate(window))
    return num / (period * (period + 1) / 2)


def highest(values: Sequence[float], period: int) -> float | None:
    """Max of the last ``period`` values; ``None`` until enough history."""
    period = int(period)
    if period < 1 or len(values) < period:
        return None
    return max(list(values)[-period:])


def lowest(values: Sequence[float], period: int) -> float | None:
    """Min of the last ``period`` values; ``None`` until enough history."""
    period = int(period)
    if period < 1 or len(values) < period:
        return None
    return min(list(values)[-period:])


def stoch_k(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> float | None:
    """Raw stochastic %K over the last ``period`` bars (Pine ``ta.stoch``), 0–100.

    ``100 × (close − lowest low) / (highest high − lowest low)``. Smooth to %D with
    :func:`sma` over your own %K history. ``None`` during warmup and on a flat window
    (highest high == lowest low).
    """
    period = int(period)
    n = min(len(highs), len(lows), len(closes))
    if period < 1 or n < period:
        return None
    hh = max(list(highs)[-period:])
    ll = min(list(lows)[-period:])
    if hh == ll:
        return None
    return 100.0 * (closes[-1] - ll) / (hh - ll)


def cci(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> float | None:
    """Commodity channel index over typical prices, standard 0.015 constant (Pine ``ta.cci``).

    ``(tp − sma(tp)) / (0.015 × mean deviation)`` with ``tp = (high + low + close) / 3``.
    ``None`` during warmup and when the window's mean deviation is zero (flat tape).
    """
    period = int(period)
    n = min(len(highs), len(lows), len(closes))
    if period < 1 or n < period:
        return None
    hs, ls, cs = list(highs)[-period:], list(lows)[-period:], list(closes)[-period:]
    tps = [(h + low + c) / 3.0 for h, low, c in zip(hs, ls, cs, strict=True)]
    mean = sum(tps) / period
    mad = sum(abs(tp - mean) for tp in tps) / period
    if mad == 0.0:
        return None
    return (tps[-1] - mean) / (0.015 * mad)


def bars_since(flags: Sequence[bool]) -> int | None:
    """Bars since the latest ``True`` in a self-kept flag sequence (Pine ``ta.barssince``).

    ``0`` when the flag is true on the current (last) entry; ``None`` if it was never true.
    Keep the flags in a ``deque`` you append to each bar, like any other history.
    """
    for i in range(len(flags) - 1, -1, -1):
        if flags[i]:
            return len(flags) - 1 - i
    return None


def cross_above(fast_prev: float, fast_now: float, slow_prev: float, slow_now: float) -> bool:
    """True on the bar where ``fast`` crosses from at-or-below ``slow`` to above it."""
    return fast_prev <= slow_prev and fast_now > slow_now


def cross_below(fast_prev: float, fast_now: float, slow_prev: float, slow_now: float) -> bool:
    """True on the bar where ``fast`` crosses from at-or-above ``slow`` to below it.

    Exact mirror of :func:`cross_above` (Pine ``ta.crossunder``).
    """
    return fast_prev >= slow_prev and fast_now < slow_now
