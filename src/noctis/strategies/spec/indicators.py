"""The primitive indicator library — one vectorised ``vector`` fn and one incremental ``State``
per primitive, ported from grid-mng ``client/src/pages/strategy/indicators/*`` + ``nodes/``.

The two forms are golden-tested to agree (``indicator_golden.json`` + random walks), which is
what lets a spec's ``signals()`` and ``on_bar()`` code paths agree by construction: they read
the *same* indicator values, one as a whole ``pd.Series``, the other as a per-bar scalar.

Conventions carried verbatim from grid-mng (do not "simplify" to a pandas one-liner — the
warmup + seeding differ):
  * **EMA** is SMA-seeded at index ``period-1`` (NOT ``ewm(adjust=False)`` from index 0).
  * **RSI/ATR** use Wilder smoothing; first value at index ``period``.
  * **MACD** signal line seeds from the first ``signalPeriod`` *valid* macd values only.
  * **rollingExtreme** default window is the PRIOR ``period`` bars (excludes the current bar).
  * **z-score** window is ``[t-N, t-1]``, population std, warmup/zero-variance → null / false.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from noctis.strategies.base import Bar

NAN = float("nan")
_EPS_RSI = 1e-10
_NS_PER_DAY = 86_400 * 1_000_000_000


def _isnan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _day_ordinal(ts_event: int) -> int:
    """UTC-day bucket for a UTC-ns timestamp (VWAP resets per session)."""
    return int(ts_event) // _NS_PER_DAY


def _series(values: np.ndarray, index) -> pd.Series:
    return pd.Series(values, index=index, dtype="float64")


# ─────────────────────────────────────────────────────────────────────────────
# EMA core (shared by ema + macd) — SMA-seeded, matches grid-mng computeEMA.
# ─────────────────────────────────────────────────────────────────────────────
def _ema_array(vals: np.ndarray, period: int) -> np.ndarray:
    n = len(vals)
    out = np.full(n, np.nan)
    if n < period or period < 1:
        return out
    mult = 2.0 / (period + 1.0)
    ema = float(vals[:period].mean())
    out[period - 1] = ema
    for i in range(period, n):
        ema = (vals[i] - ema) * mult + ema
        out[i] = ema
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised primitives (frame → series / dict)
# ─────────────────────────────────────────────────────────────────────────────
def sma_vector(frame: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    close = frame["close"].astype("float64").reset_index(drop=True)
    return close.rolling(period).mean()


def ema_vector(frame: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    close = frame["close"].astype("float64").to_numpy()
    return _series(_ema_array(close, period), None)


def rsi_vector(frame: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    vals = frame["close"].astype("float64").to_numpy()
    n = len(vals)
    out = np.full(n, np.nan)
    if n <= period:
        return _series(out, None)
    avg_gain = avg_loss = 0.0
    for i in range(1, period + 1):
        d = vals[i] - vals[i - 1]
        if d >= 0:
            avg_gain += d
        else:
            avg_loss -= d
    avg_gain /= period
    avg_loss /= period
    out[period] = 100.0 if avg_loss < _EPS_RSI else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        d = vals[i] - vals[i - 1]
        gain = d if d >= 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = 100.0 if avg_loss < _EPS_RSI else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return _series(out, None)


def atr_vector(frame: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    high = frame["high"].astype("float64").to_numpy()
    low = frame["low"].astype("float64").to_numpy()
    close = frame["close"].astype("float64").to_numpy()
    n = len(close)
    out = np.full(n, np.nan)
    if n <= period:
        return _series(out, None)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = float(tr[1 : period + 1].sum()) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + tr[i]) / period
        out[i] = atr
    return _series(out, None)


def adx_vector(frame: pd.DataFrame, period: int) -> dict[str, pd.Series]:
    """Wilder ADX with ``plus_di``/``minus_di`` ports; twin of :class:`AdxState`."""
    period = int(period)
    high = frame["high"].astype("float64").to_numpy()
    low = frame["low"].astype("float64").to_numpy()
    close = frame["close"].astype("float64").to_numpy()
    n = len(close)
    adx = np.full(n, np.nan)
    pdi = np.full(n, np.nan)
    mdi = np.full(n, np.nan)
    if period < 1 or n <= period:
        return {
            "adx": _series(adx, None),
            "plus_di": _series(pdi, None),
            "minus_di": _series(mdi, None),
        }
    tr_s = pdm_s = mdm_s = 0.0
    dx_sum = 0.0
    cur_adx = NAN
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        pdm = up if (up > down and up > 0.0) else 0.0
        mdm = down if (down > up and down > 0.0) else 0.0
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        if i <= period:
            tr_s += tr
            pdm_s += pdm
            mdm_s += mdm
            if i < period:
                continue
            tr_s /= period
            pdm_s /= period
            mdm_s /= period
        else:
            tr_s = (tr_s * (period - 1) + tr) / period
            pdm_s = (pdm_s * (period - 1) + pdm) / period
            mdm_s = (mdm_s * (period - 1) + mdm) / period
        if tr_s == 0.0:
            pdi[i] = mdi[i] = 0.0
        else:
            pdi[i] = 100.0 * pdm_s / tr_s
            mdi[i] = 100.0 * mdm_s / tr_s
        di_sum = pdi[i] + mdi[i]
        dx = 0.0 if di_sum == 0.0 else 100.0 * abs(pdi[i] - mdi[i]) / di_sum
        k = i - period + 1  # how many DX values so far
        if k <= period:
            dx_sum += dx
            if k < period:
                continue
            cur_adx = dx_sum / period
        else:
            cur_adx = (cur_adx * (period - 1) + dx) / period
        adx[i] = cur_adx
    return {
        "adx": _series(adx, None),
        "plus_di": _series(pdi, None),
        "minus_di": _series(mdi, None),
    }


def obv_vector(frame: pd.DataFrame) -> pd.Series:
    """On-balance volume; twin of :class:`ObvState`. Anchored at 0 on the first bar."""
    close = frame["close"].astype("float64").to_numpy()
    vol = frame["volume"].astype("float64").to_numpy()
    n = len(close)
    out = np.full(n, np.nan)
    obv = 0.0
    for i in range(n):
        if i > 0:
            if close[i] > close[i - 1]:
                obv += vol[i]
            elif close[i] < close[i - 1]:
                obv -= vol[i]
        out[i] = obv
    return _series(out, None)


def stoch_vector(
    frame: pd.DataFrame, period: int, k_smooth: int = 3, d_smooth: int = 3
) -> dict[str, pd.Series]:
    """Smoothed stochastic ``%K``/``%D`` ports; twin of :class:`StochState`.

    Raw %K over ``period`` bars (flat window → 50.0, keeping the smoothing chain total),
    ``%K`` = SMA(raw, ``k_smooth``), ``%D`` = SMA(%K, ``d_smooth``) — the classic slow
    stochastic when called as ``(14, 3, 3)``.
    """
    period, k_smooth, d_smooth = int(period), int(k_smooth), int(d_smooth)
    high = frame["high"].astype("float64").to_numpy()
    low = frame["low"].astype("float64").to_numpy()
    close = frame["close"].astype("float64").to_numpy()
    n = len(close)
    k_out = np.full(n, np.nan)
    d_out = np.full(n, np.nan)
    if period < 1 or k_smooth < 1 or d_smooth < 1:
        return {"k": _series(k_out, None), "d": _series(d_out, None)}
    raw = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = float(high[i - period + 1 : i + 1].max())
        ll = float(low[i - period + 1 : i + 1].min())
        raw[i] = 50.0 if hh == ll else 100.0 * (close[i] - ll) / (hh - ll)
    first_k = period - 1 + k_smooth - 1
    for i in range(first_k, n):
        k_out[i] = float(raw[i - k_smooth + 1 : i + 1].mean())
    first_d = first_k + d_smooth - 1
    for i in range(first_d, n):
        d_out[i] = float(k_out[i - d_smooth + 1 : i + 1].mean())
    return {"k": _series(k_out, None), "d": _series(d_out, None)}


def supertrend_vector(frame: pd.DataFrame, period: int, mult: float = 3.0) -> dict[str, pd.Series]:
    """Supertrend ``st``/``dir`` ports; twin of :class:`SupertrendState`."""
    period, mult = int(period), float(mult)
    high = frame["high"].astype("float64").to_numpy()
    low = frame["low"].astype("float64").to_numpy()
    close = frame["close"].astype("float64").to_numpy()
    atr = atr_vector(frame, period).to_numpy()
    n = len(close)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    fub = flb = NAN
    cur = 0.0
    for i in range(n):
        if math.isnan(atr[i]):
            continue
        mid = (high[i] + low[i]) / 2.0
        bub = mid + mult * atr[i]
        blb = mid - mult * atr[i]
        if cur == 0.0:
            fub, flb = bub, blb
            cur = 1.0 if close[i] > fub else -1.0
        else:
            if bub < fub or close[i - 1] > fub:
                fub = bub
            if blb > flb or close[i - 1] < flb:
                flb = blb
            if cur < 0:
                cur = 1.0 if close[i] > fub else -1.0
            else:
                cur = -1.0 if close[i] < flb else 1.0
        st[i] = fub if cur < 0 else flb
        direction[i] = cur
    return {"st": _series(st, None), "dir": _series(direction, None)}


def vwap_vector(frame: pd.DataFrame, period: int | None = None) -> pd.Series:
    """Session VWAP (resets per UTC day). ``period`` is accepted for API symmetry but unused."""
    close = frame["close"].astype("float64").to_numpy()
    vol = frame["volume"].astype("float64").to_numpy()
    ts = (
        frame["ts_event"].astype("int64").to_numpy()
        if "ts_event" in frame.columns
        else np.zeros(len(close), dtype="int64")
    )
    n = len(close)
    out = np.full(n, np.nan)
    pv = v = 0.0
    cur_day = None
    for i in range(n):
        day = _day_ordinal(ts[i])
        if day != cur_day:
            pv = v = 0.0
            cur_day = day
        # The golden reference engine weights by CLOSE (not typical price); match it.
        price = close[i]
        pv += price * vol[i]
        v += vol[i]
        out[i] = price if v == 0 else pv / v
    return _series(out, None)


def macd_vector(
    frame: pd.DataFrame, fastPeriod: int, slowPeriod: int, signalPeriod: int
) -> dict[str, pd.Series]:
    fast, slow, sig_p = int(fastPeriod), int(slowPeriod), int(signalPeriod)
    close = frame["close"].astype("float64").to_numpy()
    n = len(close)
    fast_ema = _ema_array(close, fast)
    slow_ema = _ema_array(close, slow)
    macd = np.full(n, np.nan)
    for i in range(n):
        if not (math.isnan(fast_ema[i]) or math.isnan(slow_ema[i])):
            macd[i] = fast_ema[i] - slow_ema[i]

    first_valid = slow - 1
    valid_macd = [macd[i] for i in range(first_valid, n) if not math.isnan(macd[i])]
    sig_compact = _ema_array(np.asarray(valid_macd, dtype=float), sig_p)
    signal = np.full(n, np.nan)
    for j in range(len(sig_compact)):
        signal[first_valid + j] = sig_compact[j]

    out_signal = np.full(n, np.nan)
    out_hist = np.full(n, np.nan)
    for i in range(n):
        if math.isnan(macd[i]) or i < slow + sig_p - 2:
            continue
        out_signal[i] = signal[i]
        if not math.isnan(signal[i]):
            out_hist[i] = macd[i] - signal[i]
    return {
        "macd": _series(macd, None),
        "signal": _series(out_signal, None),
        "histogram": _series(out_hist, None),
    }


def rolling_extreme_vector(
    frame: pd.DataFrame,
    mode: str,
    period: int,
    field: str | None = None,
    excludeCurrent: bool = True,
) -> pd.Series:
    period = int(period)
    key = field or ("high" if mode == "max" else "low")
    vals = frame[key].astype("float64").to_numpy()
    n = len(vals)
    out = np.full(n, np.nan)
    if period < 1:
        return _series(out, None)
    for i in range(n):
        end = i - 1 if excludeCurrent else i
        start = end - period + 1
        if start < 0:
            continue
        window = vals[start : end + 1]
        out[i] = float(window.max()) if mode == "max" else float(window.min())
    return _series(out, None)


def zscore_vector(
    series_in: pd.Series,
    lookback: int,
    upperThreshold: float,
    lowerThreshold: float,
    epsilon: float = 1e-8,
) -> dict[str, pd.Series]:
    lookback = int(lookback)
    vals = series_in.astype("float64").to_numpy()
    n = len(vals)
    z = np.full(n, np.nan)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    above = np.zeros(n)
    below = np.zeros(n)
    for t in range(n):
        x = vals[t]
        if math.isnan(x):
            continue
        window = [vals[i] for i in range(max(0, t - lookback), t) if not math.isnan(vals[i])]
        if len(window) < lookback:
            continue
        m = sum(window) / lookback
        s = math.sqrt(sum((v - m) ** 2 for v in window) / lookback)
        mean[t] = m
        std[t] = s
        if s <= epsilon:
            z[t] = 0.0
            continue
        zz = (x - m) / s
        z[t] = zz
        above[t] = 1.0 if zz > upperThreshold else 0.0
        below[t] = 1.0 if zz < lowerThreshold else 0.0
    return {
        "zscore": _series(z, None),
        "mean": _series(mean, None),
        "std": _series(std, None),
        "above": _series(above, None),
        "below": _series(below, None),
    }


def series_op_vector(
    a: pd.Series, b: pd.Series | None, scalar: float | None, op: str, epsilon: float = 1e-8
) -> pd.Series:
    av = a.astype("float64").to_numpy()
    n = len(av)
    bv = b.astype("float64").to_numpy() if b is not None else np.full(n, float(scalar or 0.0))
    out = np.full(n, np.nan)
    for i in range(n):
        x, y = av[i], bv[i]
        if math.isnan(x) or math.isnan(y):
            continue
        if op == "add":
            out[i] = x + y
        elif op == "sub":
            out[i] = x - y
        elif op == "mul":
            out[i] = x * y
        elif op == "div":
            denom = y if abs(y) >= epsilon else (epsilon if y >= 0 else -epsilon)
            out[i] = x / denom
    return _series(out, None)


# ─────────────────────────────────────────────────────────────────────────────
# Incremental State (one scalar per bar)
# ─────────────────────────────────────────────────────────────────────────────
class SmaState:
    def __init__(self, period: int):
        self.period = int(period)
        self.buf: deque[float] = deque(maxlen=self.period)

    def update(self, bar: Bar) -> float:
        self.buf.append(bar.close)
        if len(self.buf) < self.period:
            return NAN
        return sum(self.buf) / self.period


class EmaState:
    def __init__(self, period: int):
        self.period = int(period)
        self.mult = 2.0 / (self.period + 1.0)
        self.seed: list[float] = []
        self.ema: float | None = None

    def _step(self, value: float) -> float:
        if self.ema is None:
            self.seed.append(value)
            if len(self.seed) < self.period:
                return NAN
            self.ema = sum(self.seed) / self.period
            return self.ema
        self.ema = (value - self.ema) * self.mult + self.ema
        return self.ema

    def update(self, bar: Bar) -> float:
        return self._step(bar.close)


class RsiState:
    def __init__(self, period: int):
        self.period = int(period)
        self.prev: float | None = None
        self.count = 0
        self.gsum = self.lsum = 0.0
        self.avg_gain = self.avg_loss = 0.0

    def update(self, bar: Bar) -> float:
        c = bar.close
        if self.prev is None:
            self.prev = c
            return NAN
        d = c - self.prev
        self.prev = c
        gain = d if d >= 0 else 0.0
        loss = -d if d < 0 else 0.0
        self.count += 1
        if self.count <= self.period:
            self.gsum += gain
            self.lsum += loss
            if self.count < self.period:
                return NAN
            self.avg_gain = self.gsum / self.period
            self.avg_loss = self.lsum / self.period
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
        return (
            100.0
            if self.avg_loss < _EPS_RSI
            else 100.0 - 100.0 / (1.0 + self.avg_gain / self.avg_loss)
        )


class AtrState:
    def __init__(self, period: int):
        self.period = int(period)
        self.prev_close: float | None = None
        self.count = 0
        self.trsum = 0.0
        self.atr = 0.0

    def update(self, bar: Bar) -> float:
        if self.prev_close is None:
            self.prev_close = bar.close
            return NAN
        tr = max(
            bar.high - bar.low,
            abs(bar.high - self.prev_close),
            abs(bar.low - self.prev_close),
        )
        self.prev_close = bar.close
        self.count += 1
        if self.count <= self.period:
            self.trsum += tr
            if self.count < self.period:
                return NAN
            self.atr = self.trsum / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        return self.atr


class AdxState:
    """Wilder ADX with ``+DI``/``−DI`` properties (classic Wilder averages variant).

    TR/+DM/−DM are Wilder-smoothed like :class:`AtrState` (mean-seeded at ``period``
    deltas, then ``(prev·(p−1) + x)/p``). DI is available from bar index ``period``;
    ADX averages the first ``period`` DX values, so its first value lands at bar index
    ``2·period − 1``. ``update`` returns ADX (``nan`` during warmup); a zero DI sum
    (dead-flat tape) yields DX = 0, never a division error.
    """

    def __init__(self, period: int):
        self.period = int(period)
        self.prev_high: float | None = None
        self.prev_low = 0.0
        self.prev_close = 0.0
        self.count = 0
        self.tr_s = self.pdm_s = self.mdm_s = 0.0
        self.plus_di = NAN
        self.minus_di = NAN
        self.dx_count = 0
        self.dx_sum = 0.0
        self.adx = NAN

    def update(self, bar: Bar) -> float:
        if self.prev_high is None:
            self.prev_high, self.prev_low, self.prev_close = bar.high, bar.low, bar.close
            return NAN
        up = bar.high - self.prev_high
        down = self.prev_low - bar.low
        pdm = up if (up > down and up > 0.0) else 0.0
        mdm = down if (down > up and down > 0.0) else 0.0
        tr = max(
            bar.high - bar.low,
            abs(bar.high - self.prev_close),
            abs(bar.low - self.prev_close),
        )
        self.prev_high, self.prev_low, self.prev_close = bar.high, bar.low, bar.close
        self.count += 1
        if self.count <= self.period:
            self.tr_s += tr
            self.pdm_s += pdm
            self.mdm_s += mdm
            if self.count < self.period:
                return NAN
            self.tr_s /= self.period
            self.pdm_s /= self.period
            self.mdm_s /= self.period
        else:
            self.tr_s = (self.tr_s * (self.period - 1) + tr) / self.period
            self.pdm_s = (self.pdm_s * (self.period - 1) + pdm) / self.period
            self.mdm_s = (self.mdm_s * (self.period - 1) + mdm) / self.period
        if self.tr_s == 0.0:
            self.plus_di = self.minus_di = 0.0
        else:
            self.plus_di = 100.0 * self.pdm_s / self.tr_s
            self.minus_di = 100.0 * self.mdm_s / self.tr_s
        di_sum = self.plus_di + self.minus_di
        dx = 0.0 if di_sum == 0.0 else 100.0 * abs(self.plus_di - self.minus_di) / di_sum
        self.dx_count += 1
        if self.dx_count <= self.period:
            self.dx_sum += dx
            if self.dx_count < self.period:
                return NAN
            self.adx = self.dx_sum / self.period
        else:
            self.adx = (self.adx * (self.period - 1) + dx) / self.period
        return self.adx


class ObvState:
    """On-balance volume — cumulative signed volume, so state is unavoidable.

    Anchored at ``0.0`` on the first bar (no warmup ``nan``: OBV is defined immediately);
    an up-close adds the bar's volume, a down-close subtracts it, a flat close adds nothing.
    Matches Pine ``ta.obv`` up to its anchor convention.
    """

    def __init__(self) -> None:
        self.prev_close: float | None = None
        self.obv = 0.0

    def update(self, bar: Bar) -> float:
        if self.prev_close is not None:
            if bar.close > self.prev_close:
                self.obv += bar.volume
            elif bar.close < self.prev_close:
                self.obv -= bar.volume
        self.prev_close = bar.close
        return self.obv


class StochState:
    """Smoothed stochastic ``%K``/%D`` (two SMA stages over raw %K); twin of ``stoch_vector``.

    ``update`` returns ``{"k": …, "d": …}`` with ``nan`` during each stage's warmup: raw %K
    needs ``period`` bars, %K lands ``k_smooth − 1`` bars later, %D another ``d_smooth − 1``.
    A flat raw window (highest high == lowest low) contributes 50.0, keeping the chain total.
    """

    def __init__(self, period: int, k_smooth: int = 3, d_smooth: int = 3):
        self.period = int(period)
        self.highs: deque[float] = deque(maxlen=self.period)
        self.lows: deque[float] = deque(maxlen=self.period)
        self.raw_buf: deque[float] = deque(maxlen=int(k_smooth))
        self.k_buf: deque[float] = deque(maxlen=int(d_smooth))

    def update(self, bar: Bar) -> dict[str, float]:
        self.highs.append(bar.high)
        self.lows.append(bar.low)
        if len(self.highs) < self.period:
            return {"k": NAN, "d": NAN}
        hh, ll = max(self.highs), min(self.lows)
        raw = 50.0 if hh == ll else 100.0 * (bar.close - ll) / (hh - ll)
        self.raw_buf.append(raw)
        if len(self.raw_buf) < self.raw_buf.maxlen:  # type: ignore[operator]
            return {"k": NAN, "d": NAN}
        k = sum(self.raw_buf) / len(self.raw_buf)
        self.k_buf.append(k)
        if len(self.k_buf) < self.k_buf.maxlen:  # type: ignore[operator]
            return {"k": k, "d": NAN}
        return {"k": k, "d": sum(self.k_buf) / len(self.k_buf)}


class SupertrendState:
    """Supertrend — Wilder-ATR bands with the classic final-band ratchet; path-dependent.

    ``update`` returns ``{"st": line, "dir": ±1.0}`` (``nan`` until ATR warms up at bar
    ``period``). Basic bands are ``hl2 ± mult·ATR``; the *final* upper band only moves
    down (resets when the prior close breaks above it), the final lower band only moves
    up, and the line flips between them when the close crosses the active band — the
    standard Supertrend recursion. Seeded on the first ATR bar: uptrend only if the
    close is already above the upper band.
    """

    def __init__(self, period: int, mult: float = 3.0):
        self.mult = float(mult)
        self.atr = AtrState(int(period))
        self.prev_close: float | None = None
        self.fub = self.flb = NAN
        self.dir = 0.0  # 0 while warming up, then ±1

    def update(self, bar: Bar) -> dict[str, float]:
        prev_close = self.prev_close
        self.prev_close = bar.close
        atr = self.atr.update(bar)
        if math.isnan(atr):
            return {"st": NAN, "dir": NAN}
        mid = (bar.high + bar.low) / 2.0
        bub = mid + self.mult * atr
        blb = mid - self.mult * atr
        if self.dir == 0.0:  # first ATR bar: bands start at the basic bands
            self.fub, self.flb = bub, blb
            self.dir = 1.0 if bar.close > self.fub else -1.0
        else:
            assert prev_close is not None
            if bub < self.fub or prev_close > self.fub:
                self.fub = bub
            if blb > self.flb or prev_close < self.flb:
                self.flb = blb
            if self.dir < 0:
                self.dir = 1.0 if bar.close > self.fub else -1.0
            else:
                self.dir = -1.0 if bar.close < self.flb else 1.0
        return {"st": self.fub if self.dir < 0 else self.flb, "dir": self.dir}


class VwapState:
    def __init__(self, period: int | None = None):
        self.pv = self.v = 0.0
        self.cur_day: int | None = None

    def update(self, bar: Bar) -> float:
        day = _day_ordinal(bar.ts_event)
        if day != self.cur_day:
            self.pv = self.v = 0.0
            self.cur_day = day
        # Close-weighted to match the golden reference engine.
        self.pv += bar.close * bar.volume
        self.v += bar.volume
        return bar.close if self.v == 0 else self.pv / self.v


class MacdState:
    def __init__(self, fastPeriod: int, slowPeriod: int, signalPeriod: int):
        self.fast = EmaState(int(fastPeriod))
        self.slow = EmaState(int(slowPeriod))
        self.sig = EmaState(int(signalPeriod))
        self.slow_p = int(slowPeriod)
        self.sig_p = int(signalPeriod)
        self.i = -1

    def update(self, bar: Bar) -> dict[str, float]:
        self.i += 1
        f = self.fast.update(bar)
        s = self.slow.update(bar)
        macd = NAN if (_isnan(f) or _isnan(s)) else f - s
        signal = NAN
        hist = NAN
        if not _isnan(macd):
            signal = self.sig._step(macd)
        if self.i < self.slow_p + self.sig_p - 2:
            signal = NAN
        if not _isnan(macd) and not _isnan(signal):
            hist = macd - signal
        return {"macd": macd, "signal": signal, "histogram": hist}


class RollingExtremeState:
    def __init__(
        self, mode: str, period: int, field: str | None = None, excludeCurrent: bool = True
    ):
        self.mode = mode
        self.period = int(period)
        self.field = field or ("high" if mode == "max" else "low")
        self.exclude = excludeCurrent
        self.buf: deque[float] = deque(maxlen=self.period)

    def update(self, bar: Bar) -> float:
        v = getattr(bar, self.field)
        if self.exclude:
            res = NAN if len(self.buf) < self.period else self._extreme()
            self.buf.append(v)
            return res
        self.buf.append(v)
        return NAN if len(self.buf) < self.period else self._extreme()

    def _extreme(self) -> float:
        return max(self.buf) if self.mode == "max" else min(self.buf)


class ZScoreState:
    def __init__(
        self,
        lookback: int,
        upperThreshold: float,
        lowerThreshold: float,
        epsilon: float = 1e-8,
    ):
        self.lookback = int(lookback)
        self.upper = float(upperThreshold)
        self.lower = float(lowerThreshold)
        self.eps = float(epsilon)
        self.buf: deque[float] = deque(maxlen=self.lookback)

    _NULL = {"zscore": NAN, "mean": NAN, "std": NAN, "above": 0.0, "below": 0.0}

    def update(self, x: float) -> dict[str, float]:
        if _isnan(x):
            return dict(self._NULL)
        if len(self.buf) < self.lookback:
            self.buf.append(x)
            return dict(self._NULL)
        window = list(self.buf)
        m = sum(window) / self.lookback
        s = math.sqrt(sum((v - m) ** 2 for v in window) / self.lookback)
        self.buf.append(x)
        if s <= self.eps:
            return {"zscore": 0.0, "mean": m, "std": s, "above": 0.0, "below": 0.0}
        z = (x - m) / s
        return {
            "zscore": z,
            "mean": m,
            "std": s,
            "above": 1.0 if z > self.upper else 0.0,
            "below": 1.0 if z < self.lower else 0.0,
        }


class SeriesOpState:
    def __init__(self, op: str, epsilon: float = 1e-8):
        self.op = op
        self.eps = float(epsilon)

    def update(self, a: float, b: float) -> float:
        if _isnan(a) or _isnan(b):
            return NAN
        if self.op == "add":
            return a + b
        if self.op == "sub":
            return a - b
        if self.op == "mul":
            return a * b
        denom = b if abs(b) >= self.eps else (self.eps if b >= 0 else -self.eps)
        return a / denom


# ─────────────────────────────────────────────────────────────────────────────
# Registry of the golden-tested bar primitives: kind → (vector_fn, State_cls).
#
# Deliberately NOT widened with the signal-surface additions (adx/obv/stoch, the
# author-facing tail functions, …): this registry is the legacy spec-DSL's surface,
# and growing a legacy graph language is a separate decision from growing the
# author library. New primitives stay author-surface-only until that call is made.
# ─────────────────────────────────────────────────────────────────────────────
REGISTRY: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "sma": (sma_vector, SmaState),
    "ema": (ema_vector, EmaState),
    "rsi": (rsi_vector, RsiState),
    "atr": (atr_vector, AtrState),
    "vwap": (vwap_vector, VwapState),
    "macd": (macd_vector, MacdState),
    "rollingExtreme": (rolling_extreme_vector, RollingExtremeState),
}
