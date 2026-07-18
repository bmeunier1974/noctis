"""Tail-function indicator contract: warmup ``None``, exact math, golden agreement.

Covers the author-facing tail functions in ``noctis.strategies.indicators`` added by the
signal-surface epic. Expected fixture values are computed offline with a pandas reference
engine and pasted into ``tests/fixtures/indicator_golden.json`` (same regime as the
existing six primitives).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from noctis.strategies import indicators as ind

_GOLDEN = Path(__file__).parent / "fixtures" / "indicator_golden.json"

# --- cross_below: mirror of cross_above ---------------------------------------------------


def test_cross_below_fires_only_on_the_crossing_bar():
    # fast was at-or-above, now strictly below -> True
    assert ind.cross_below(10.0, 8.0, 9.0, 9.0) is True
    # equality on the previous bar still counts as "was at-or-above"
    assert ind.cross_below(9.0, 8.0, 9.0, 9.0) is True
    # staying above -> False
    assert ind.cross_below(10.0, 9.5, 9.0, 9.0) is False
    # touching but not crossing (fast_now == slow_now) -> False
    assert ind.cross_below(10.0, 9.0, 9.0, 9.0) is False
    # already below and staying below -> False (no re-fire)
    assert ind.cross_below(8.0, 7.0, 9.0, 9.0) is False


# --- stdev: population sigma over the tail window ------------------------------------------


def test_stdev_known_value_and_warmup():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # classic textbook case: population sigma is exactly 2.0
    assert ind.stdev(values, 8) == 2.0
    # warmup: not enough history -> None
    assert ind.stdev(values[:7], 8) is None
    assert ind.stdev([], 3) is None
    # only the last `period` values count
    assert ind.stdev([100.0, *values], 8) == 2.0


def test_stdev_degenerate_windows():
    # a single-value window has zero deviation
    assert ind.stdev([5.0, 7.0], 1) == 0.0
    # constant series -> 0.0, not None (zero sigma is a real value)
    assert ind.stdev([3.0] * 10, 5) == 0.0
    # period < 1 is not a window
    assert ind.stdev([1.0, 2.0], 0) is None
    # float period is coerced like the existing helpers
    assert ind.stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], 8.0) == 2.0


# --- zscore: (last - mean) / sigma, None on a flat window ----------------------------------


def test_zscore_known_value_and_warmup():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # mean 5.0, population sigma 2.0, last 9.0 -> z = 2.0
    assert ind.zscore(values, 8) == 2.0
    assert ind.zscore(values[:7], 8) is None


def test_zscore_flat_window_returns_none_not_zero_division():
    # zero deviation -> None (the documented edge), never a ZeroDivisionError
    assert ind.zscore([3.0] * 10, 5) is None
    assert ind.zscore([1.0, 3.0], 1) is None  # single-value window is always flat


# --- bollinger: (upper, mid, lower) ---------------------------------------------------------


def test_bollinger_known_value_and_warmup():
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    got = ind.bollinger(values, 8, 2.0)
    assert got == (9.0, 5.0, 1.0)  # mean 5, sigma 2, mult 2
    assert ind.bollinger(values[:7], 8, 2.0) is None


def test_bollinger_constant_series_collapses_to_mid():
    got = ind.bollinger([4.0] * 6, 5, 2.0)
    assert got == (4.0, 4.0, 4.0)


# --- roc: percent rate-of-change ------------------------------------------------------------


def test_roc_known_value_and_warmup():
    values = [100.0, 102.0, 105.0, 110.0]
    # vs 3 bars ago: (110 - 100) / 100 * 100 = 10%
    assert ind.roc(values, 3) == 10.0
    # period=1 is the one-bar percent change
    got = ind.roc(values, 1)
    assert got is not None and abs(got - (110.0 - 105.0) / 105.0 * 100.0) < 1e-12
    # needs period + 1 values
    assert ind.roc(values, 4) is None
    assert ind.roc([], 1) is None


def test_roc_zero_base_returns_none():
    assert ind.roc([0.0, 5.0], 1) is None


# --- wma: linear-weighted moving average ----------------------------------------------------


def test_wma_known_value_and_warmup():
    # weights 1,2,3 with the most recent heaviest: (1*1 + 2*2 + 3*3) / 6
    got = ind.wma([1.0, 2.0, 3.0], 3)
    assert got is not None and abs(got - 14.0 / 6.0) < 1e-12
    assert ind.wma([1.0, 2.0], 3) is None
    # period=1 is just the last value
    assert ind.wma([1.0, 2.0, 3.0], 1) == 3.0
    # constant series -> the constant
    assert ind.wma([4.0] * 8, 5) == 4.0


def test_wma_uses_only_the_tail_window():
    assert ind.wma([99.0, 1.0, 2.0, 3.0], 3) == ind.wma([1.0, 2.0, 3.0], 3)


# --- AdxState: Wilder ADX with +DI/−DI, house warmup contract --------------------------------


def _bar(i: int, high: float, low: float, close: float):
    from noctis.strategies.base import Bar

    return Bar(i * 60_000_000_000, close, high, low, close, 1000.0)


def test_adx_state_warmup_and_hand_checked_ramp():
    # steady +1 ramp, period=2: TR=1.5 and +DM=1 every bar after the first,
    # so +DI = 100·(1/1.5) = 66.67, −DI = 0, DX = 100 from the first DI bar on.
    bars = [
        _bar(0, 10.0, 9.0, 9.5),
        _bar(1, 11.0, 10.0, 10.5),
        _bar(2, 12.0, 11.0, 11.5),
        _bar(3, 13.0, 12.0, 12.5),
    ]
    st = ind.AdxState(2)
    out = [st.update(b) for b in bars]
    # ADX needs period DX values: nan through index 2·period − 2, first value at 2·period − 1
    assert all(math.isnan(v) for v in out[:3])
    assert abs(out[3] - 100.0) < 1e-9
    # DI is available one Wilder-window earlier (index period), pinned by the ramp
    assert abs(st.plus_di - 100.0 / 1.5) < 1e-9
    assert st.minus_di == 0.0


def test_adx_state_di_warmup_is_nan():
    st = ind.AdxState(3)
    st.update(_bar(0, 10.0, 9.0, 9.5))
    assert math.isnan(st.plus_di) and math.isnan(st.minus_di)
    assert math.isnan(st.update(_bar(1, 11.0, 10.0, 10.5)))
    assert math.isnan(st.plus_di)  # count=1 < period=3: still warming up


def test_adx_state_flat_tape_is_zero_not_nan_after_warmup():
    # dead-flat bars: TR=0, DM=0 -> DI=0, DX defined as 0 (not a division error)
    st = ind.AdxState(2)
    out = [st.update(_bar(i, 5.0, 5.0, 5.0)) for i in range(6)]
    assert all(math.isnan(v) for v in out[:3])
    assert out[3] == 0.0 and out[5] == 0.0
    assert st.plus_di == 0.0 and st.minus_di == 0.0


# --- ObvState: cumulative on-balance volume ---------------------------------------------------


def _cbar(i: int, close: float, volume: float):
    from noctis.strategies.base import Bar

    return Bar(i * 60_000_000_000, close, close, close, close, volume)


def test_obv_accumulates_signed_volume():
    closes_vols = [(10.0, 100.0), (11.0, 200.0), (11.0, 300.0), (9.0, 400.0)]
    st = ind.ObvState()
    out = [st.update(_cbar(i, c, v)) for i, (c, v) in enumerate(closes_vols)]
    # first bar anchors at 0; up-close adds volume, flat close adds nothing, down subtracts
    assert out == [0.0, 200.0, 200.0, -200.0]


# --- StochState: smoothed %K/%D (two SMA stages over raw %K) ----------------------------------


def test_stoch_state_hand_checked_and_warmup():
    bars = [
        _bar(0, 10.0, 8.0, 9.0),
        _bar(1, 11.0, 9.0, 10.0),
        _bar(2, 12.0, 10.0, 11.0),
        _bar(3, 11.0, 9.0, 10.0),
    ]
    st = ind.StochState(2, 2, 2)
    rows = [st.update(b) for b in bars]
    # raw %K first at index period−1=1, %K (smoothed) at 2, %D one SMA stage later at 3
    assert math.isnan(rows[0]["k"]) and math.isnan(rows[1]["k"])
    assert abs(rows[2]["k"] - 200.0 / 3.0) < 1e-9  # mean(66.67, 66.67)
    assert math.isnan(rows[2]["d"])
    assert abs(rows[3]["k"] - 50.0) < 1e-9  # mean(66.67, 33.33)
    assert abs(rows[3]["d"] - 175.0 / 3.0) < 1e-9  # mean(66.67, 50)


# --- SupertrendState: ATR-band trend flip -----------------------------------------------------


def test_supertrend_hand_checked_flip():
    # period=2, mult=1. Steady +1 drift (ATR=2) starts in downtrend with the upper band
    # ratcheting down; the surge bar (18,16,17) blows through the final upper band 15,
    # flipping to uptrend with the supertrend line on the lower band 14.
    bars = [
        _bar(0, 12.0, 10.0, 11.0),
        _bar(1, 13.0, 11.0, 12.0),
        _bar(2, 14.0, 12.0, 13.0),
        _bar(3, 15.0, 13.0, 14.0),
        _bar(4, 18.0, 16.0, 17.0),
    ]
    st = ind.SupertrendState(2, 1.0)
    rows = [st.update(b) for b in bars]
    assert all(math.isnan(r["st"]) and math.isnan(r["dir"]) for r in rows[:2])
    assert rows[2] == {"st": 15.0, "dir": -1.0}
    assert rows[3] == {"st": 15.0, "dir": -1.0}  # final upper band ratchets, not the basic band
    assert rows[4] == {"st": 14.0, "dir": 1.0}  # the flip


# --- discoverability: the golden-tested states are reachable from the author module ----------


def test_existing_state_classes_are_reexported():
    for name in ("ZScoreState", "RollingExtremeState", "ObvState", "StochState", "AdxState"):
        assert name in ind.__all__ and hasattr(ind, name)


# --- golden agreement: pandas-reference expectations, pasted into the shared fixture --------


def _matches(got: float | tuple | None, expected: float | None, tol: float) -> bool:
    if expected is None:
        return got is None
    return got is not None and abs(float(got) - float(expected)) <= tol


def test_tail_functions_match_golden_fixture():
    gold = json.loads(_GOLDEN.read_text())
    tol = gold["meta"]["tolerance"]
    p = gold["meta"]["params"]
    for sname, series in gold["series"].items():
        closes = [float(c) for c in series["close"]]
        highs = [float(h) for h in series["high"]]
        lows = [float(lo) for lo in series["low"]]
        exp = gold["expected"][sname]
        n = len(closes)
        boll_p, boll_mult = p["boll"]
        for i in range(n):
            tail = closes[: i + 1]
            ht, lt = highs[: i + 1], lows[: i + 1]
            checks = [
                (f"stdev_{p['stdev']}", ind.stdev(tail, p["stdev"])),
                (f"zscore_{p['zscore']}", ind.zscore(tail, p["zscore"])),
                (f"roc_{p['roc']}", ind.roc(tail, p["roc"])),
                (f"wma_{p['wma']}", ind.wma(tail, p["wma"])),
                (f"stoch_{p['stoch']}", ind.stoch_k(ht, lt, tail, p["stoch"])),
                (f"cci_{p['cci']}", ind.cci(ht, lt, tail, p["cci"])),
            ]
            for key, got in checks:
                assert _matches(got, exp[key][i], tol), f"{sname}.{key}[{i}]: {got}"
            bgot = ind.bollinger(tail, boll_p, boll_mult)
            bexp = exp[f"boll_{boll_p}_{int(boll_mult)}"]
            if bexp["mid"][i] is None:
                assert bgot is None, f"{sname}.boll[{i}]: expected warmup, got {bgot}"
            else:
                assert bgot is not None, f"{sname}.boll[{i}]: unexpected warmup"
                for j, port in enumerate(("upper", "mid", "lower")):
                    assert not math.isnan(bgot[j]) and abs(bgot[j] - bexp[port][i]) <= tol, (
                        f"{sname}.boll.{port}[{i}]: {bgot[j]} != {bexp[port][i]}"
                    )


# --- stoch_k: raw %K over highs/lows/closes -------------------------------------------------


def test_stoch_k_known_value_and_warmup():
    highs = [10.0, 12.0, 11.0]
    lows = [8.0, 9.0, 9.5]
    closes = [9.0, 11.0, 10.0]
    # window: hh=12, ll=8, close=10 -> 100*(10-8)/(12-8) = 50
    assert ind.stoch_k(highs, lows, closes, 3) == 50.0
    # close pinned to the extremes
    assert ind.stoch_k(highs, lows, [9.0, 11.0, 12.0], 3) == 100.0
    assert ind.stoch_k(highs, lows, [9.0, 11.0, 8.0], 3) == 0.0
    # warmup: needs `period` bars
    assert ind.stoch_k(highs[:2], lows[:2], closes[:2], 3) is None


def test_stoch_k_flat_range_returns_none():
    # high == low over the whole window -> undefined, never a ZeroDivisionError
    assert ind.stoch_k([5.0] * 4, [5.0] * 4, [5.0] * 4, 4) is None


# --- cci: typical-price commodity channel index ----------------------------------------------


def test_cci_known_value_and_warmup():
    # typical prices 10, 12, 14: sma=12, mean deviation=4/3
    # cci = (14 - 12) / (0.015 * 4/3) = 100 — the textbook "steady trend pins 100" case
    highs = [12.0, 14.0, 16.0]
    lows = [8.0, 10.0, 12.0]
    closes = [10.0, 12.0, 14.0]
    got = ind.cci(highs, lows, closes, 3)
    assert got is not None and abs(got - 100.0) < 1e-12
    assert ind.cci(highs[:2], lows[:2], closes[:2], 3) is None


def test_cci_flat_window_returns_none():
    assert ind.cci([5.0] * 5, [5.0] * 5, [5.0] * 5, 5) is None


# --- bars_since: Pine ta.barssince over a self-kept flag sequence -----------------------------


def test_bars_since_counts_from_latest_true():
    # condition fired on the current bar -> 0
    assert ind.bars_since([False, True]) == 0
    # fired two bars ago
    assert ind.bars_since([True, False, False]) == 2
    # latest occurrence wins
    assert ind.bars_since([True, False, True, False]) == 1


def test_bars_since_never_true_returns_none():
    assert ind.bars_since([]) is None
    assert ind.bars_since([False, False, False]) is None


def test_cross_below_mirrors_cross_above():
    cases = [
        (10.0, 8.0, 9.0, 9.0),
        (9.0, 8.0, 9.0, 9.0),
        (8.0, 10.0, 9.0, 9.0),
        (9.0, 10.0, 9.0, 9.0),
        (8.0, 7.0, 9.0, 9.0),
    ]
    for fp, fn, sp, sn in cases:
        # swapping the fast/slow roles turns a bearish cross into a bullish one
        assert ind.cross_below(fp, fn, sp, sn) == ind.cross_above(sp, sn, fp, fn)
