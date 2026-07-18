"""The structural symbol screener — features from bars, pool-relative bands, and the
determinism + no-holdout-leak guarantees the cross-sectional guardrail depends on."""

from __future__ import annotations

import pandas as pd
import pytest

from noctis.research.symbols import (
    SymbolFeatures,
    SymbolScreener,
    compute_features,
    screen,
    validate_profile,
)

_NS_PER_MINUTE = 60 * 1_000_000_000


def tape(close: list[float], volume: float = 1000) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "ts_event": [i * _NS_PER_MINUTE for i in range(n)],
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [volume] * n,
        }
    )


def features(**overrides) -> SymbolFeatures:
    base = dict(symbol="SYM", timeframe="1d", bars=100, trend=0.5, volatility=0.25, liquidity=1e8)
    base.update(overrides)
    return SymbolFeatures(**base)


# ── compute_features ────────────────────────────────────────────────────────────────────
def test_trend_efficiency_separates_trending_from_chop():
    trending = compute_features("UP", tape([100 + i * 0.1 for i in range(320)]))
    choppy = compute_features("CHOP", tape([100 + (i % 2) for i in range(320)]))
    assert trending is not None and choppy is not None
    assert trending.trend == pytest.approx(1.0)
    assert choppy.trend < 0.1
    # A one-tick zigzag realizes far more per-bar variance than a smooth drift.
    assert choppy.volatility > trending.volatility


def test_liquidity_scales_with_volume():
    close = [100 + (i % 5) * 0.1 for i in range(320)]
    thin = compute_features("THIN", tape(close, volume=1000))
    thick = compute_features("THICK", tape(close, volume=2000))
    assert thin is not None and thick is not None
    assert thick.liquidity == pytest.approx(2.0 * thin.liquidity)


def test_too_short_training_window_yields_none():
    assert compute_features("S", tape([100.0 + i * 0.01 for i in range(30)])) is None


def test_features_never_read_the_holdout_tail():
    """Mutating ONLY the reserved forward-holdout tail must not move a single feature —
    symbol selection shaped by holdout structure would be cross-sectional lookahead."""
    n_minutes = 126 * 60  # 126 hourly bars: too few daily bars, so "1h" is chosen
    close = [100 + (i % 7) * 0.2 for i in range(n_minutes)]
    base = compute_features("X", tape(close))
    assert base is not None and base.timeframe == "1h"
    total_hours = n_minutes // 60
    holdout_hours = total_hours - base.bars
    assert holdout_hours > 0  # the geometry actually reserved a tail here
    wild = list(close)
    for i in range((total_hours - holdout_hours) * 60, n_minutes):
        wild[i] = 500.0 + (i % 3) * 40.0  # a regime the training window never saw
    assert compute_features("X", tape(wild)) == base


# ── screen ──────────────────────────────────────────────────────────────────────────────
def _pool():
    # Six names with strictly increasing trend; vol/liquidity vary independently.
    return [
        features(
            symbol=f"S{i}", trend=0.1 + 0.15 * i, volatility=0.1 + 0.05 * i, liquidity=1e8 * (6 - i)
        )
        for i in range(6)
    ]


def test_screen_high_band_matches_top_tercile_ranked_by_strength():
    result = screen(_pool(), {"trend": "high"})
    assert [m.features.symbol for m in result.matched] == ["S5", "S4"]
    assert result.matched[0].strength > result.matched[1].strength
    assert set(result.rejected) == {"S0", "S1", "S2", "S3"}
    assert result.rejected["S0"]["trend"] == "low"
    cuts = result.cutoffs["trend"]
    assert cuts["low_max"] < cuts["high_min"]


def test_screen_low_and_medium_bands():
    low = screen(_pool(), {"volatility": "low"})
    assert [m.features.symbol for m in low.matched] == ["S0", "S1"]
    medium = screen(_pool(), {"trend": "medium"})
    assert {m.features.symbol for m in medium.matched} == {"S2", "S3"}
    for m in medium.matched:
        assert m.bands["trend"] == "medium"


def test_screen_all_any_ranks_by_liquidity():
    result = screen(_pool(), {})
    assert [m.features.symbol for m in result.matched] == [f"S{i}" for i in range(6)]
    assert result.rejected == {}


def test_screen_combined_dimensions_and_determinism():
    profile = {"trend": "high", "liquidity": "low"}
    a = screen(_pool(), profile)
    b = screen(reversed(_pool()), profile)  # input order must not matter
    assert [m.features.symbol for m in a.matched] == [m.features.symbol for m in b.matched]
    assert a.cutoffs == b.cutoffs
    for m in a.matched:
        assert m.bands["trend"] == "high" and m.bands["liquidity"] == "low"


def test_screen_empty_pool():
    result = screen([], {"trend": "high"})
    assert result.matched == [] and result.rejected == {} and result.cutoffs == {}


def test_validate_profile_rejects_unknowns_and_normalizes_case():
    assert validate_profile({"trend": "High"})["trend"] == "high"
    assert validate_profile({})["volatility"] == "any"
    with pytest.raises(ValueError, match="unknown profile dimensions"):
        validate_profile({"sector": "tech"})
    with pytest.raises(ValueError, match="volatility must be one of"):
        validate_profile({"volatility": "extreme"})


# ── SymbolScreener cache ────────────────────────────────────────────────────────────────
class _CountingLake:
    def __init__(self, bars_by_symbol):
        self.bars = bars_by_symbol
        self.get_calls = 0

    def check_symbol_ready(self, symbol, dataset=None, schema=None):
        return symbol in self.bars

    def get_bars(self, dataset, schema, symbols, start, end):
        self.get_calls += 1
        return {s: self.bars[s] for s in symbols if s in self.bars}


def test_screener_caches_by_fingerprint_and_recomputes_on_growth():
    frame = tape([100 + i * 0.1 for i in range(320)])
    lake = _CountingLake({"UP": frame})
    screener = SymbolScreener(lake, "DATASET", "ohlcv-1m")
    first = screener.features(["UP", "MISSING"])["UP"]
    assert screener.features(["UP"])["UP"] is first  # same fingerprint → cached object
    grown = tape([100 + i * 0.1 for i in range(400)])
    lake.bars["UP"] = grown
    assert screener.features(["UP"])["UP"] is not first  # ensure_data growth recomputes
    assert "MISSING" not in screener.features(["MISSING"])  # unready names are skipped
