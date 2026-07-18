"""Structural symbol screener — the thesis picks the KIND of symbol, the data picks the tickers.

Per-symbol character features derived from lake bars (trend efficiency, realized
volatility, dollar-volume liquidity), assigned pool-relative low/medium/high bands and
matched against a requested :data:`profile <PROFILE_DIMENSIONS>`. Everything here is
deterministic given the lake contents, so the same profile over the same data always
names the same tickers — the model chooses characteristics, never symbols.

Two rules keep the screen honest (the cross-sectional guardrail):

* **Structure only, never strategy PnL.** Selecting symbols where a strategy already
  shows profit is cherry-picking — the symbol axis of lookahead. No strategy return
  ever enters a feature; matching a profile is evidence of *character*, not of edge.
* **Training window only.** Features are computed with the forward-holdout tail cut
  off (the same reservation :meth:`preview_bars` honors), so symbol selection cannot
  be shaped by holdout-window structure.

Sector is deliberately absent from v1 features: the lake stores bars only, and sector
membership is general knowledge the agent already has (plus web_search) — the screener's
job is the part the model cannot do honestly, ranking real local data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from noctis.backtest import PipelineConfig
from noctis.data.aggregate import aggregate_bars, bars_per_year

# The profile dimensions a screen can constrain, and the bands a request may ask for.
PROFILE_DIMENSIONS = ("trend", "volatility", "liquidity")
BANDS = ("low", "medium", "high")
ANY = "any"

# Features are computed at the coarsest timeframe whose training window still holds a
# usable sample — a months-deep lake lands on "1d", a shallow one degrades gracefully.
_FEATURE_TIMEFRAMES = ("1d", "1h", "15m", "5m", "1m")
_MIN_FEATURE_BARS = 60
_DAILY_BARS_PER_YEAR = bars_per_year("1d")


@dataclass(frozen=True)
class SymbolFeatures:
    """One symbol's structural character, computed on its training window.

    ``trend`` is Kaufman's efficiency ratio on closes (0 = pure chop, 1 = one-way move);
    ``volatility`` is the annualized standard deviation of per-bar returns;
    ``liquidity`` is the median per-bar dollar volume scaled to a per-day equivalent,
    so symbols screened at different timeframes stay comparable.
    """

    symbol: str
    timeframe: str
    bars: int
    trend: float
    volatility: float
    liquidity: float

    def value(self, dimension: str) -> float:
        return {"trend": self.trend, "volatility": self.volatility, "liquidity": self.liquidity}[
            dimension
        ]


def compute_features(symbol: str, bars_1m: pd.DataFrame) -> SymbolFeatures | None:
    """Features from a symbol's native-granularity lake bars, or ``None`` when the
    training window is too short at every timeframe to say anything about character."""
    for timeframe in _FEATURE_TIMEFRAMES:
        agg = aggregate_bars(bars_1m, timeframe)
        # The same geometry heuristic evaluation uses, at this timeframe: reserve the
        # forward-holdout tail so screening never reads structure the search can't.
        holdout = PipelineConfig.auto(len(agg)).holdout_size
        train = agg.iloc[: len(agg) - holdout] if holdout else agg
        if len(train) < _MIN_FEATURE_BARS:
            continue
        close = train["close"].astype("float64")
        if float(close.iloc[0]) == 0.0:
            return None
        returns = close.pct_change().dropna()
        moves = close.diff().abs().sum()
        trend = float(abs(close.iloc[-1] - close.iloc[0]) / moves) if moves > 0 else 0.0
        volatility = float(returns.std()) * math.sqrt(bars_per_year(timeframe))
        dollar = (close * train["volume"].astype("float64")).median()
        liquidity = float(dollar) * bars_per_year(timeframe) / _DAILY_BARS_PER_YEAR
        return SymbolFeatures(
            symbol=symbol,
            timeframe=timeframe,
            bars=int(len(train)),
            trend=trend,
            volatility=volatility,
            liquidity=liquidity,
        )
    return None


@dataclass(frozen=True)
class ScreenMatch:
    features: SymbolFeatures
    bands: dict[str, str]  # dimension -> low | medium | high (pool-relative)
    strength: float  # how strongly the symbol expresses the requested profile, [0, 1]


@dataclass(frozen=True)
class ScreenResult:
    matched: list[ScreenMatch]  # ranked strongest-first; ties break alphabetically
    rejected: dict[str, dict[str, str]]  # symbol -> its bands, for pool names that missed
    cutoffs: dict[str, dict[str, float]]  # dimension -> {"low_max": ..., "high_min": ...}


def validate_profile(profile: Mapping[str, str]) -> dict[str, str]:
    """Normalize a requested profile; unknown dimensions or bands are hard errors."""
    unknown = sorted(set(profile) - set(PROFILE_DIMENSIONS))
    if unknown:
        raise ValueError(f"unknown profile dimensions {unknown}; supported: {PROFILE_DIMENSIONS}")
    out = {}
    for dim in PROFILE_DIMENSIONS:
        band = str(profile.get(dim, ANY)).strip().lower()
        if band not in (*BANDS, ANY):
            raise ValueError(f"{dim} must be one of {(*BANDS, ANY)}; got {band!r}")
        out[dim] = band
    return out


def screen(features: Iterable[SymbolFeatures], profile: Mapping[str, str]) -> ScreenResult:
    """Match a pool of symbol features against a profile, deterministically.

    Bands are pool-relative terciles per dimension (reported in ``cutoffs`` so a reader
    can see exactly what "high" meant for this pool); they degenerate gracefully on very
    small pools. ``strength`` ranks how strongly each matched name expresses the
    requested bands — an all-``any`` profile ranks by liquidity (most tradable first).
    """
    wanted = validate_profile(profile)
    pool = sorted(features, key=lambda f: f.symbol)
    if not pool:
        return ScreenResult(matched=[], rejected={}, cutoffs={})

    cutoffs: dict[str, dict[str, float]] = {}
    bands: dict[str, dict[str, str]] = {f.symbol: {} for f in pool}
    percentile: dict[str, dict[str, float]] = {f.symbol: {} for f in pool}
    for dim in PROFILE_DIMENSIONS:
        values = pd.Series([f.value(dim) for f in pool], dtype="float64")
        low_max = float(values.quantile(1 / 3))
        high_min = float(values.quantile(2 / 3))
        cutoffs[dim] = {"low_max": low_max, "high_min": high_min}
        ranks = values.rank(method="average")
        for f, rank in zip(pool, ranks, strict=True):
            v = f.value(dim)
            bands[f.symbol][dim] = "high" if v >= high_min else "low" if v <= low_max else "medium"
            percentile[f.symbol][dim] = (rank - 1) / (len(pool) - 1) if len(pool) > 1 else 0.5

    matched: list[ScreenMatch] = []
    rejected: dict[str, dict[str, str]] = {}
    for f in pool:
        if any(wanted[d] not in (ANY, bands[f.symbol][d]) for d in PROFILE_DIMENSIONS):
            rejected[f.symbol] = bands[f.symbol]
            continue
        scores = []
        for dim in PROFILE_DIMENSIONS:
            p = percentile[f.symbol][dim]
            if wanted[dim] == "high":
                scores.append(p)
            elif wanted[dim] == "low":
                scores.append(1.0 - p)
            elif wanted[dim] == "medium":
                scores.append(1.0 - 2.0 * abs(p - 0.5))
        strength = sum(scores) / len(scores) if scores else percentile[f.symbol]["liquidity"]
        matched.append(ScreenMatch(features=f, bands=bands[f.symbol], strength=strength))
    matched.sort(key=lambda m: (-m.strength, m.features.symbol))
    return ScreenResult(matched=matched, rejected=rejected, cutoffs=cutoffs)


class SymbolScreener:
    """Session-scoped feature store over the lake: fetch once, compute once, reuse.

    The cache keys on each symbol's (row count, last timestamp) fingerprint, so an
    ``ensure_data`` that extends a series recomputes its features automatically.
    """

    def __init__(self, lake, dataset: str, schema: str):
        self.lake = lake
        self.dataset = dataset
        self.schema = schema
        self._cache: dict[str, tuple[tuple[int, int], SymbolFeatures | None]] = {}

    def features_from_bars(self, symbol: str, bars_1m: pd.DataFrame) -> SymbolFeatures | None:
        """Cache-aware feature computation for bars a caller already fetched."""
        fingerprint = (len(bars_1m), int(bars_1m["ts_event"].iloc[-1]) if len(bars_1m) else 0)
        hit = self._cache.get(symbol)
        if hit is not None and hit[0] == fingerprint:
            return hit[1]
        features = compute_features(symbol, bars_1m)
        self._cache[symbol] = (fingerprint, features)
        return features

    def features(self, symbols: Iterable[str]) -> dict[str, SymbolFeatures | None]:
        """Features per lake-ready symbol; ``None`` marks a too-short training window."""
        wanted = [s for s in symbols if self.lake.check_symbol_ready(s)]
        if not wanted:
            return {}
        raw = self.lake.get_bars(self.dataset, self.schema, wanted, 0, 2**63 - 1)
        return {s: self.features_from_bars(s, raw[s]) for s in wanted if s in raw}
