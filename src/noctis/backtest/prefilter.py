"""The vectorised pre-filter — cheap screening of many candidates.

Runs each candidate's vectorised ``signals()`` over catalog bars, applies coarse costs, and
ranks by a quick metric to produce a top-K shortlist. It is a **filter, never a promoter**:
promotion metrics come only from the execution-realistic validation stage. A vectorbt
backend can slot in behind this seam later; the default is a NumPy/pandas screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from noctis.backtest.candidate import Candidate
from noctis.backtest.scorecard import (
    DEFAULT_ANNUALIZATION_CAP,
    DEFAULT_MAX_PERIOD_RATIO,
    Metric,
)
from noctis.strategies.families import FamilyRegistry


@dataclass(frozen=True)
class PrefilterConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    metric: Metric = Metric.SHARPE  # coarse ranking metric
    periods_per_year: int = 252
    # Metric-robustness caps (see scorecard._annualization / _cap_ratio); active by default so the
    # coarse rank uses the same bounded units as validation.
    annualization_cap: int | None = DEFAULT_ANNUALIZATION_CAP
    max_period_ratio: float | None = DEFAULT_MAX_PERIOD_RATIO


def vectorized_returns(
    bars: pd.DataFrame, targets: pd.Series, fee_bps: float, slippage_bps: float
) -> pd.Series:
    """Per-bar strategy returns with next-bar execution and coarse per-trade costs.

    The position **held** over bar *i* is the target decided on bar *i−1* (shift by one) —
    this is the same next-bar execution the event engine uses, so the pre-filter is
    lookahead-free too. Costs are charged in proportion to position changes.
    """
    close = bars["close"].astype("float64").reset_index(drop=True)
    tgt = pd.Series(targets).astype("float64").reset_index(drop=True)
    bar_ret = close.pct_change().fillna(0.0)
    held = tgt.shift(1).fillna(0.0)  # executed next bar
    gross = held * bar_ret
    # cost when the executed position changes (turnover), in return terms.
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    trades = held.diff().abs().fillna(held.abs())
    cost = trades * cost_rate
    return (gross - cost).rename("returns")


def coarse_score(
    candidate: Candidate,
    bars: pd.DataFrame,
    config: PrefilterConfig | None = None,
    families: FamilyRegistry | None = None,
) -> float:
    """A cheap ranking score for one candidate (default: Sharpe of coarse returns)."""
    config = config or PrefilterConfig()
    strat = candidate.build(families if families is not None else FamilyRegistry())
    targets = type(strat).signals(bars, strat.params)
    rets = vectorized_returns(bars, targets, config.fee_bps, config.slippage_bps)
    # parse tolerates a plain-string config, so an unknown metric fails with the one message.
    return Metric.parse(config.metric).from_returns(
        list(rets),
        config.periods_per_year,
        annualization_cap=config.annualization_cap,
        max_period_ratio=config.max_period_ratio,
    )


@dataclass(frozen=True)
class ScreenResult:
    candidate: Candidate
    score: float


def screen(
    candidates: list[Candidate],
    bars: pd.DataFrame,
    top_k: int,
    config: PrefilterConfig | None = None,
    families: FamilyRegistry | None = None,
) -> list[ScreenResult]:
    """Score all candidates and return the top-K by coarse metric (descending)."""
    config = config or PrefilterConfig()
    scored = [ScreenResult(c, coarse_score(c, bars, config, families)) for c in candidates]
    scored = [s for s in scored if not np.isnan(s.score)]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
