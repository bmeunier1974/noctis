"""The execution-realistic validation stage.

Runs the event-driven simulator (next-bar execution, realistic fees + slippage) over each
walk-forward split — separately on the train and test windows — and produces per-split
metrics. This stage, not the pre-filter, decides promotion. It refuses symbols that are not
``idle`` in the coverage registry. Results are deterministic: identical inputs and seed
yield identical output.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from noctis.backtest.candidate import Candidate
from noctis.backtest.scorecard import (
    DEFAULT_ANNUALIZATION_CAP,
    DEFAULT_MAX_PERIOD_RATIO,
    Metrics,
    SplitScore,
    compute_metrics,
)
from noctis.backtest.splits import Split
from noctis.broker.paper import PaperBroker
from noctis.broker.seam import FeeModel, SlippageModel
from noctis.broker.simulator import simulate
from noctis.strategies.families import FamilyRegistry


@dataclass(frozen=True)
class ValidationConfig:
    starting_cash: float = 100_000.0
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    alloc: float = 0.95
    periods_per_year: int = 252
    # Metric-robustness caps (see scorecard._annualization / _cap_ratio); active by default.
    annualization_cap: int | None = DEFAULT_ANNUALIZATION_CAP
    max_period_ratio: float | None = DEFAULT_MAX_PERIOD_RATIO
    seed: int = 0  # reserved: the engine is deterministic, so results don't depend on it


class SymbolNotReadyError(RuntimeError):
    """Raised when a backtest is asked to run on a symbol that is not idle/tracked."""


def require_symbols_ready(registry, symbols, dataset=None, schema=None) -> None:
    """Refuse symbols that are untracked or not ``idle`` in the coverage registry."""
    not_ready = [s for s in symbols if not registry.check_symbol_ready(s, dataset, schema)]
    if not_ready:
        raise SymbolNotReadyError(
            f"symbols not ready for backtest (untracked or not idle): {sorted(not_ready)}"
        )


def _metrics_for(strategy, window: pd.DataFrame, config: ValidationConfig) -> Metrics:
    broker = PaperBroker(
        starting_cash=config.starting_cash,
        fee_model=FeeModel(config.fee_bps),
        slippage_model=SlippageModel(config.slippage_bps),
    )
    result = simulate(strategy, window, broker, alloc=config.alloc)
    return compute_metrics(
        result.equity_curve,
        result.targets,
        config.periods_per_year,
        annualization_cap=config.annualization_cap,
        max_period_ratio=config.max_period_ratio,
    )


def score_window(
    candidate: Candidate,
    window: pd.DataFrame,
    config: ValidationConfig | None = None,
    families: FamilyRegistry | None = None,
) -> Metrics:
    """Metrics from one causal simulation over ``window`` — the forward-holdout scorer.

    Runs the same event-driven, next-bar simulator as walk-forward validation, on a freshly
    built strategy, so a holdout metric is directly comparable to the per-split test metrics."""
    config = config or ValidationConfig()
    families = families if families is not None else FamilyRegistry()
    return _metrics_for(candidate.build(families), window.reset_index(drop=True), config)


def validate_candidate(
    candidate: Candidate,
    bars: pd.DataFrame,
    splits: list[Split],
    config: ValidationConfig | None = None,
    families: FamilyRegistry | None = None,
) -> list[SplitScore]:
    """Validate a candidate across walk-forward splits, returning per-split train/test metrics."""
    config = config or ValidationConfig()
    families = families if families is not None else FamilyRegistry()
    rows = bars.reset_index(drop=True)
    scores: list[SplitScore] = []
    for split in splits:
        train = rows.iloc[split.train_slice()].reset_index(drop=True)
        test = rows.iloc[split.test_slice()].reset_index(drop=True)
        train_metrics = _metrics_for(candidate.build(families), train, config)
        test_metrics = _metrics_for(candidate.build(families), test, config)
        scores.append(SplitScore(split.index, train_metrics, test_metrics))
    return scores
