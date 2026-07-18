"""Noctis backtest — two-stage measurement core.

A cheap vectorised pre-filter screens many candidates; survivors are validated with an
execution-realistic, walk-forward event backtest and scored. Ranking is by the out-of-sample
test metric; the train − test gap is the overfit signal. Deterministic, no lookahead.
"""

from __future__ import annotations

from noctis.backtest.candidate import Candidate
from noctis.backtest.pipeline import PipelineConfig, evaluate
from noctis.backtest.prefilter import (
    PrefilterConfig,
    ScreenResult,
    coarse_score,
    screen,
    vectorized_returns,
)
from noctis.backtest.scorecard import (
    Metric,
    Metrics,
    Scorecard,
    SplitScore,
    SymbolScore,
    compute_metrics,
    max_drawdown,
    sharpe,
    sortino,
)
from noctis.backtest.splits import Split, walk_forward
from noctis.backtest.validate import (
    SymbolNotReadyError,
    ValidationConfig,
    require_symbols_ready,
    validate_candidate,
)

__all__ = [
    "Candidate",
    "PipelineConfig",
    "evaluate",
    "PrefilterConfig",
    "ScreenResult",
    "coarse_score",
    "screen",
    "vectorized_returns",
    "Metric",
    "Metrics",
    "Scorecard",
    "SplitScore",
    "SymbolScore",
    "compute_metrics",
    "max_drawdown",
    "sharpe",
    "sortino",
    "Split",
    "walk_forward",
    "SymbolNotReadyError",
    "ValidationConfig",
    "require_symbols_ready",
    "validate_candidate",
]
