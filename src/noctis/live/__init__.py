"""Noctis live — the TRADING loop and its risk manager.

Champions trade replayed/live bars through the paper broker; every order is risk-checked and
a degraded feed halts emission. No real-money order path exists here.
"""

from __future__ import annotations

from noctis.live.feed import BarFeed, ReplayBarFeed
from noctis.live.node import (
    SessionConfig,
    TradingDayResult,
    TradingSummary,
    run_trading,
    run_trading_day,
)
from noctis.live.risk import RiskDecision, RiskLimits, RiskManager

__all__ = [
    "BarFeed",
    "ReplayBarFeed",
    "SessionConfig",
    "TradingSummary",
    "TradingDayResult",
    "run_trading",
    "run_trading_day",
    "RiskDecision",
    "RiskLimits",
    "RiskManager",
]
