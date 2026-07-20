"""Typed configuration and the paper/live safety gate."""

from noctis.config.gate import SafetyGateError, resolve_execution_mode
from noctis.config.settings import (
    BacktestConfig,
    DataConfig,
    PromotionConfig,
    RiskConfig,
    SessionConfig,
    Settings,
    load_settings,
)

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "PromotionConfig",
    "RiskConfig",
    "SessionConfig",
    "Settings",
    "load_settings",
    "SafetyGateError",
    "resolve_execution_mode",
]
