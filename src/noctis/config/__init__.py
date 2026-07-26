"""Typed configuration, the paper/live safety gate, and the deny-by-default settings overlay."""

from noctis.config.gate import SafetyGateError, resolve_execution_mode
from noctis.config.overlay import (
    OverlayError,
    apply_patch,
    assert_gates_unmoved,
    classify,
    gate_snapshot,
)
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
    "OverlayError",
    "apply_patch",
    "assert_gates_unmoved",
    "classify",
    "gate_snapshot",
]
