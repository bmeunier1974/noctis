"""Noctis champions — the persistent champion set and promotion rules.

The registry survives restarts (atomic JSON persistence) and holds a decision history so
champion changes can be explained in the close report and memory. Promotion is decided by a
pure function: better out-of-sample test metric, guarded by the train − test gap.
"""

from __future__ import annotations

from pathlib import Path

from noctis.champions.promotion import Decision, PromotionRules, decide
from noctis.champions.registry import ChampionEntry, ChampionRegistry

__all__ = [
    "Decision",
    "PromotionRules",
    "decide",
    "ChampionEntry",
    "ChampionRegistry",
    "registry_path",
    "build_registry",
]


def registry_path(settings) -> Path:
    return Path(settings.state_dir) / "champions.json"


def build_registry(settings) -> ChampionRegistry:
    """Construct the champion registry from application settings."""
    return ChampionRegistry(registry_path(settings), settings.champion_count)
