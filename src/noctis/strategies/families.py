"""The strategy-family registry — name → :class:`TraderStrategy` class.

One :class:`FamilyRegistry` instance is built per process by the composition root
(``noctis.bootstrap.build_families``: in-repo seeds → persisted spec-families → library
files, in that order) and injected into everything that resolves families by name; tests
build their own throwaway instances. Nothing here is module-global, so one session's
registrations can never leak into another's.

The one exception to "everything is injected" is the backtest process pools: a registry
holding spec/library-minted classes is not picklable, so it crosses a **fork** boundary by
inheritance (``noctis.backtest.pipeline``) and a spawn-safe worker rebuilds its own from
the library directory (``noctis.research.tools``). Candidates themselves stay picklable —
a family *name* plus params — which is why resolution happens against a registry at all.
"""

from __future__ import annotations

from collections.abc import Iterable

from noctis.strategies.base import ParamSpec, TraderStrategy
from noctis.strategies.donchian_breakout import DonchianBreakout
from noctis.strategies.rsi_meanrev import RsiMeanReversion
from noctis.strategies.sma_crossover import SmaCrossover

# The in-repo seed families every registry starts from (``FamilyRegistry(())`` for none).
SEED_CLASSES: tuple[type[TraderStrategy], ...] = (
    SmaCrossover,
    RsiMeanReversion,
    DonchianBreakout,
)

# The families the research proposer samples from by default. Registering an extra family
# makes it resolvable via the registry but does NOT auto-enroll it in the proposer — the
# proposer must be given an explicit rotation (or ``add_family``) to include it.
SEED_FAMILIES: tuple[str, ...] = tuple(cls.name for cls in SEED_CLASSES)


class FamilyRegistry:
    """Name + params → strategy instance, over an explicit set of registered families."""

    def __init__(self, classes: Iterable[type[TraderStrategy]] | None = None) -> None:
        self._families: dict[str, type[TraderStrategy]] = {
            cls.name: cls for cls in (SEED_CLASSES if classes is None else classes)
        }

    def register(self, cls: type[TraderStrategy]) -> None:
        """Register a strategy family (idempotent; re-registering a name overwrites it)."""
        self._families[cls.name] = cls

    def __contains__(self, name: str) -> bool:
        return name in self._families

    def names(self) -> list[str]:
        return sorted(self._families)

    def get_class(self, name: str) -> type[TraderStrategy]:
        if name not in self._families:
            raise KeyError(f"unknown strategy {name!r}; known: {sorted(self._families)}")
        return self._families[name]

    def create(self, name: str, params: dict | None = None) -> TraderStrategy:
        return self.get_class(name).create(**(params or {}))

    def param_space(self, name: str) -> list[ParamSpec]:
        return self.get_class(name).param_space()
