"""The strategy abstraction shared by research and live.

A ``TraderStrategy`` exposes two code paths driven by the **same parameters**:

* :meth:`signals` — a vectorised classmethod over an OHLCV frame returning a target-position
  series (+1 long / 0 flat / −1 short) per bar. The fast pre-filter consumes this.
* :meth:`on_bar` — an incremental, event-driven decision (no pandas) used in backtest
  validation and live.

The **parity contract** is that both paths make the same decision on the same bars. The
default :meth:`signals` replays :meth:`on_bar` over the frame, so for strategies that don't
override it parity holds *by construction* — an authored strategy implements only
``on_start``/``on_bar``/``param_space``. A strategy may still override ``signals`` with a
vectorised implementation for a faster pre-filter path; that override is what the golden
parity tests guard. ``param_space`` declares the tunable parameters for the search layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar (UTC-ns timestamp)."""

    ts_event: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class ExitRules:
    """Protective-exit percentages a strategy declares alongside its target.

    Declarative and engine-enforced: the strategy states the rules, the engine evaluates
    them intrabar against subsequent OHLC, and the strategy never observes whether one
    fired (see the fill-model section of docs/architecture.md). All three are fractions
    of the entry price (scale-free across a symbol panel), each armed only when not None.
    """

    stop_pct: float | None = None  # exit if adverse move ≥ this fraction of entry
    take_profit_pct: float | None = None
    trail_pct: float | None = None  # exit if drawdown from best-since-entry ≥ this


@dataclass(frozen=True)
class ParamSpec:
    """A tunable parameter's search domain (consumed by the Optuna factory later)."""

    name: str
    kind: str  # "int" | "float" | "categorical"
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple = ()


@runtime_checkable
class Context(Protocol):
    """How a strategy expresses intent to the engine during ``on_bar``."""

    def set_target(self, target: int, *, exits: ExitRules | None = None) -> None:
        """Set the desired directional position for the current bar (+1/0/−1).

        ``exits`` declares protective-exit rules alongside the target — re-declared with
        every call, stateless from the strategy's side; the engine associates them with
        the position. Omitting it (every pre-exits call site) declares no protection.
        """
        ...


class TraderStrategy(ABC):
    """Base class for all strategies. Subclasses set ``name`` and ``params_cls``."""

    name: str = "base"
    params_cls: type
    # The bar granularity the thesis needs (see ``noctis.data.aggregate.TIMEFRAMES``).
    # The lake stores 1-minute bars; research and live aggregate to this on the way in,
    # so ``on_bar``/``signals`` always see bars of the declared timeframe.
    timeframe: str = "1m"

    def __init__(self, params):
        self.params = params

    @classmethod
    def create(cls, **kwargs) -> TraderStrategy:
        """Instantiate from keyword parameters (fills defaults for anything omitted)."""
        return cls(cls.params_cls(**kwargs))

    @classmethod
    def default(cls) -> TraderStrategy:
        return cls(cls.params_cls())

    def params_dict(self) -> dict:
        if is_dataclass(self.params) and not isinstance(self.params, type):
            return asdict(self.params)
        raise TypeError("params object is not a dataclass; override params_dict()")

    # --- the two parity-linked code paths ---
    @classmethod
    def signals(cls, data: pd.DataFrame, params) -> pd.Series:
        """Vectorised target-position series (+1/0/−1) aligned to ``data`` rows.

        Default: replay :meth:`on_bar` over the frame (see :func:`replay_targets`), so both
        code paths agree by construction. Override with a true vectorised implementation
        only as a performance optimization — the override must preserve parity.
        """
        return pd.Series(replay_targets(cls(params), data), dtype=int)

    @abstractmethod
    def on_start(self, ctx: Context) -> None:
        """Reset incremental state at the start of a run."""

    @abstractmethod
    def on_bar(self, ctx: Context, bar: Bar) -> None:
        """React to one bar; call ``ctx.set_target(...)`` with the desired position."""

    # --- search domain ---
    @classmethod
    @abstractmethod
    def param_space(cls) -> list[ParamSpec]:
        """Declarative tunable-parameter domain."""

    # --- known-outcome oracle ---
    @classmethod
    def scenarios(cls) -> list:
        """Known-outcome scenarios (see ``noctis.strategies.scenarios``).

        Library-authored strategies must override this — the ``write_strategy`` gate
        replays the declared tapes and rejects code that violates its own expectations.
        Non-library strategies (spec-compiled, test doubles) keep this empty default.
        """
        return []

    @classmethod
    def warmup_bars(cls, params) -> int:
        """Decision bars before which this strategy promises to stay flat.

        Higher-timeframe filters included — the author multiplies here, once,
        in code it can see. Default 0 means undeclared (exempt from the
        honesty check), so nothing outside the library breaks.
        """
        return 0


class TargetContext:
    """The concrete :class:`Context` every driver uses: captures the target ``on_bar`` sets.

    Shared by the replay reference path below, the backtest simulator, and the live loop —
    one implementation, so a target can never mean different things per driver.
    """

    def __init__(self) -> None:
        self.target = 0
        self.exits: ExitRules | None = None

    def set_target(self, target: int, *, exits: ExitRules | None = None) -> None:
        self.target = int(target)
        self.exits = exits


def replay_targets(strategy: TraderStrategy, frame: pd.DataFrame) -> list[int]:
    """Replay ``on_bar`` over ``frame`` and return the per-bar target series.

    This is the reference (event-path) decision sequence: the default ``signals()`` wraps
    it, and the ideation parity gate compares vectorised overrides against it.
    """
    ctx = TargetContext()
    strategy.on_start(ctx)
    out: list[int] = []
    for row in frame.reset_index(drop=True).itertuples(index=False):
        bar = Bar(
            int(row.ts_event),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        strategy.on_bar(ctx, bar)
        out.append(ctx.target)
    return out


def params_to_dict(params) -> dict:
    if is_dataclass(params):
        return {f.name: getattr(params, f.name) for f in fields(params)}
    raise TypeError("params object is not a dataclass")
