"""SMA crossover — long when the fast SMA is above the slow SMA, else flat."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


@dataclass(frozen=True)
class SmaParams:
    fast: int = 10
    slow: int = 30


class SmaCrossover(TraderStrategy):
    name = "sma_crossover"
    params_cls = SmaParams

    @classmethod
    def signals(cls, data: pd.DataFrame, params: SmaParams) -> pd.Series:
        close = data["close"].astype("float64").reset_index(drop=True)
        fast = close.rolling(params.fast).mean()
        slow = close.rolling(params.slow).mean()
        target = (fast > slow).astype(int)
        target[fast.isna() | slow.isna()] = 0
        return target.astype(int)

    def on_start(self, ctx: Context) -> None:
        self._closes: deque[float] = deque(maxlen=self.params.slow)

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._closes.append(bar.close)
        if len(self._closes) < self.params.slow:
            ctx.set_target(0)
            return
        slow_ma = sum(self._closes) / len(self._closes)
        fast_window = list(self._closes)[-self.params.fast :]
        fast_ma = sum(fast_window) / len(fast_window)
        ctx.set_target(1 if fast_ma > slow_ma else 0)

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("fast", "int", low=3, high=30, step=1),
            ParamSpec("slow", "int", low=20, high=100, step=1),
        ]
