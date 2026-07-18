"""Donchian breakout — long on a break above the prior N-bar high, exit below the low."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


@dataclass(frozen=True)
class DonchianParams:
    channel: int = 20


class DonchianBreakout(TraderStrategy):
    name = "donchian_breakout"
    params_cls = DonchianParams

    @classmethod
    def signals(cls, data: pd.DataFrame, params: DonchianParams) -> pd.Series:
        high = data["high"].astype("float64").reset_index(drop=True)
        low = data["low"].astype("float64").reset_index(drop=True)
        close = data["close"].astype("float64").reset_index(drop=True)
        # Prior N-bar channel (shift excludes the current bar → no lookahead).
        upper = high.rolling(params.channel).max().shift(1)
        lower = low.rolling(params.channel).min().shift(1)

        targets: list[int] = []
        pos = 0
        for i in range(len(close)):
            u = upper.iat[i]
            low_bound = lower.iat[i]
            if pd.isna(u) or pd.isna(low_bound):
                targets.append(0)
                continue
            if pos == 0 and close.iat[i] > u:
                pos = 1
            elif pos == 1 and close.iat[i] < low_bound:
                pos = 0
            targets.append(pos)
        return pd.Series(targets, dtype=int)

    def on_start(self, ctx: Context) -> None:
        n = self.params.channel
        self._highs: deque[float] = deque(maxlen=n)
        self._lows: deque[float] = deque(maxlen=n)
        self._pos = 0

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        if len(self._highs) < self.params.channel:
            self._highs.append(bar.high)
            self._lows.append(bar.low)
            ctx.set_target(0)
            return
        upper = max(self._highs)
        lower = min(self._lows)
        if self._pos == 0 and bar.close > upper:
            self._pos = 1
        elif self._pos == 1 and bar.close < lower:
            self._pos = 0
        ctx.set_target(self._pos)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [ParamSpec("channel", "int", low=5, high=60, step=1)]
