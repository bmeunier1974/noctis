"""RSI mean reversion — go long when oversold, exit when overbought (long/flat)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


@dataclass(frozen=True)
class RsiParams:
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


class RsiMeanReversion(TraderStrategy):
    name = "rsi_meanrev"
    params_cls = RsiParams

    @classmethod
    def signals(cls, data: pd.DataFrame, params: RsiParams) -> pd.Series:
        close = data["close"].astype("float64").reset_index(drop=True)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(params.period).mean()
        avg_loss = loss.rolling(params.period).mean()

        targets: list[int] = []
        pos = 0
        for ag, al in zip(avg_gain, avg_loss, strict=True):
            if pd.isna(ag) or pd.isna(al):
                targets.append(0)
                continue
            rsi = _rsi_from_avgs(ag, al)
            if pos == 0 and rsi < params.oversold:
                pos = 1
            elif pos == 1 and rsi > params.overbought:
                pos = 0
            targets.append(pos)
        return pd.Series(targets, dtype=int)

    def on_start(self, ctx: Context) -> None:
        self._prev_close: float | None = None
        self._gains: deque[float] = deque(maxlen=self.params.period)
        self._losses: deque[float] = deque(maxlen=self.params.period)
        self._pos = 0

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        if self._prev_close is None:
            self._prev_close = bar.close
            ctx.set_target(0)
            return
        delta = bar.close - self._prev_close
        self._prev_close = bar.close
        self._gains.append(max(delta, 0.0))
        self._losses.append(max(-delta, 0.0))
        if len(self._gains) < self.params.period:
            ctx.set_target(0)
            return
        avg_gain = sum(self._gains) / len(self._gains)
        avg_loss = sum(self._losses) / len(self._losses)
        rsi = _rsi_from_avgs(avg_gain, avg_loss)
        if self._pos == 0 and rsi < self.params.oversold:
            self._pos = 1
        elif self._pos == 1 and rsi > self.params.overbought:
            self._pos = 0
        ctx.set_target(self._pos)

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("period", "int", low=5, high=30, step=1),
            ParamSpec("oversold", "float", low=10.0, high=40.0, step=1.0),
            ParamSpec("overbought", "float", low=60.0, high=90.0, step=1.0),
        ]
