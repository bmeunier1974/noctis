"""Ride the medium-term trend: hold long while the fast moving average is above the slow.

The classic trend filter: a fast SMA above a slow SMA says recent demand outpaces the
longer-run average, so stay long; below, stay flat. No shorting — the edge claimed is
trend persistence, not symmetry.

status: candidate
style: momentum
"""

from collections import deque
from dataclasses import dataclass

from noctis.strategies import indicators as ind
from noctis.strategies import scenarios as sc
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class SmaCrossover(TraderStrategy):
    name = "sma_crossover"

    @dataclass(frozen=True)
    class Params:
        fast: int = 10
        slow: int = 30

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        self._closes: deque[float] = deque(maxlen=self.params.slow)

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._closes.append(bar.close)
        fast = ind.sma(self._closes, self.params.fast)
        slow = ind.sma(self._closes, self.params.slow)
        if fast is None or slow is None:
            ctx.set_target(0)
            return
        ctx.set_target(1 if fast > slow else 0)

    @classmethod
    def warmup_bars(cls, params) -> int:
        # The slow SMA is the last indicator to come online — it needs `slow` closes before
        # ind.sma returns a value, so nothing can trade until then.
        return params.slow

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("fast", "int", low=3, high=30, step=1),
            ParamSpec("slow", "int", low=20, high=100, step=1),
        ]

    @classmethod
    def scenarios(cls) -> list[sc.Scenario]:
        # Windows derive from the Params defaults so promotion write-back keeps passing.
        warm = cls.params_cls().slow
        return [
            sc.Scenario(
                "trend_ride_then_rollover",
                segments=[sc.flat(warm + 5), sc.trend(40, 0.12), sc.selloff(40, 0.20)],
                expect=[
                    sc.flat_until(warm),
                    sc.long_within(warm + 5, warm + 45),
                    sc.flat_by(warm + 80),
                ],
            ),
            sc.Scenario(
                "steady_decline_never_longs",
                segments=[sc.flat(warm + 5), sc.selloff(60, 0.25)],
                expect=[sc.always_flat()],
            ),
        ]
