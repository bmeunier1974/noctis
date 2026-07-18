"""Buy oversold dips and exit once momentum snaps back to overbought.

Short-horizon mean reversion: a low RSI marks capitulation-flavoured selling that tends
to bounce; a high RSI marks the bounce being spent. Long/flat only, with hysteresis —
enter below ``oversold``, hold until ``overbought`` — so the position doesn't chatter
around a single threshold.

status: candidate
style: mean-reversion
"""

from collections import deque
from dataclasses import dataclass

from noctis.strategies import indicators as ind
from noctis.strategies import scenarios as sc
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class RsiMeanRev(TraderStrategy):
    name = "rsi_meanrev"

    @dataclass(frozen=True)
    class Params:
        period: int = 14
        oversold: float = 30.0
        overbought: float = 70.0

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        # period deltas need period + 1 closes; ind.rsi returns None until then.
        self._closes: deque[float] = deque(maxlen=self.params.period + 1)
        self._pos = 0

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        self._closes.append(bar.close)
        value = ind.rsi(self._closes, self.params.period)
        if value is None:
            ctx.set_target(0)
            return
        if self._pos == 0 and value < self.params.oversold:
            self._pos = 1
        elif self._pos == 1 and value > self.params.overbought:
            self._pos = 0
        ctx.set_target(self._pos)

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("period", "int", low=5, high=30, step=1),
            ParamSpec("oversold", "float", low=10.0, high=40.0, step=1.0),
            ParamSpec("overbought", "float", low=60.0, high=90.0, step=1.0),
        ]

    @classmethod
    def scenarios(cls) -> list[sc.Scenario]:
        # Windows derive from the Params defaults so promotion write-back keeps passing.
        # A flat tape pins Cutler RSI at 100 (zero losses), so the base never enters; the
        # RSI window is all-gains at latest `period` bars into the recovery, forcing exit.
        p = cls.params_cls().period
        return [
            sc.Scenario(
                "capitulation_then_recovery",
                segments=[sc.flat(p + 15), sc.selloff(10, 0.06), sc.recovery(40, 0.11)],
                expect=[
                    sc.flat_until(p),
                    sc.long_within(p + 15, p + 26),
                    sc.flat_by(2 * p + 28),
                ],
            ),
            sc.Scenario(
                "steady_grind_up_never_oversold",
                segments=[sc.flat(p + 15), sc.recovery(60, 0.15)],
                expect=[sc.always_flat()],
            ),
        ]
