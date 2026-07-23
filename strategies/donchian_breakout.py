"""Chase confirmed breakouts: long on a close above the prior N-bar high, out below the low.

Donchian channel logic — a close above everything the market paid in the last N bars is
evidence of fresh information or flow, and trends born that way tend to run; a close
below the channel low says the move failed. The channel excludes the current bar, so
the breakout is judged against strictly prior history (no lookahead).

status: candidate
style: breakout
"""

from collections import deque
from dataclasses import dataclass

from noctis.strategies import indicators as ind
from noctis.strategies import scenarios as sc
from noctis.strategies.base import Bar, Context, ParamSpec, TraderStrategy


class DonchianBreakout(TraderStrategy):
    name = "donchian_breakout"

    @dataclass(frozen=True)
    class Params:
        channel: int = 20

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        n = self.params.channel
        self._highs: deque[float] = deque(maxlen=n)
        self._lows: deque[float] = deque(maxlen=n)
        self._pos = 0

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        # Channel from the PRIOR N bars only — read before appending the current bar.
        upper = ind.highest(self._highs, self.params.channel)
        lower = ind.lowest(self._lows, self.params.channel)
        if upper is None or lower is None:
            self._highs.append(bar.high)
            self._lows.append(bar.low)
            ctx.set_target(0)
            return
        if self._pos == 0 and bar.close > upper:
            self._pos = 1
        elif self._pos == 1 and bar.close < lower:
            self._pos = 0
        ctx.set_target(self._pos)
        self._highs.append(bar.high)
        self._lows.append(bar.low)

    @classmethod
    def warmup_bars(cls, params) -> int:
        # The channel reads the prior `channel` bars (highest/lowest return None until the
        # deque holds `channel` values), so no breakout can fire before then.
        return params.channel

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        return [ParamSpec("channel", "int", low=5, high=60, step=1)]

    @classmethod
    def scenarios(cls) -> list[sc.Scenario]:
        # Windows derive from the Params defaults so promotion write-back keeps passing.
        # The flat tape never closes above its own channel high (close * 1.002), so the
        # breakout only fires on the trend leg.
        chan = cls.params_cls().channel
        return [
            sc.Scenario(
                "breakout_run_then_breakdown",
                segments=[sc.flat(chan + 5), sc.trend(20, 0.10), sc.selloff(30, 0.20)],
                expect=[
                    sc.flat_until(chan),
                    sc.long_within(chan + 5, chan + 10),
                    sc.flat_by(chan + 50),
                ],
            ),
            sc.Scenario(
                "flat_tape_never_breaks_out",
                segments=[sc.flat(max(60, chan + 40))],
                expect=[sc.always_flat()],
            ),
        ]
