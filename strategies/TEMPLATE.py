"""ONE-SENTENCE THESIS: what edge does this capture, and why should it persist?

(Elaborate the thesis here in a short paragraph: the market behaviour you are betting
on, why it exists — flows, risk premia, behavioural bias — and what would falsify it.
Then keep the header fields below; the research loop stamps status/symbols/tuned.)

status: draft
style: momentum | mean-reversion | breakout | seasonality | vol-regime | ...
"""

from collections import deque
from dataclasses import dataclass

# Tail helpers (None during warmup): sma/ema/rsi/atr/stdev/zscore/bollinger/roc/wma/
# highest/lowest/stoch_k/cci/bars_since + cross_above/cross_below. Stateful (nan warmup):
# Sma/Ema/Rsi/Atr/Macd/Vwap/Adx/Obv/Stoch/Supertrend/ZScore/RollingExtreme States.
# See the module docstring for variants and what is deliberately not provided.
from noctis.strategies import indicators as ind
from noctis.strategies import scenarios as sc  # segment builders + behavioral expectations

# Session-clock helpers over bar.ts_event (US-equity RTH, DST-correct): minute_of_session,
# is_rth, minutes_to_close, session_date, new_session. Gate the last N minutes with
#   m = session.minutes_to_close(bar.ts_event)
#   if m is not None and m <= 15: ctx.set_target(0); return
# and reset per-day state with `if session.new_session(self._prev_ts, bar.ts_event): ...`.
from noctis.strategies import session  # noqa: F401 — session-clock helpers for intraday theses

# Higher-timeframe access (a 5m strategy reading a 1h trend filter), lookahead-free — own an
# ind.HtfBars(coarser_timeframe) like any other state:
#   def on_start(self, ctx):
#       self.htf = ind.HtfBars("1h"); self.htf_ema = ind.EmaState(20); self.trend = float("nan")
#   def on_bar(self, ctx, bar):
#       done = self.htf.add(bar)                  # completed 1h bar or None
#       if done is not None: self.trend = self.htf_ema.update(done.close)
# The HTF must be a MULTIPLE of `timeframe`; HTF warmup multiplies (a 1h EMA(20) needs weeks
# of base bars — size scenarios() accordingly); the session-final partial bucket is never
# emitted. Vectorised twin for signals() overrides: aggregate.last_completed_htf(frame, tf).
#
# Protective exits (engine-enforced; see README §Protective exits): declare percent-based
# rules WITH your target and re-declare them every bar — forward ordinary float Params:
#   ctx.set_target(1, exits=ExitRules(stop_pct=self.params.stop_pct, trail_pct=0.03))
# You never observe your own stop-outs (on_bar sees no fills), and after one fires the
# engine holds the symbol flat until your target series CHANGES VALUE (the re-arm latch).
from noctis.strategies.base import (  # noqa: F401 — ExitRules for protective exits
    Bar,
    Context,
    ExitRules,
    ParamSpec,
    TraderStrategy,
)


class MyStrategy(TraderStrategy):
    # MUST equal the file name (write_strategy validates this).
    name = "TEMPLATE"

    # The bar granularity the THESIS needs: "1m", "5m", "15m", "30m", "1h", or "1d".
    # The lake stores 1m bars; research and live aggregate to this automatically, so
    # on_bar sees bars of this size and lookbacks count bars of this size. Pick the
    # horizon whose per-trade move clears the round-trip cost.
    timeframe = "1m"

    # Frozen dataclass of tunable parameters. The defaults are what `noctis backtest
    # <name>` runs with; on champion promotion the loop rewrites them to the tuned values.
    @dataclass(frozen=True)
    class Params:
        lookback: int = 20
        threshold: float = 1.0

    params_cls = Params

    def on_start(self, ctx: Context) -> None:
        """Reset ALL incremental state (a strategy instance may be replayed many times)."""
        self._closes: deque[float] = deque(maxlen=self.params.lookback)
        self._pos = 0

    def on_bar(self, ctx: Context, bar: Bar) -> None:
        """React to one bar; end by ctx.set_target(+1) long, ctx.set_target(-1) short, or
        ctx.set_target(0) flat (this template only goes long/flat; short is available too).

        Long/flat only. Keep the work per bar O(lookback): accumulate into bounded deques
        and use the `ind` tail helpers (they return None during warmup — guard for it).
        Decisions may only use bars ALREADY seen (this bar and earlier) — no lookahead.
        """
        self._closes.append(bar.close)
        mean = ind.sma(self._closes, self.params.lookback)
        if mean is None:
            ctx.set_target(0)
            return
        ctx.set_target(1 if bar.close > mean * self.params.threshold else 0)

    @classmethod
    def warmup_bars(cls, params) -> int:
        """Decision bars before which this strategy promises to stay flat.

        The only model-owned number in the oracle — derive it from your own lookback logic
        (here the SMA needs `lookback` closes before it yields a value). The write gate replays
        your scenarios() and rejects the file if any nonzero target appears before this bar, so
        an honest declaration is one your own tapes prove. Multiply here for higher-timeframe
        filters (a 1h EMA(20) over 5m bars warms up in ~20 completed hours of base bars). Leave
        the base default of 0 only if the strategy genuinely trades from the first bar.
        """
        return params.lookback

    @classmethod
    def param_space(cls) -> list[ParamSpec]:
        """The search domain run_sweep explores. Cover every Params field worth tuning."""
        return [
            ParamSpec("lookback", "int", low=5, high=60, step=1),
            ParamSpec("threshold", "float", low=0.95, high=1.05, step=0.005),
        ]

    @classmethod
    def scenarios(cls) -> list[sc.Scenario]:
        """Known-outcome tapes the write gate replays against this code.

        Declare these from the THESIS, before writing on_bar: 2-8 scenarios, at least one
        tape whose shape demands an entry (long_within/holds_long_through) and one no-trade
        tape (always_flat). Derive windows from `cls.params_cls()` defaults so promotion
        write-back (which rewrites the defaults) keeps passing. If the code violates its
        own scenarios, write_strategy rejects it.
        """
        warm = cls.params_cls().lookback
        return [
            sc.Scenario(
                "rally_above_mean_then_fade",
                segments=[sc.flat(warm + 5), sc.trend(40, 0.15), sc.selloff(30, 0.20)],
                expect=[
                    sc.flat_until(warm),  # no position during SMA warmup
                    sc.long_within(warm + 5, warm + 15),  # the rally must pull us long
                    sc.flat_by(warm + 60),  # the fade must shake us out
                ],
            ),
            sc.Scenario(
                "steady_decline_stays_flat",
                segments=[sc.flat(warm + 5), sc.selloff(50, 0.25)],
                expect=[sc.always_flat()],  # below the mean throughout: never long
            ),
        ]
