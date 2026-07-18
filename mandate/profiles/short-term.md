---
summary: Intraday/short holds, downside-focused; must clear the cost hurdle, scored on sortino.
config:
  promotion:
    metric: sortino
---
# Short-term

Trade like an intraday operator who lives and dies by short holds. Prefer strategies that
open and close within a session or over a couple of days, on names liquid enough to enter
and exit cleanly. Upside volatility is welcome; it is the downside that must be controlled,
so score on Sortino, which penalises only harmful (downside) deviation: the `config:`
overlay binds `promotion.metric: sortino`.

Bias idea generation toward intraday momentum, opening-range and breakout plays, and quick
mean-reversion snaps — but respect the arithmetic: short holds trade often, so every
candidate must still clear the transaction-cost hurdle after fees and slippage. A signal
that only looks good gross is not an edge. High turnover is fine only when the net,
post-cost edge survives.

This profile steers idea and symbol selection only; it never loosens a gate, the exhaustion
rule, or the honesty contract — those still bind. If a short-horizon thesis cannot clear the
cost hurdle or the gates, record the class-level conclusion and pivot to the nearest viable
variant rather than pushing a strategy whose edge evaporates net of costs.
