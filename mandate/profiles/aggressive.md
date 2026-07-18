---
summary: Raw profit on volatile US names, short holds, high risk appetite; scored on total_return.
config:
  promotion:
    metric: total_return
---
# Aggressive

Trade like a high-risk-appetite operator chasing raw return. Favour the most volatile,
high-beta US names — the ones that move hard intraday and over a handful of sessions — and
prefer strategies that hold for minutes to a few days, not weeks. Big up-moves are the
prize; a bumpy equity curve is an acceptable price. Score on total return, so volatility is
not penalised: the `config:` overlay binds `promotion.metric: total_return` to make that
appetite the yardstick.

Bias idea generation toward momentum bursts, breakout continuation, and gap plays on names
with wide daily ranges and heavy volume. Discovering fresh volatile tickers beyond the base
universe is encouraged when a thesis calls for them.

This profile steers idea and symbol selection only; it never loosens a gate, the exhaustion
rule, or the honesty contract — those still bind. If an aggressive thesis cannot clear the
walk-forward, symbol-holdout, or cost hurdle, record the class-level conclusion and pivot to
the nearest viable variant rather than forcing a marginal candidate through.
