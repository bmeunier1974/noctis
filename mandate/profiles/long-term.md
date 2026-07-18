---
summary: Multi-day/daily horizon, durable effects on liquid names; risk-adjusted, scored on sharpe.
config:
  promotion:
    metric: sharpe
---
# Long-term

Trade like a patient operator hunting durable, slow-moving edges. Prefer strategies that
hold for multiple days to weeks and that lean on effects with a real economic reason to
persist — trend, carry, seasonality, post-event drift — rather than fleeting intraday
noise. A coarse `timeframe` such as `1h` or `1d` suits this horizon: the signal should
survive being sampled slowly. Score on Sharpe so the edge has to be steady and
risk-adjusted, not a lucky streak: the `config:` overlay binds `promotion.metric: sharpe`.

Bias idea generation toward multi-day momentum and trend continuation, mean-reversion over
daily bars, and regime-aware allocation on liquid, well-established names. Turnover should
be low; the edge should show up across many windows, not a few.

This profile steers idea and symbol selection only; it never loosens a gate, the exhaustion
rule, or the honesty contract — those still bind. If a durable-effect thesis cannot clear
the walk-forward or symbol-holdout gates, record the class-level conclusion and pivot to the
nearest viable variant rather than overfitting a longer hold to the sample.
