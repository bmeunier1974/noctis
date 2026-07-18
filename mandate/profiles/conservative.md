---
summary: Capital preservation on liquid large-caps; smooth returns, scored on sharpe.
config:
  promotion:
    metric: sharpe
---
# Conservative

Trade like a capital-preservation operator who prizes a smooth ride over a big number.
Concentrate on the most liquid US large-caps — deep order books, tight spreads, names that
do not gap violently — and prefer strategies whose equity curve is steady and whose
drawdowns are shallow. A modest, consistent edge beats a spectacular but erratic one. Score
on Sharpe, which penalises all volatility (upside and downside alike): the `config:` overlay
binds `promotion.metric: sharpe` to make risk-adjusted steadiness the yardstick.

Bias idea generation toward mean-reversion on liquid names, low-turnover trend-following,
and positions sized to keep exposure calm. Avoid thinly traded or headline-driven tickers.

This profile steers idea and symbol selection only; it never loosens a gate, the exhaustion
rule, or the honesty contract — those still bind. If a low-volatility thesis cannot clear
the gates or the cost hurdle, record the class-level conclusion and pivot to the nearest
viable variant rather than relaxing the evidence bar.
