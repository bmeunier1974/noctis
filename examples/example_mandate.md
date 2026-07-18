---
summary: Conservative long-only trend-following on large-cap US index names (Sharpe)
config:
  promotion:
    metric: sharpe
---
<!--
  A minimal EXAMPLE operator mandate. A mandate is the operator's own input to the research
  agent — the one place you say, in your own words, what kind of trader the system should be.
  To use a mandate like this, put your version in mandate/MANDATE.md and set
  `research.mandate: MANDATE` in config.yaml. See mandate/README.md for the full authoring
  guide, the shipped profiles, and precedence rules.

  The front-matter `config:` block may bind EXACTLY ONE knob — promotion.metric
  (sharpe | sortino | total_return). It steers the risk dial; it never loosens a gate,
  the exhaustion rule, or the honesty contract.
-->

I want a conservative, long-only trend-following system on large, liquid US names — broad index
ETFs and mega-cap leaders (e.g. SPY, AAPL, MSFT). Favor durable, multi-week trends over fast
intraday moves: enter long only after a trend is clearly established and a healthy pullback has
passed, and step flat on a genuine regime change rather than trying to short the down-leg.

Prefer steadiness to peak return: I would rather a smoother equity curve (hence the Sharpe
election metric) than a higher headline number with violent drawdowns. Avoid thinly-traded or
highly speculative names entirely.
