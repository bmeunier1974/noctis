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

  The front-matter `config:` block binds the RUN-SHAPING settings this personality needs —
  the model seam, the spend ceilings, the search shape (promotion.metric among them), the
  data window, the seed universe. It never touches the ARENA: the safety mode, the fill
  costs, the promotion thresholds, the holdout geometry, the paths and the secrets are
  refused by name, and a refused key stops the run at startup with the reason printed.
  The whole surface ships commented out in mandate/MANDATE.md.example; the reference is
  docs/configuration.md ("The mandate overlay"). This example binds only the risk dial.
-->

I want a conservative, long-only trend-following system on large, liquid US names — broad index
ETFs and mega-cap leaders (e.g. SPY, AAPL, MSFT). Favor durable, multi-week trends over fast
intraday moves: enter long only after a trend is clearly established and a healthy pullback has
passed, and step flat on a genuine regime change rather than trying to short the down-leg.

Prefer steadiness to peak return: I would rather a smoother equity curve (hence the Sharpe
election metric) than a higher headline number with violent drawdowns. Avoid thinly-traded or
highly speculative names entirely.
