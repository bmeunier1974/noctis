# Validation

This is the document behind Noctis's central claim: that it can tell a *real* edge from noise.
Every guarantee here is enforced **structurally** — by gates and seams in the code — not by
prompting or reviewer discipline. If you only read one page to decide whether to trust the
project, read this one.

The short version: a candidate strategy is tuned on one slice of data and must then survive
**out-of-sample tests on two independent axes** before it can displace a champion, and it is
compared to incumbents on a **scale-free footing** through a fixed **gate order**. A strategy
that can't clear the gates is not a bug to be worked around — it is a signal, and the honest
response is a better thesis or a rejection.

## The pipeline, end to end

```
Data (fetch-once lake)
  → Research agent (formulate → match → optimize → decide)
  → Strategy generation (one reviewable .py, validated on write)
  → Backtesting (vectorbt-style pre-filter → walk-forward validation)
  → Out-of-sample validation (temporal holdout + symbol holdout)
  → Promotion (the gate order below)
  → Forward paper record (champions trade only bars no tuning ever saw)
```

The rest of this page zooms in on the validation and promotion stages — the parts that make
the forward record credible. For the research loop itself see [research.md](research.md); for
the phase loop and modules see [architecture.md](architecture.md).

## Out-of-sample on two axes

Research is **panel** research: a candidate is tuned and validated across a *set* of symbols,
not one. Two out-of-sample axes are carved out **before** any search touches the data and are
scored once per candidate:

- **Temporal holdout (forward holdout).** The most-recent slice of each symbol's history,
  sealed off before tuning and scored last. It is the backstop against lookahead and
  selection bias — a strategy that was implicitly curve-fit to the past has no reason to work
  on bars the search never saw. Surfaced as the Scorecard's `holdout_metric`.
- **Symbol holdout.** A set of symbols (`research.symbol_holdout_size` of them) that are never
  used in tuning or selection at all — the cross-sectional twin of the temporal holdout. A
  strategy validated on AAPL/JPM/NVDA must also show something on names it was never fit on
  before it earns a slot. Surfaced as `symbol_holdout_metric`.

The fit set (`research.fit_set_size` symbols) is where tuning happens; scores across it are
**panel means**, so a single lucky symbol can't carry a candidate. A candidate validated on a
panel of one is bound to that one symbol only — it is *not* assumed to generalize to the rest
of the universe.

## No lookahead, ever

The temporal axis is enforced by **walk-forward splits** (`src/noctis/backtest/splits.py`):
every test window sits strictly *after* its training window, and the most-recent temporal
holdout sits after all of them. Both backtest stages decide on bar *t* and fill at bar
*t+1*'s open — you can never act on a bar you are still forming. Previews and market digests
shown to the research agent never expose holdout bars. Any change that lets future or holdout
information reach a decision is a correctness bug, full stop.

**The pre-filter is exit-blind, on purpose.** With engine-enforced protective exits (the
fill-model section of [architecture.md](architecture.md)), realized P&L is no longer a pure
function of the target series, and the vectorised pre-filter cannot see exit fills. It keeps
its coarse selection-filter role unchanged; the event-driven walk-forward — authoritative for
the Scorecard and every promotion gate — prices exits exactly. Stops are never approximated
vectorially: that is where lookahead bugs are born.

## The promotion gate order

Promotion (`src/noctis/champions/promotion.py::decide`) is a **pure decision function** over
scorecards — no I/O, no hidden state, easy to test. A candidate first has to be a *validated*
panel card (a pre-filter-killed or structurally-empty card can never promote), then clear a
fixed sequence of **quality gates**, and only then is it measured against the board.

**Quality gates** — each rejects outright on failure:

1. **Activity floor** (`promotion.min_test_activity`). A candidate that barely trades can post a
   flattering metric on a handful of lucky bars, so it must show trading activity on a minimum
   fraction of test splits. *Off by default (`0.0`); opt in to enforce.*
2. **Overfit-gap guard** (`promotion.max_gap`, default `1.0`). The train−test gap
   (`avg_train_metric − avg_test_metric`, both panel means on the election metric) must stay
   within bounds. A large gap is the fingerprint of overfitting — caught before any holdout is
   even consulted.
3. **Forward temporal holdout** (`promotion.min_holdout_metric`). When a temporal holdout was
   reserved, its metric must clear the bar: the candidate has to work on the most-recent bars
   the search never touched.
4. **Symbol holdout** (`promotion.min_symbol_holdout_metric`). When a symbol holdout exists, its
   metric must clear the bar: the candidate has to work on names it was never fit on.
5. **Consistency / breadth** (`promotion.min_symbol_consistency`). The edge must show up across a
   minimum *fraction* of the panel's symbols, not concentrate in one or two. *Off by default.*

**Then the slot logic** — how a survivor takes a seat on the board of `champion_count` (default
`3`):

6. **Free slot.** If the board isn't full, the candidate is crowned as long as it clears the
   minimum bar (`promotion.min_test_metric`).
7. **Stale champion first.** A champion scored under a *different* election metric than the
   candidate is **stale**: its number is in incomparable units, so rather than defend its seat
   with a value that can't be compared, it is displaced like a free slot (again gated only by
   `min_test_metric`). This is a deliberate `metric_name` *string* check, not a cross-metric
   numeric comparison.
8. **Beat the weakest.** Otherwise the candidate must beat the weakest current champion's
   panel-mean test metric to take its slot. Because that comparison is a panel mean on a shared
   election metric, champions with different fit sets still compare on a **scale-free footing**.

The election metric itself (`promotion.metric` — `sharpe`, `sortino`, or `total_return`) is the
*one* knob an operator mandate may bind; it reinterprets every threshold above but can never
loosen a gate.

> **On defaults and honesty.** Several thresholds ship at `0.0`, and two gates (activity,
> consistency) ship off. That is deliberate, not a loophole: the holdout gates at `0.0` still
> demand a **non-negative** out-of-sample result — a strategy that loses money on unseen data is
> rejected — and an operator raises the bars for a stricter regime. The *structure* (two
> independent out-of-sample axes, the gap guard, scale-free comparison) binds regardless of how
> aggressive the numeric thresholds are set.

## A failing strategy is a signal, not a bug

This is the load-bearing cultural rule, and it is defended in code. When a candidate can't clear
the gates, the answer is a **better thesis** or an honest `reject_strategy` — **never** a
loosened gate. Do not raise a gap tolerance, lower a holdout bar, or shrink the holdout set to
make something "pass": that is overfitting with extra steps, and it would quietly destroy the
one thing that makes the forward record worth anything. The gates are the arbiter of quality by
design. (See also the non-negotiable invariants in [CONTRIBUTING.md](../.github/CONTRIBUTING.md) and the
safety boundary in [SECURITY.md](../.github/SECURITY.md).)

## Reproducibility: every champion replays exactly what shipped

When a candidate is promoted, its winning parameters are written **back into the strategy file**
as the `Params` defaults, and the header is re-stamped (`status: champion`, `tuned: <date>`).
Before the crown is granted, that rewrite is re-validated in a fresh subprocess — including the
file's known-outcome `scenarios()` — so tuned params that would break the file are refused
rather than shipped, and a crowned champion is never stranded on a broken rewrite. So the file
on disk *is* the champion — there is no separate, drifting record of "the params we actually
used." Anyone can re-run

```bash
uv run python -m noctis backtest <name>
```

and replay exactly the strategy that earned the slot, on its shipped defaults. Combined with
fixed seeds and versioned catalog snapshots, that makes every promotion decision auditable and
reproducible from the repository alone. The strategy-file contract — including the known-outcome
`scenarios()` that validate on write — is documented in
[`strategies/README.md`](../strategies/README.md).

## Why this adds up to credibility

Take the pieces together: tuning is isolated from two independent out-of-sample axes; no
decision can see the future or the holdout; the gate order rejects the specific failure modes
of quantitative research (inactivity, overfitting, single-symbol luck, cross-metric apples and
oranges) in a fixed, pure, testable function; a failing candidate is rejected rather than
accommodated; and every surviving champion is byte-for-byte reproducible from the repo. That is
the difference between *"a backtest looked good once"* and **validated, reproducible research
infrastructure**.
