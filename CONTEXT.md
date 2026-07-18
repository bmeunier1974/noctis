# CONTEXT.md — domain glossary

Ubiquitous language for Noctis. Code, docs, and reviews use these terms with exactly
these meanings. Sharpen an entry here the moment a conversation sharpens the concept.

## Panel

The set of **fit symbols** a candidate is tuned and validated across. Research is panel
research: identical split geometry on every symbol, aggregates are panel means. A single
symbol is a **panel of one** — not a different mode, just the smallest panel.

## Scorecard

The currency of promotion: per-split and aggregate out-of-sample metrics for one candidate.
**A Scorecard is always a panel** (decided 2026-07-11): per-symbol splits live under
`symbols`, and `avg_train_metric` / `avg_test_metric` / `gap` / `test_activity` have exactly
one meaning — panel means (a panel of one reproduces the old single-symbol numbers).
Legacy persisted cards (top-level `splits`, `symbols=None`) are normalized on read into a
panel of one under the sentinel symbol `"*"`; the sentinel never binds eligibility.

## Election metric

The one metric a candidate is ranked and promoted on (`promotion.metric`; the only knob a
mandate may bind). Stated once and threaded through the whole evaluation pipeline —
prefilter coarse ranking, validation, Scorecard, promotion — by `PipelineConfig.auto()`.
Champions scored under a different election metric are **stale** (displaceable), because
cross-metric numbers aren't comparable.

## Evaluation pipeline

The funnel `evaluate(candidate, panel, config) → Scorecard`: coarse pre-filter (median
kill across the panel; a filter, never a promoter) → walk-forward validation → temporal
holdout → symbol holdout → one panel Scorecard. Takes only a panel
(`dict[symbol, DataFrame]`); callers with one symbol pass a panel of one.

## Forward holdout (temporal holdout)

The most-recent bars carved off before any search touches the data, scored once at the
end. The backstop against selection/lookahead bias. `holdout_metric` on the Scorecard;
gate 3 in promotion.

## Symbol holdout

Symbols never used in tuning or selection, scored with one causal pass each. The
cross-sectional twin of the forward holdout. `symbol_holdout_metric`; gate 4 in promotion.

## Fit symbols / live symbols

Eligibility, bound once at promotion on the `ChampionEntry` (not on the Scorecard): a
champion trades live only the symbols it was fit on (`live_symbols = fit_symbols`).
`fit_symbols = None` marks a legacy pre-panel champion, eligible everywhere — new
promotions always bind to their panel, including panels of one (decided 2026-07-11: a
strategy validated on one symbol is not validated on the rest of the universe).

## Bar feed

The one contract the TRADING driver drinks its minutes from (`BarFeed`:
`symbols` / `degraded` / `exhausted` / `poll_once` / `flush`), with two adapters: the live
yfinance feed is **clock-bounded** (never exhausted — the session close ends the day; delay
is normal, staleness degrades) and the catalog `ReplayBarFeed` is **data-bounded** (its
slice's exhaustion ends the day; never degraded on its own; nothing held back to flush).
Live and replay differ only in where the minutes come from, never in how they are traded.

## Experiment journal

The durable, append-only record of research evidence: one JSONL file per strategy under
`state/experiments/`, one line per event (`trial` / `sweep_complete` / `class_tag` /
`verdict`). The journal — never the agent's context — is the ground truth the research
discipline reads: the exhaustion gate counts its distinct param sets, the symbol-holdout
taint check scans its trial symbols, `reject_strategy` recovers its best-observed params.
`ExperimentJournal` (`noctis.research.journal`, decided 2026-07-11) owns the record schema
end-to-end — explicit `record_*` writers, typed reads — so no caller re-parses `event`
strings by hand.

## Trading day (the settle order)

One session end-to-end (`TradingDay`, decided 2026-07-11): trade the feed → attribute
forward P&L (derived evidence — never blocks) → persist the account **first** → advance the
session high-water mark **second**, identically for live and replay days. A crash between
the last two re-trades the session (safe) rather than silently skipping it. Before the
unification the live path never advanced the mark, so a live-traded day followed by a
replay day was re-traded on the carried account.
