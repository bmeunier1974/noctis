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

The one metric a candidate is ranked and promoted on (`promotion.metric`). Stated once and
threaded through the whole evaluation pipeline — prefilter coarse ranking, validation,
Scorecard, promotion — by `PipelineConfig.auto()`. Champions scored under a different
election metric are **stale** (displaceable), because cross-metric numbers aren't
comparable. It is the operator's risk appetite, so an operator mandate may bind it — and it
is the *only* `promotion.*` key a mandate may bind: the thresholds beside it are read in the
metric's own units, so they are no more comparable across a metric change than a stale
champion's number is.

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

One session end-to-end (`TradingDay`, `noctis.engine.trading_phase` — the one TRADING module,
beside the `TradingPhase` entry that builds it; decided 2026-07-11): trade the feed →
attribute forward P&L (derived evidence — never blocks) → persist the account **first** → advance
the session high-water mark **second**, identically for live and replay days. A crash between
the last two re-trades the session (safe) rather than silently skipping it. Before the
unification the live path never advanced the mark, so a live-traded day followed by a
replay day was re-traded on the carried account. The settle ends by **folding** the session
straight into the entry's one `TradingOutcome` (equity and positions from the last session,
trades and events across all of them) and handing back the stamped `TradingSummary` that is the
session's own evidence — a replay catch-up folds several sessions into that one outcome, and no
per-session wrapper shape sits in between (decided 2026-08-23).

## Close phase

The one CLOSE entry (`ClosePhase`, `noctis.engine.close`, decided 2026-08-23), run in a fixed
order: tail-only catalog sync → integrity check + repair → reconcile the session's live bars
against the synced catalog → **one** read of the paper account + forward ledger → the day's
equity mark → assemble one frozen report → write both files (`.md`, then `.json` from that one
value) → periodic distillation → reorganize memory. The order is the contract for two reasons:
the reconcile compares against T+1 vendor bars, so it can only follow the sync, and **the day's
events are complete before the report is rendered**, so what the close itself discovers — a
flagged feed drift — reaches both files instead of being appended to a report already on disk.
One `gather_account_forward` parses `paper_account.json` once, so the equity mark and the report
state the same account because they are the same read. Every step is isolated: a failure is
recorded on the `CloseResult`, never fatal, and memory upkeep always runs last.

## Position driver

The one module that walks **one symbol's position** through bars under the fill model
(`PositionDriver`, `noctis.broker.position_driver`, decided 2026-08-22): pending target →
execute at the open → re-anchor exit tracking → evaluate exits → ratchet → `on_bar` → latch →
carry. Two entry points, in a fixed order per bar: `at_open(bar)` (the broker-touching half)
then `at_close(bar)` (the deciding half). The backtest simulator and the live trading day are
its two drivers and differ only in what they pass in — never in the step order, which lives
here alone. **Sizing is a seam** (`Sizer`: how many units a target means at this price; the
backtest's `alloc` formula and the live `RiskManager` are its two adapters; a dead-band skip
or a risk refusal is the live adapter's answer, not the driver's). A driver seeds from the
broker's carried position (`from_position`), so a flat account and a carried one start through
the same constructor. It returns what happened (fill, exit trigger) and never counts, logs or
emits — those belong to the caller.

## Run segment

One process's stretch of work on a run: opened once through `open_segment` (`noctis.bootstrap`,
decided 2026-08-22), closed once with a **reason** (`stopped_reason`) and its counters, always
releasing the run lock — on **every** exit path (normal return, early exit, an exception, a
`typer.Exit`). Opening takes the lock, mints or resumes the run id, freezes the session's inputs,
records every engine-change note, and builds the `--debug` recorder and the event sink. `run` and
`research` are its two drivers and differ only in what they drive and what they say — never in the
lifecycle, which lives in the band alone. `Segment` is the handle the work holds: `finish` is its
**only mutation** (what stopped this segment, and what it measured; called twice, the first reason
wins), and `checkpoint` is the incremental record write the day loop's `on_cycle_close` seam calls.
A body that never reports closes at the `"startup"` sentinel — **measured nothing, never zeros**.
The band never imports Typer, so a live lock leaves as `RunLockedError` and the CLI maps it to red
text and an exit code, once.

## Run tree

One run's directory and everything in it: `workspace/runs/<run_id>/` — `run.json` (the record),
`run.lock` (liveness), and everything that run produced (`state/`, `strategies/{__tmp,champions}/`,
`memory/`, `reports/`, `qa/`), with the derived `runs/index.json` rolled up beside it. A run owns
its state, so two trees in one workspace are two experiments.

`src/noctis/reporting/run_tree/` is the **only** code that touches it (decided 2026-08-23): five
modules over one narrow read, and the `store` that holds them — layered
`record ← {address, index, lock, evidence} ← store`, pinned there by
`tests/test_run_tree_boundary.py`. `record` is the bottom — the tree's names, the one `read_record`,
the one atomic `write`; `address` turns one operator-typed string into one run dir (four forms,
fixed order); `index` derives the listing roll-up, never authoritative; `lock` is the whole liveness
protocol, and the one failure here that is fatal rather than latched; `evidence` is the six
collectors plus `read_engine_identity` and **every** heavy import in the package (`research`,
`champions`, `broker`, `data`, `config.settings`, `pandas` — all deferred), so a record write stays
cheap; and `store` — the lifecycle verbs `open_run` / `finish_run` / `prune_run_state`,
`read_artifacts` and `RunStore` — is the one module that imports the rest. The readers hold nothing
but `record`, which is why resolving `@label` takes no lock and runs no collector.

A record is read in **two halves**, and each verb says which it needs: `read_artifacts` parses the
prior record (every verb does), `derive_evidence` takes the six reads **once**, at write time only.
Beside the package, `reporting/run_record.py` (`build(artifacts) -> dict`) and
`reporting/schema.py` (`validate`) stay pure — that boundary is what the package exists to hold.
The record's own field-by-field contract is [`docs/run-record.md`](docs/run-record.md).

## Reading a run

One process's **look** at a run — its tree, and the inputs the run was frozen with — opened once
through `open_reading` (`noctis.bootstrap`, decided 2026-08-23), never through a raw `load_settings`.
The read-only twin of a **run segment**, and defined by what it does *not* do: **no lock is taken,
no record is written**, nothing is created, and none of `assert_resumable`, `assert_mode_unchanged`,
the engine-change note or the config rebase runs. A reader acts on nothing — so a `completed` run is
readable (terminal is about *working* a run, not reading one), and so is one another engine is live
on.

An **address** names the run (the four forms of `run_tree.resolve_run_dir`), and what the reading
resolves is that run's own frozen inputs — the same recipe `--resume` runs, so a reading of a run
sees what a resume of it would run under. No address means the reserved `legacy` run, read through
the whole precedence chain over the current files (settings → gate when asked → mandate overlay):
what a run minted right now would be told. That is the bug the term exists to prevent — a reader
that stopped at the raw settings labelled a champion board with a `promotion.metric` the run it was
reading had been steered off.

`Reading` is a **value, not a context manager**: nothing was opened, so nothing has to be closed. It
carries the resolved settings, the `run_dir` every collaborator is built from, the address an
operator typed and the record itself — and it *builds* nothing (no registry, no lake, no memory, no
report); the command body assembles its collaborators from `reading.settings`. A **pruned** run is
refused with `RunPrunedError` unless the verb says it can read one (`readable_pruned`: `run-record`,
`--finish` and `run-prune`, whose subject is the record retention kept). The CLI's one wrapper,
`_reading_or_exit`, adds only what a terminal needs — red text through the one refusal table, and
the legacy-layout guard asked of an **unaddressed** reading alone, because that guard's question is
"which tree does this command read?" and an address is the answer.

## Research toolbox

The one object a research session holds for its whole life — and the **surface** every reader of it
holds is `noctis.research.surface` (decided 2026-08-22): tools, derived facts and a counters
snapshot, declared in two tiers. `ResearchFacts` is what a *renderer* reads — a briefing, the system
prompt, an eval site rebuilding a past ask: the champion board, the library index, the market
economics, one candidate's journaled evidence, the four ceilings as one frozen `limits` value, and
nothing that could change the session. `Toolbox` (⊃ `ResearchFacts`) is what a *driver* holds: the
same facts, plus the tool registry it dispatches through, the capture seams and `session_counters()`
— a snapshot, because the live counters keep moving under a reporter that held a view of them.
Nothing a consumer reads is a collaborator: the briefings and the prompt render *facts*, the toolbox
owns *where they come from* (journal, lake, registry, memory, library tiers), and a reach through it
for one — `toolbox.journal`, or a `getattr` probe that invents the fact when it misses — is what
`tests/test_toolbox_boundary.py` refuses. A test double conforms to the Protocol, never to the
implementation.

## Fingerprint ratchet

A committed statement of what part of the repo *is*, plus a rule that refuses an undeclared change
to it. **One mechanism, two policies** (`noctis.observability.ratchet`, decided 2026-08-23): the
shared module owns everything but the rule — the record built from the tree, loaded and written,
the check, `--write`, the report — and a policy is a `RatchetSpec` plus the one judging callable
that *is* its rule. `engine_ratchet` judges on the arbiter/searcher tier and the declared
`ENGINE_VERSION` (arbiter drift fails, searcher drift warns and passes); `prompt_ratchet` judges on
the declared-change rule over `docs/prompt-changelog.md` (the newest entry must name the drifted
site **and** post-date the record; nothing here warns). **Two records on two clocks** —
`engine_fingerprint.json` and `prompt_fingerprint.json`, one command each
(`scripts/engine_fingerprint.py`, `scripts/prompt_fingerprint.py`) — because prompts and arbiter
behaviour drift independently. `--write` regenerates every case except the one its policy exists to
catch (an undeclared arbiter move, an undeclared prompt change): regenerating is the advice every
failure prints, so it can never also be how one gets recorded. **One null rule**,
`engine_id.compare` over `name → digest` maps: a name present on one side only moved (a fingerprint
surface *appearing* is the news), two nulls did not, a non-string reads as null — shared with the
resume policy `engine_change`, so "may this change land" and "may this run continue" can never
answer one edit differently.
