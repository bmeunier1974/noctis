# Research

How Noctis researches while the market is closed, and what a strategy must survive to be
promoted. The design principle throughout: **discipline is structural, not prompted** — gates
and seams enforce honesty, not instructions to a model.

## The night loop

With an LLM configured, RESEARCH is an agent session (`research.mode: agent`, the default): the
model authors and revises real one-file Python strategies in the gitignored `strategies/__tmp/`
working area and drives the whole loop through curated tools —

> **formulate** a thesis → **match** symbols to it → **optimize** until the parameter space is
> exhausted → **decide**: challenge the champion, revise, or reject

— with every trial journaled. The toolbox: `list_strategies`, `get_strategy`, `list_symbols`,
`preview_bars`, `screen_symbols`, `get_champions`, `get_experiment_log`, `ensure_data`,
`write_strategy`, `run_backtest`, `run_sweep`, `evaluate_vs_champion`, `reject_strategy`.

**A session always belongs to a run.** Whether it comes from the night loop or from a standalone
`noctis research`, the session reads and writes *one run's* tree — its champions, paper account,
experiment journals, strategy `__tmp/`/`champions/` tiers and per-run `MEMORY.md`. A bare
`noctis research` mints its own run; `noctis research --resume <address>` appends a research-only
segment to an existing one, under that run's frozen config, so a night of standalone research
accumulates into the same record (and the same research hours and trial count) as a night of
`noctis run` — [cli.md](cli.md#a-research-session-belongs-to-a-run--research---resume-address).
None of the discipline below changes with the verb: the gates are the run's, not the command's.

The structural gates:

- **Validation-on-write.** `write_strategy` validates in an isolated interpreter (import +
  smoke replay + scenario replay + signals/on_bar parity) — a broken file can never land.
- **Exhaustion.** Every trial auto-journals to the run's `state/experiments/<name>.jsonl`; the verdict
  tools refuse until ≥ `research.min_trials` distinct parameter sets (or one completed sweep)
  have been journaled.
- **Aggregates only.** Backtests return scorecards, not bar-level results; previews never cross
  into holdout bars.
- **Budgeted spend.** Data fetches sit behind the cost preflight (see [data.md](data.md)).

On promotion the file is moved out of `__tmp/` into the gitignored `strategies/champions/`, the
winning parameters are written back as its `Params` defaults, and the header is stamped
`status: champion` / `tuned: <date>`, so `noctis backtest <name>` replays exactly what shipped.
A champion file is immutable after that — improving one means authoring a new name. Committed
seed files at the `strategies/` root are never mutated in place. The strategy-file contract and
the three-tier layout are documented in `strategies/README.md`.

## Draft housekeeping and session-end honesty

An authored draft has three exits, not two: promote, reject, or **archive**. Not every strategy
the driver writes reaches a verdict — a session can exhaust its budget, or the loop can stop, with
a file still `draft`/`candidate` in `__tmp/`. Two seams keep that honest instead of letting
undecided drafts accumulate silently.

**Prune-on-start.** On each research-session assembly, before the library loads, a sweep **moves**
every still-undecided top-level `__tmp/` file whose mtime predates `research.draft_ttl_hours`
(default 48h; `null`/`0` disables — see [configuration.md](configuration.md)) into an
`__tmp/archive/` subdirectory. It moves bytes verbatim — no re-stamp, no rejection record, no gate
(AGENTS.md rule 2) — capped at 50 with the oldest evicted, so a fresh session never inherits a
stale draft it abandoned days ago. When anything is archived, the count and names log at INFO
(`pruned N stale working-tier draft(s) …`). The experiment journals under
the run's `state/experiments/` are untouched and stay the ground truth for what was tried.

**Session-end honesty.** However a loop exits, any strategy authored but never carried to a verdict
is left undecided. The session names them in a WARNING (`… will be archived after the TTL`) and
records the sorted list on `ResearchSummary.undecided`, so the abandonment is visible, not silent:
`noctis research` prints it (`Left undecided (N): … — archived after the TTL`) and the CLOSE
report's Research rollup lists them under *Undecided (authored, no verdict)*.

## The coder-model split: brief in, validated file out

`write_strategy` demands a complete strategy file that survives fresh-subprocess validation in
one shot — the one job a cheap or local driver thrashes on. Setting `research.agent.coder_model`
(see [configuration.md](configuration.md)) splits the role: the **driver** keeps the thesis and
the protocol, and a dedicated **coder** model does nothing but turn a structured *brief* into one
validated file. The driver never writes source; the coder never invents edge.

The brief is the division-of-labor guard. In coder mode `write_strategy` swaps its `source`
field for a required `brief` object whose required parts — `thesis`, `entry_exit`, `param_space`,
and a `scenarios` sketch — force the driver to commit the research *before* any code exists. A
brief can't degenerate to "write me something profitable"; if it could, research would have
silently moved to the coder and the split would be fake. The schema switch is total: the driver
only ever sees **one** authoring mode. Without a coder, `source` stays required and nothing
changes; with one, `brief` is required and `source` stays optional — a capable driver can still
hand-write a revision.

Authoring is a stateless, single-completion loop. Each job gives the coder a fresh prompt (the
strategy contract, the brief, and — when the brief names a `reference` — that library strategy's
full source to adapt, or the current file's source when the name already exists, as a revision
request), makes one tool-free completion, and flows its output through the exact same
`library.write_strategy` gate every write passes. The coder runs with **thinking on** by default
(`research.agent.coder_thinking`) — authoring is the reasoning-heavy sub-task, so it reasons
through the scenario-window and warmup arithmetic instead of repeating an error it was just shown;
the (enlarged) system prompt is prompt-cached where the provider supports it, so the private
retries below re-read it rather than re-paying it. Authoring completions **stream** where the
provider can, so the transport timeout bounds silence between chunks rather than the whole
thinking+generation wall clock, and a thinking coder's output ceiling grows by a fixed thinking
allowance so the file's token budget survives the reasoning that authors it (#98). The paid
*escalation* fallback (`coder_fallback_model`, [configuration.md](configuration.md)) runs its own
thinking dial, **off by default** — the strong model spends its whole ceiling on the file (#98). On a validation failure the coder is re-prompted
privately with the error, up to two retries; those retries are invisible to the driver. When the
retries are spent the last gate error comes back as a **repairable code bug** — refine the brief
and resubmit the *same* name — never as a verdict on the thesis. Validation stays the sole arbiter
of what lands: a revision that never validates leaves the previous version untouched, and an
unknown `reference` is rejected before any completion is spent.

Coder completions are Class-B spend, bounded by `research.agent.max_author_calls` (per profile
`20` / `12` / `6`; the `cost_profile` scales it with the rest). The toolbox counts every
completion — private retries included — and refuses to *start* a new brief-authoring job once the
budget is spent, telling the driver to revise by hand or proceed to a verdict; the hand-written
`source` path is never gated. Each completion emits an `author` event (`✎`) — coder model,
strategy name, attempt number, validation outcome — so `noctis research -v` shows authoring
happen instead of a silent gap where a file appears from nowhere, and the coder-call count lands
in the session summary beside the backtest count.

If the coder's provider key or `[llm]` extra is missing, the split degrades **loudly** to
driver-authored mode at composition time — a warning, and the session still assembles with the
driver writing source itself. Nothing else moves: this is purely an authoring seam, so the
promotion gates, the exhaustion floor, journaling, and both out-of-sample holdouts are untouched.

## The fixed oracle: the machine simulates, the model reacts

A strategy's own `scenarios()` are its correctness oracle — known-outcome tapes the write gate
replays to prove the code does what its thesis claims. Letting the *coder* author code, tape, and
assertion windows together is the self-fulfilling-oracle trap: it can quietly draw the target
around whatever the code happened to do. The **episodic** driver closes that by inverting who
authors the tape — the machine fixes the oracle, the model only reacts to it.

**FORMULATE emits a structured `scenario_spec`, not prose.** The FORMULATE episode returns a
`SpecSuite` (`src/noctis/strategies/scenario_spec.py`) in a small fixed vocabulary — the model
reasons about tape *shape* and one behavior per tape and **never writes a bar index**:

- **Leg kinds** — the segment builders each tape is made of: `flat`, `trend`, `selloff`,
  `recovery`, `chop`, `vol_spike`, `gap`. A leg carries a `kind`, a decision-bar **length**
  (`bars` — a length, never an index; `0` for a `gap`), and its shape numbers (`pct` for
  trend/selloff/recovery/gap; `amplitude`/`period` for chop/vol_spike).
- **Behavior tags** — the one thing a scenario asserts: `enter_long_during_leg`,
  `enter_short_during_leg`, `hold_long_through_leg`, `hold_short_through_leg`,
  `flat_by_end_of_leg`, `never_trade`. A directional tag names its target leg by index;
  `never_trade` targets none.
- **Suite shape rules** — 2–8 scenarios, unique names, at least one directional entry and at least
  one `never_trade` tape, each compiling to 60–2000 bars.

The pure, warmup-parametric compiler (`compile_spec`) derives every assertion window from the leg
lengths and the strategy's declared warmup — the model authors none of that arithmetic.

**Compile-failure re-prompt.** FORMULATE compiles the spec at parse time (a structural validity
check) as it parses the episode; a spec that violates a shape rule surfaces the compiler's precise
`SpecError`, and the model re-formulates through the same schema-misfire path a missing field takes
— the message folded into the correction so it fixes the spec on the re-prompt.

**The fixed-oracle AUTHOR brief.** On the spec path the coder is briefed against the fixed oracle,
rendered faithfully by `describe_spec` (tape shapes, behaviors, target legs — no bar index), and
told **not** to author a `scenarios()` method: the write gate stamps one from the spec and rejects
any `scenarios()` the coder writes. The coder changes only the trading logic (`on_start` / `on_bar`
/ `param_space` / `Params`) to satisfy the fixed tape. Every private retry re-authors that logic
against the *same* oracle — the tape and behaviors never move — and an **escalation** to the paid
fallback coder (`coder_fallback_model`, [configuration.md](configuration.md)) inherits the
identical brief and spec, so a candidate is judged against one unchanging target end to end.

**The needs-more-history exit.** Each tape is preceded by a flat setup pad sized to the strategy's
own declared warmup, so a modest warmup always fits. A warmup too large for the fixed tape (the
compiled scenario overruns the 2000-bar maximum) is rejected with an actionable message, and the
driver ends that strategy in a **refined-brief** outcome — the honest move is a lighter thesis that
needs less history, never a bent gate or a stretched tape. The next FORMULATE round can propose one.

**Machine-stamped `scenarios()`.** On success the gate stamps a warmup-parametric `scenarios()`
block into the installed file — it embeds the spec and re-derives the identical oracle at runtime —
so the file stays the whole artifact and `noctis backtest <name>` replays exactly what was gated.
See `strategies/README.md` for the stamped-block contract and the tolerant-both write gate.

**Auditability.** The AUTHOR stage names the spec's scenarios on both the session ledger and the
live `-v` narration, so an operator can see which fixed oracle each candidate was gated against —
carried through to the per-candidate trail in the CLOSE report's research rollup. A gate-rejected
coder attempt persists that oracle alongside the observed-behavior diagnostics (what the code
actually did on the tape) in the capped `__tmp/failed/` store, so a post-mortem shows both the
target the code missed and the miss.

## Panel research: out-of-sample on two axes

Research is cross-sectional, not single-symbol. Every candidate is tuned and validated on a
**fit set** of the first `research.fit_set_size` ready universe symbols (identical split
geometry per symbol; scores are panel means), while the next `research.symbol_holdout_size`
ready symbols form a **symbol holdout** — fixed for the whole run, never used in tuning or
selection, scored once per candidate.

The temporal axis is enforced by walk-forward splits: an execution-realistic backtest (decide
on bar *t*, fill at bar *t+1*'s open) whose test windows sit strictly after their train
windows, plus a most-recent **temporal holdout** slice the search never touched.

### The structural screener: the thesis picks the kind, the data picks the tickers

Symbol selection is grounded by a deterministic feature screen
(`src/noctis/research/symbols.py`, surfaced as the `screen_symbols` tool). The agent states
the *character* a thesis needs — trend, volatility, and liquidity bands — and the screener
maps it to lake symbols using bar-derived features only: trend efficiency (Kaufman ratio),
annualized realized volatility, and daily-equivalent dollar volume, banded low/medium/high
relative to the pool and computed on **training-window bars** (the forward-holdout tail stays
unseen, exactly as in `preview_bars`). The same per-symbol `character` numbers are inlined
into the session's MARKET REALITY digest, so profile choices are grounded before the first
tool call.

The guardrail: the screen reads **structure, never strategy PnL** — picking symbols where a
strategy already shows profit is the cross-sectional twin of lookahead. A match is evidence of
character, not of edge; edge is still decided by the gates. `screen_symbols` also proposes
`reserved_holdout` names from the same matched pool, which the agent keeps out of all tuning
so it can nominate them as `holdout_symbols` at verdict time (the toolbox refuses any nominee
that appears in the strategy's experiment journal).

## Promotion

Promotion (`src/noctis/champions/promotion.py`) is a pure decision function over scorecards.
The gate order:

1. activity floor
2. overfit gap guard (train−test)
3. forward temporal holdout (`promotion.min_holdout_metric`)
4. symbol holdout (`promotion.min_symbol_holdout_metric`)
5. consistency breadth (`promotion.min_symbol_consistency`, optional)
6. beat the weakest current champion

Comparison is on a scale-free footing, and a champion scored under a *different* metric is
treated as "stale" (displaceable) because cross-metric numbers aren't comparable. A candidate
that fails is a signal, not a bug: the answer is a better thesis or an honest
`reject_strategy` — never a loosened gate.

The full methodology — each gate's semantics and config knob, how the two out-of-sample axes
are constructed, and how every champion is made reproducible — is written up in
[validation.md](validation.md).

## Provider-neutral, and free at the limit

The agent talks to one neutral seam, so the model is a config line: `research.model` takes a
LiteLLM `provider/model` string — any hosted provider, or a local / self-hosted backend
(`ollama/…`, `vllm/…`, or any endpoint speaking the standard chat-completions protocol via
`research.base_url`). Hosted keys resolve per prefix from `.env` (the matching `*_API_KEY`); a
**local backend needs no key and costs $0/token**.

Provider-specific levers capability-gate to clean no-ops: prompt-cache breakpoints, reasoning
effort, and thinking apply only where supported, and server-side `web_search` auto-disables on
a provider that can't serve it (optional grounding degrades; no gate, holdout, or journal entry
depends on it). Known per-model quirks are pinned at selection time so swapping models never
silently changes spend.

## One cost knob, never a hidden throttle

`research.cost_profile` (`full` / `balanced` / `economy`) scales the research budgets — tool
rounds, backtests, sweep trials, coder-model author completions, web searches, reasoning effort,
prompt-prefix trim — together, and those ceilings live in a single profile table
(`src/noctis/research/cost.py`), never
hardcoded lower anywhere else. `balanced` (the default) is exactly the standard ceilings;
`economy` reduces spend; `full` restores the maximums and is the automatic choice on a
free/local provider. The knob binds *resource ceilings only*: it can never lower the
`min_trials` exhaustion floor or touch a promotion gate — those are quality, not cost.

## What a session spent, and what it bought

Both loops end a session by reporting what it burned: `ResearchSummary.tokens_total`, the same
tokens **split** the four ways they are billed (`usage`: input / output / cache-write / cache-read),
and `usd_estimate` — those tokens priced through the versioned table in
`src/noctis/research/pricing.py`. Neither loop is instrumented twice for it: the conversation loop
fills the split from the per-round usage it already accumulates, and the episodic driver *sums the
ledger it already wrote* (one `episode` line per judgment, carrying its own stage, model and
split). That is what lets the run record re-derive the same numbers from the same lines at write
time instead of trusting a counter to survive a restart.

The run record turns those lines into an attribution — spend **by model, by stage and by segment** —
plus the efficiency numbers a run is actually compared on: USD per champion, USD per trial, trials
per hour, research hours per champion. Every ratio is `null` when its denominator is zero or
unknown; a run that has crowned no champion yet has no cost per champion, and that is the normal
state of a young run, not an error.

Two honesty rules apply throughout, and they are the reason the block is worth reading at all:

- **`null`, never zero.** A model the price table does not carry costs `null`, and so does any
  total it belongs to. A run with no LLM key journals no ledger and reports `null` spend — never a
  `$0` bill it did not earn. A ledger written before the split existed reports its token *total*
  and `null` for the four fields, because tokens without their split cannot be priced.
- **Every dollar figure says `estimate`** — in the record's field names and in the CLI line — since
  the prices are list prices, not receipts. See
  [configuration.md → Pricing the spend estimate](configuration.md#pricing-the-spend-estimate) for
  the table, the config override, and how an overridden table identifies itself.

Note what is *not* in the figure: authoring runs on a separate coder client whose completions are
not token-metered, so the estimate covers the **judgment/driver model** only — the same boundary
`tokens_total` has always drawn. Vendor data spend is tracked separately by the data preflight
(`data.budget_usd`).

## Mandates + a growing universe

A human steers agent sessions through the `mandate/` folder — the ownable input surface:
`MANDATE.md` (your own first-person brief, gitignored — copy it from the committed
`MANDATE.md.example`, a balanced Sortino swing brief for liquid US large/mid-caps), five
shipped `profiles/` personalities (`aggressive`, `conservative`, `long-term`, `short-term`,
`sector-specialist`), and small supporting `references/`. Only the
scaffold is committed; your own mandate, custom personalities, and personal references stay
local so steering never pollutes the repo. `research.mandate` selects which governs a run — a profile name,
`MANDATE`, `auto` (the agent picks per session — the shipped example config's default), or
`null` (unconstrained). For one session,
`--mandate <name>` or an inline `--directive "<text>"` wins over the config selector (the two
flags are mutually exclusive).

A mandate **configures the run it steers**: its front-matter `config:` block overlays the
run-shaping settings — which model thinks (`research.model`, the coder split, the loop), what
one session may spend (the Class-B ceilings, the wall-clock budgets), how big its prompt gets,
how much history its names are fetched over, the seed `universe`, and `promotion.metric`, the
risk dial it started as. A mandate written for a local 30B coder and one written for a hosted
frontier model are different research *personalities*, and the personality file is the honest
place to say which brain it needs. What it may **not** touch is the arena: the safety mode, the
fill costs, the promotion thresholds, the fit-set/symbol-holdout geometry, the state paths, the
secrets are all refused by name, and a refused, unknown, or invalid key is **fatal at startup**
with its reason printed — never a warning under a multi-day run. Two knobs move one way only:
`research.min_trials` raises (more evidence per verdict, never less) and `data.budget_usd`
lowers (less of your vendor money, never more). A `--metric` CLI flag, when passed, still wins
over the overlay. The full tier tables, the precedence chain, and the refusal reasons live in
[configuration.md](configuration.md#the-mandate-overlay); the whole surface also ships
commented-out in `mandate/MANDATE.md.example`.

Under the shipped `research.mandate: auto`, the agent picks its profile *mid-session*, long
after settings are assembled — so an auto-selected profile's `config:` block never reaches the
overlay at all. Startup warns and names every profile whose keys would be lost, with the remedy
(pin the mandate); `promotion.metric` alone is suppressed from that warning because `auto` is
metric-neutral by contract. The five shipped profiles are therefore deliberately kept
**metric-only**: a profile `auto` may pick must never declare a knob that would be silently
inert on the default config, and a second key in a shipped profile would make every stock
install warn.

**Two ticker surfaces, two different jobs.** A mandate's front-matter `symbols:` is a **search
prior**: those names join the session's research focus set (the prompt's market digest, the
holdout candidate pool, the episodic driver's session-start fetch) — what to *look at*. A
mandate's `config: universe:` is the **seed trading roster** — what is *traded*, and what the
research panel is drawn from. Both normalize identically (upper-case, de-duped, first-mention
order), and the roster carries a guard the prior does not need: it must name at least
`research.fit_set_size + research.symbol_holdout_size` symbols, or the run stops rather than let
the symbol-holdout gate go inert for want of names ([configuration.md](configuration.md#the-mandate-overlay)).

To satisfy a profile the configured universe lacks, the agent **discovers symbols**. On the
conversation loop that is the agent's own tool sequence: `web_search` → `preview_bars` →
`ensure_data` (budget-gated) → `screen_symbols` to confirm
the fetched names actually express the requested character. Every fetched symbol joins the
**effective universe** permanently (config seed ∪ lake-tracked ready symbols — the lake is the
store), so discovered names are researched, holdout-checked, and traded like any other. At
verdict time the agent may nominate `holdout_symbols` it deliberately kept out of all tuning;
the toolbox refuses any name found in the strategy's experiment journal. See
`mandate/README.md` to author your own or pick a shipped profile.

Those declared `symbols:` are honored the same way on **both** loops. The episodic
driver opens each session with a deterministic **PREFLIGHT** stage (#111, no model call): every
declared symbol the lake cannot research yet is fetched in one `ensure_data` call over the
`data.history_days` window ending at the session date (its end sits at the vendor's T+1
boundary, like `run`'s auto-backfill), so the first screen, fit set, symbol-holdout reservation,
and fallback panel already see the operator's names. The spend rides the same cost preflight
against `data.budget_usd` — steering can never bypass the data budget — and a refusal or fetch
error is ledgered on the stage's own `preflight` line (per-symbol status, rows, cost, plus the
session's total data spend) while the session continues on the lake it already has. A session
with no mandate, or one whose declared names are all lake-ready, makes zero `ensure_data` calls
and behaves exactly as before. No new knobs: `data.history_days` and `data.budget_usd` keep
their meanings, now honored by both loops.

The episodic driver also **discovers** — where it used to fall back in silence. When a thesis's
structural screen finds no lake match, MATCH no longer just researches the default panel: the driver
spends one small **DISCOVER** episode (#112) asking which real tickers express the character the lake
lacks (`{"symbols": [1–6 tickers], "rationale": "…"}`, the smallest of the three emit contracts, in a
briefing built from the mandate, the thesis, the band profile that failed to match, the lake
inventory, and the spend context). Everything after that answer is deterministic code: a pure
validator (upper-case, ticker shape, dedupe, drop names the lake already holds) filters the proposal
**before a dollar is spent**, the survivors are fetched in one budget-gated `ensure_data` call over
the same `history_days` window, and **exactly one re-screen** decides which of them are tuned and
which are reserved as the symbol holdout — so a discovered name can only enter research through the
screen's own fit/reserved split, and rules 2–4 are untouched. Failures degrade honestly, each with
its own ledgered reason: one corrective re-ask on a misfired or all-invalid proposal then
`discover_failed`, a `discover_refused` / `discover_fetch_failed` fall-through when the data budget
or the vendor says no (a refusal is not the episode's fault, so it spends no re-ask), and
`no_lake_match_after_discover` when the names landed but the re-screen still matched nothing. The
attempt is one line in the session ledger — tickers proposed / kept / fetched, the window, the
per-symbol status and cost — so the CLOSE rollup tells a session that *discovered* from one that
*fell back*. Discover episodes count against the session's `max_episodes` through the same
completions counter, and one is never started when that budget is already spent. There is no
`web_search` in v1 (the model's own knowledge of the universe carries the proposal) and no new knob;
a session whose screens match behaves exactly as before.

## Two agent loops, one contract

With a client, the same protocol runs one of two ways behind one seam (`research.agent.loop`): the
**conversation** loop (one long tool-use transcript) or the **episodic** driver (a deterministic
state machine that calls the model only at narrow judgment points and keeps the cross-strategy story
in a session ledger, not a growing chat context — built for small-context backends). Both return the
same `ResearchSummary`.

`auto` (the default) is an **evidence-gated flip** (#76): it selects the episodic driver when the
operator has declared a `research.agent.context_window` of at most **32,768 tokens**
(`_EPISODIC_WINDOW_MAX` in `noctis.bootstrap`, inclusive so the canonical 32k noctis-ollama box
flips), and the conversation loop for larger or unset windows (hosted backends). The evidence is
the **parity harness** (`scripts/parity_harness.py`), which runs both loops on the same model,
fixture, and mandate and reports verdicts/session and tokens/verdict side by side — see
[parity.md](parity.md). The flip criterion read **PASS** on 2026-07-23: episodic held
verdicts/session at 45% fewer tokens/verdict, and subsequent leak fixes (#89, #90) cut its
tokens/verdict a further ~4× on the same fixture. Loop selection lives in one place
(`bootstrap.resolve_research_loop`); an explicit `conversation`/`episodic` always wins over `auto`.

The conversation loop carries a **zero-verdict liveness guard**: a prose-only reply while nothing
has been decided is a protocol stall (the FORMULATE step invites exactly that turn shape), so the
loop keeps the prose as context and nudges the model onward, at most twice per session. Past the
cap a prose turn still ends the session — never an infinite nudge loop — but a zero-verdict ending
is reported as `stopped_reason: prose_stall`, not `agent_done`: with the corrections spent there is
no way to tell a deliberate empty conclusion from a model stuck narrating (#100 watched a hosted
model do exactly this for three straight sessions), so the summary carries the honest, countable
name. The episodic driver needs no such guard — its episode contract forces a structured emission,
so it cannot stall in prose by construction.

## The legacy fallback

No configured client (missing key or missing `[llm]` extra) → the legacy proposer/Optuna loop
runs over the same strategy library and returns the same `ResearchSummary`. The legacy
`StrategySpec` engine (`src/noctis/strategies/spec/`) is strategy-as-data: an LLM-minted JSON
graph compiles to a registerable family, persists to the state dir's `specs.json`, and
re-registers at startup.
