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

The structural gates:

- **Validation-on-write.** `write_strategy` validates in an isolated interpreter (import +
  smoke replay + scenario replay + signals/on_bar parity) — a broken file can never land.
- **Exhaustion.** Every trial auto-journals to `workspace/state/experiments/<name>.jsonl`; the verdict
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
`workspace/state/experiments/` are untouched and stay the ground truth for what was tried.

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
retries below re-read it rather than re-paying it. On a validation failure the coder is re-prompted
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

A mandate **carries its own risk dial**: its front-matter `config:` block may set
`promotion.metric` (`sharpe|sortino|total_return`) — and nothing else. A `--metric` CLI flag,
when passed, wins over the overlay. A mandate steers *what to look for*; it never loosens a
gate, the exhaustion rule, or the honesty contract.

To satisfy a profile the configured universe lacks, the agent **discovers symbols**:
`web_search` → `preview_bars` → `ensure_data` (budget-gated) → `screen_symbols` to confirm
the fetched names actually express the requested character. Every fetched symbol joins the
**effective universe** permanently (config seed ∪ lake-tracked ready symbols — the lake is the
store), so discovered names are researched, holdout-checked, and traded like any other. At
verdict time the agent may nominate `holdout_symbols` it deliberately kept out of all tuning;
the toolbox refuses any name found in the strategy's experiment journal. See
`mandate/README.md` to author your own or pick a shipped profile.

## The legacy fallback

No configured client (missing key or missing `[llm]` extra) → the legacy proposer/Optuna loop
runs over the same strategy library and returns the same `ResearchSummary`. The legacy
`StrategySpec` engine (`src/noctis/strategies/spec/`) is strategy-as-data: an LLM-minted JSON
graph compiles to a registerable family, persists to the state dir's `specs.json`, and
re-registers at startup.
