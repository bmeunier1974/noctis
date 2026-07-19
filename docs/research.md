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
`MANDATE.md.example`), five shipped `profiles/` personalities (`aggressive`, `conservative`,
`long-term`, `short-term`, `sector-specialist`), and small supporting `references/`. Only the
scaffold is committed; your own mandate, custom personalities, and personal references stay
local so steering never pollutes the repo. `research.mandate` selects which governs a run — a profile name,
`MANDATE`, `auto` (the agent picks per session), or `null` (unconstrained). For one session,
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
