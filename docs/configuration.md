# Configuration

Configuration lives in a **local, gitignored `config.yaml`** — created (and edited) by
`noctis setup`, or copied from the committed template `config.example.yaml` by `noctis
init`. A missing `config.yaml` is fine: every knob has a safe built-in default, which is
why the template stays short — it lists only the knobs operators actually touch, and this
page is the full reference. Secrets live in `.env` (same pattern: `.env.example` → `.env`).
Both resolve into typed settings (`src/noctis/config/settings.py`). **Environment
variables override `config.yaml`**, and `NOCTIS_CONFIG=/path/to/config.yaml` points at an
alternate file.

## The knobs

| Key | What it controls |
|---|---|
| `mode` | `paper` (default) or `live` — see [safety.md](safety.md) for the double gate |
| `universe` | Seed symbol list; the *effective* universe grows as the agent discovers symbols ([research.md](research.md)) |
| `session` | Exchange calendar + timezone |
| `research_time_budget_minutes` | Wall-clock cap on a research session |
| `research.mode` | `agent` (default) or `legacy` |
| `research.model`, `research.base_url` | LiteLLM `provider/model` string; base URL for local/self-hosted backends |
| `research.mandate` | Mandate selector: a profile name, `MANDATE`, `auto`, or `null` |
| `research.min_trials` | Exhaustion floor — verdict tools refuse before this many journaled trials |
| `research.max_iterations`, `max_backtests`, `sweep_trials`, `web_search` | Agent session budgets |
| `research.cost_profile` | `full` / `balanced` / `economy` — resource ceilings only, never quality gates |
| `research.agent.thinking` | `off` (default) / `on` — opt a **watch** session into provider-native reasoning; costs output tokens (see below) |
| `research.agent.max_tokens`, `context_window` | Small-context-backend compatibility levers (see **Local backends** below) — not cost budgets |
| `research.agent.sweep_workers` | Parallel workers for sweep trials + panel symbols (`1` = sequential) |
| `research.fit_set_size`, `symbol_holdout_size` | Panel geometry: fit set + symbol holdout sizes |
| `research.focus_size` | Cap on symbols enumerated into each session's prompt — a prompt-size lever, never the trading roster |
| `research.tuning_dispersion_penalty` | Penalizes parameter sets whose panel scores are dispersed |
| `risk` | Trading-loop risk limits (incl. the daily loss limit) |
| `trading.max_catchup_sessions` | Cap on missed replay sessions a restart catches up (newest kept) |
| `trading.min_order_notional`, `rebalance_band_pct` | Rebalance dead-band: skip immaterial same-direction re-trues; opens/exits/flips always execute |
| `trading.execution` | `auto` (derive from `data.provider`) / `replay` / `live` — selects the TRADING driver only, never real-order reachability |
| `data.provider` | Historical source *and* TRADING-phase feed — `yfinance` opts in to the live feed ([data.md](data.md)) |
| `data.budget_usd` | Data spend cap enforced by the cost preflight (default `$125`) |
| `data.dataset`, `data.lake_dir` | Vendor dataset; lake location |
| `data.auto_backfill`, `data.history_days` | Pre-loop backfill of missing history (code default off; **shipped config enables it**, `history_days: 720`) |
| `live_feed.poll_interval_s` | Live-feed poll pacing (the feed self-throttles regardless) |
| `observability.heartbeat_polls` | `-vv` live-trading heartbeat cadence in polls (0 disables; default `60` ≈ every ~2 min) |
| `promotion.metric` | `sharpe` \| `sortino` \| `total_return` — changing it reinterprets every threshold below in the new metric's units |
| `promotion.max_gap`, `min_test_metric`, `min_test_activity` | Overfit-gap guard, OOS bar, and the almost-never-trades activity floor |
| `promotion.min_holdout_metric`, `min_symbol_holdout_metric`, `min_symbol_consistency` | The out-of-sample promotion gates |
| `promotion.annualization_cap`, `max_period_ratio`, `max_reverse_gap`, `max_test_metric` | Metric-robustness caps + degeneracy backstops (sub-daily Sharpe inflation, too-good-to-be-true OOS) |
| `ideation` | The legacy StrategySpec path |
| `champion_count` | Champion board size |
| `time_limit_hours` | Global stop from any phase |
| `workspace_dir` | **The one output root** (default `workspace/`; env `NOCTIS_WORKSPACE`) — every path below derives from it when not set |
| `state_dir`, `reports_dir`, `memory_path`, `data.lake_dir` | Per-artifact overrides; each defaults to its workspace-derived location (`workspace/state`, `workspace/reports`, `workspace/memory/MEMORY.md`, `workspace/data_lake`) |
| `strategies_dir`, `mandate_dir` | The committed input surfaces: the seed strategy library and the mandate scaffold |

## The workspace

Everything the engine writes — run state, the data lake, reports, agent memory, and the
strategy working/champion tiers — lands under the single gitignored `workspace_dir`. One
gitignore line, one thing to back up or wipe. Setting the env var `NOCTIS_WORKSPACE`
relocates all of it at once (useful when running the CLI from outside the repo); an explicit
per-artifact knob is an absolute override. `noctis init` creates the workspace alongside the
local config; `noctis migrate` moves a pre-workspace layout (`state/`, `data_lake/`,
`reports/`, root `MEMORY.md`, `strategies/__tmp|champions`) into it — and every
state-touching command refuses to run beside un-migrated legacy data until it has
(`status` only warns), so a stale layout can never silently present an empty champion board.

## The mandate overlay

The active mandate's front-matter `config:` block may overlay **only** `promotion.metric` —
nothing else. Precedence: `--metric` CLI flag > mandate overlay > `config.yaml`. For one
session, `--mandate <name>` or `--directive "<text>"` (mutually exclusive) override the
`research.mandate` selector. The whole chain resolves in one place — `resolve_session` in
`src/noctis/bootstrap.py`, the composition root — so the ordering can never drift between
commands. Details: [research.md](research.md) and `mandate/README.md`.

## Local backends (noctis-ollama)

Any OpenAI-compatible or Ollama endpoint can serve `research.model` at $0/token —
[noctis-ollama](https://github.com/bmeunier1974/noctis-ollama) turns a bare GPU box into a
verified, agent-ready one with a single `./setup.sh`. `noctis setup` detects a running
server and writes the wiring for you; by hand, it is one block in `config.yaml`:

```yaml
research:
  model: ollama_chat/noctis-qwen3:14b # any tag the server carries; `ollama_chat/` prefix
  agent:
    max_tokens: 4096 # output cap — small-context backends bound prompt+output together
    context_window: 32768 # the model's num_ctx — activates the prompt-trimming levers
```

No API key is needed. The three knobs usually travel together: `max_tokens` keeps a
completion inside the window (a thinking model needs room to reason *and* emit a tool
call), and `context_window` bounds the whole request — per-result caps tier down, the
oldest tool results evict to pointer lines, and a decided strategy's history collapses at
its verdict. Both are compatibility levers, not cost budgets: the on-disk experiment
journal stays the ground truth, so no gate or holdout is affected. A non-Ollama endpoint
(vLLM, a proxy) uses `research.base_url` plus its own model id. On a free/local provider
the `cost_profile` automatically resolves to `full`.

## Watching the model reason

`research.agent.thinking` is a binary, provider-neutral watch switch, **off by default**. Turned
`on`, it opts the session into provider-native reasoning where it exists: for an Anthropic
non-Sonnet model (the fallback `claude-opus-4-8`) it sends adaptive thinking with a summarized
display, so the research loop surfaces `think` events (see [research.md](research.md) on
observability). It is a no-op on OpenAI and local backends (no thinking dial) and leaves Sonnet's
thinking pinned off (the deliberate cheap path). This is the only observability knob that changes a
request parameter and the only one that spends more — turning it on to watch the model think costs
adaptive-thinking output tokens, so **leave it `off` for unattended runs**. Adaptive thinking has
no token budget to tune (the model chooses depth), which is why the knob is a switch, not a dial.

## Live feed opt-in

`data.provider` also selects the TRADING-phase live data source. The default keeps TRADING on
offline **catalog replay**; `data.provider: yfinance` opts in to the free, ~15-min-delayed
Yahoo Finance feed (no credentials; needs the `data` extra). A bare `noctis run` never contacts
a live feed unless you explicitly ask for it. Feed behavior: [data.md](data.md).

## Secrets

Secrets live in `.env` (gitignored — copy `.env.example` and fill in). **Nothing is required to
run in paper mode.**

| Variable | Purpose |
|---|---|
| `DATABENTO_API_KEY` | Historical market-data ingests (research/backtest) |
| `<PROVIDER>_API_KEY` | Optional LLM key for hosted models — the variable name matches your `research.model` provider prefix; local backends need none |
| `ALLOW_LIVE` | The live-execution env gate (leave blank/unset for paper) |
