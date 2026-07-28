# Architecture

Noctis is one long-running process driven by the market clock. This page covers the phase
loop, the module map, the seam philosophy, the fill model, the trading day, and where state
lives.

## The phase loop

```mermaid
stateDiagram-v2
    direction LR
    [*] --> RESEARCH
    RESEARCH --> TRADING: market opens
    TRADING --> CLOSE: market closes
    CLOSE --> RESEARCH: report written
    RESEARCH --> STOPPED: time limit
    TRADING --> STOPPED: time limit
    STOPPED --> [*]
```

The state machine (`src/noctis/engine/machine.py`) researches while the market is closed, trades
while it is open, and reports at the close — looping until a ceiling is reached. There are two,
and they stop through the *same* move: `time_limit_hours` bounds **this process** (how long
tonight lasts, leaving the run resumable), and `run_limit_hours` bounds the **whole run** across
every stop/resume, marking it `completed` at the cap so two runs can be compared on equal compute
([cli.md](cli.md#bounding-a-run----run-limit-hours-and---finish)). The runtime
(`src/noctis/engine/runtime.py`) paces ticks in wall-clock time, waits out weekends, and routes
`SIGINT`/`SIGTERM` and both ceilings through one clean between-phases shutdown that flushes
state — deliberately one route, so a new way to stop can never behave differently from the one an
operator already trusts.

## The full pipeline

End to end, a strategy travels one path from raw data to a live paper record:

```
Data (fetch-once lake)
  → Research agent (formulate → match → optimize → decide)
  → Strategy generation (one reviewable .py, validated on write)
  → Backtesting (vectorbt-style pre-filter → walk-forward validation)
  → Out-of-sample validation (temporal holdout + symbol holdout)
  → Promotion (the gate order)
  → Forward paper record (champions trade only bars no tuning ever saw)
```

RESEARCH runs the first five stages while the market is closed; TRADING runs the last while it
is open; CLOSE writes the record and loops back. How a candidate earns promotion — the gate
order and the two out-of-sample axes — is the subject of [validation.md](validation.md).

## Everything heavy is a seam

The heavy engine/research/data stacks (`nautilus_trader`, `vectorbt`, `databento`, `optuna`,
`exchange-calendars`, the LLM SDKs, …) are **optional extras**. Each hides behind a swappable
seam with an in-house default, so the full test suite and bare paper mode run on the core
install alone. When a feature needs a missing package you'll see
`The '<pkg>' package is required … continuing without it` — install the extra named in the
warning (see [development.md](development.md)).

## Module map

| Area | Module | What it does |
|---|---|---|
| 🔐 Config + safety gate | `src/noctis/config` | Typed settings (`config.yaml` + `.env`); the paper/live double gate |
| 🧩 Composition root | `src/noctis/bootstrap.py` | One session-assembly seam: the settings → gate → mandate → CLI-flag precedence chain, plus the shared builders (lake, memory, console, the agent research session) every entrypoint uses instead of hand-wiring |
| 🗄️ Fetch-once data lake | `src/noctis/data` | Parquet catalog + coverage registry + coverage-diffed ingest + tail-only sync + integrity check + cost preflight |
| 📚 Strategy library | `strategies/` + `src/noctis/strategies/library.py` | One `.py` per strategy — thesis, code, tuned params, and research provenance in a docstring header; `write_strategy` validates in a subprocess so a broken file can never land. Three tiers: committed seeds in `strategies/`, plus the run's `__tmp/` working files and `champions/` (a later tier overrides an earlier one) |
| 📐 Strategies | `src/noctis/strategies` | `TraderStrategy` base: event-driven `on_bar()` plus a default `signals()` that replays it (parity by construction; a vectorised override stays possible); indicator helpers; SMA / RSI / Donchian worked examples; the candidate proposer |
| 🤖 Agent research | `src/noctis/research` | The agent loop: an LLM drives formulate → match → optimize → decide through a curated tool registry with per-strategy experiment journals and an exhaustion gate on verdicts |
| 🧬 StrategySpec engine | `src/noctis/strategies/spec` | Strategy-as-data (legacy ideation): a JSON graph compiles to a registerable family whose `signals()`/`on_bar()` share one rule evaluator; persists to the state dir's `specs.json` and re-registers at startup |
| 🏦 Broker | `src/noctis/broker` | Paper broker (a SimulatedExchange: fills, slippage, fees, P&L); event simulator; gated live stub |
| ⏪ Backtest | `src/noctis/backtest` | Two-stage pipeline: vectorbt-style pre-filter → walk-forward validation → `Scorecard` |
| 🏆 Champions | `src/noctis/champions` | Persistent registry + pure promotion rules (OOS metric, train−test gap guard) |
| ⚙️ Engine | `src/noctis/engine` | Market clock, state machine, research loop, close orchestration, runtime |
| 📡 Live | `src/noctis/live` | Trading loop + risk manager |
| 📊 Reporting | `src/noctis/reporting` | Close-of-day report, Markdown + structured JSON (`<run>/reports/<date>.md` / `.json`) + the run record/store |
| 🧠 Memory | `src/noctis/memory` | The agent-memory store (load / append / reorganize; lives at `<run>/memory/MEMORY.md`) |

## Two research paths, one contract

The agent loop (`src/noctis/research/agent.py` + the curated `ResearchToolbox`) and the legacy
proposer/Optuna loop return the *same* `ResearchSummary`, so the runtime calls either behind one
seam — no LLM configured means the legacy path runs over the same strategy library, exactly as
before. The agent loop itself runs one of two ways behind a further seam — the **conversation**
transcript or the small-context **episodic** driver — and the episodic path adds the
machine-fixed scenario oracle (FORMULATE authors the tape, the coder only satisfies it; see
[research.md](research.md)). See [research.md](research.md) for how a strategy earns promotion.

## Validation-on-write: the shared funnel and the Tier-1 invariants

`write_strategy` (`src/noctis/strategies/library.py`) is the one gate every authored file passes,
and it validates in a **fresh subprocess** (clean import → smoke replay → scenario replay →
`signals`/`on_bar` parity) so a broken or non-deterministic file can never land or poison an
import cache. The scenario replay is the file's own correctness oracle — its declared
known-outcome tapes — and each tape additionally runs the **Tier-1 invariant suite**
(`src/noctis/strategies/scenarios.py`, `check_invariants`): one extensible, ordered chain of pure
structural-honesty checks replayed over every tape, so both validator runners inherit every check
with no drift.

1. **Warmup honesty** — no nonzero target before the strategy's declared `warmup_bars(params)`;
   a default of `0` is undeclared and exempt (strategies outside the library are untouched).
2. **Determinism** — a second replay must produce an identical target series (it gates the
   replay-and-compare checks below, which a non-deterministic strategy would make meaningless).
3. **Truncation no-lookahead** — `signals(tape[:k]) == signals(tape)[:k]` at a handful of cut
   points: a vectorised `signals()` override that peeks at the future (a centered window, a
   full-series `max`/`mean`, a `shift(-k)`) is exposed by a prefix decision that changes when
   later bars are removed. The event `on_bar` path is causal by construction.
4. **Price-scale invariance** — scaling every price column ×10 must leave the target series
   unchanged, so an absolute-price threshold (which can't transfer across a symbol panel) is a
   structural defect, not a scale-free feature.

**The tolerant-both write gate.** `write_strategy` takes an optional compiled scenario spec. With
**no spec** (a hand-written source, the conversation loop) the path is byte-identical to before —
the file's own `scenarios()` are validated as authored. With a **spec** (the episodic driver's
fixed oracle) the gate resolves `warm` from the candidate's *own* declared warmup, replays the
compiled oracle at that warmup, **rejects** any coder-authored `scenarios()` block, and — on
success — machine-stamps a warmup-parametric `scenarios()` into the installed file. Either way one
validated file lands; the spec path just moves who authors the tape from the coder to the machine.

**Where the compiler sits.** The scenario spec vocabulary and the pure compiler
(`src/noctis/strategies/scenario_spec.py`: `LegSpec` / `Behavior` / `ScenarioSpec` / `SpecSuite`,
`compile_spec`, `describe_spec`) live in the **strategy layer**, not the research layer.
Compilation is a pure, deterministic function of `(spec, warm)` — no LLM, no I/O, no clock, no
randomness — and the module imports nothing from `noctis.research`, so the same spec always
compiles to the same `Scenario` objects and the gate owns the oracle independently of who proposed
it.

## The fill model

**The base contract: decide on bar *t*, fill at bar *t+1*'s open — and nothing else can create
a fill.** Both backtest stages and the live driver share this single-fill-source rule
(`src/noctis/broker/simulator.py`), and it is what makes the no-lookahead guarantee checkable:
every fill traces to a target decided strictly before the bar that prices it, and fills route
through the normal slippage/fee models, adverse to the trading side.

**Protective exits are the one sanctioned extension** — fixed **stop-loss**, **take-profit**,
and **trailing stop**, all expressed as *percentages* — declared by the strategy alongside its
target, evaluated by the **engine** intrabar against subsequent OHLC. Exits are opt-in and
declarative: the strategy states the rules, the engine enforces them, and the strategy never
observes whether one fired. A strategy remains a pure function of the bars it has seen, so
`signals()`/`on_bar` parity stays about the *target* series and the write gate's replay
semantics are untouched. The four decisions below are **resolved**; the implementation phases
in [protective-exits-plan.md](protective-exits-plan.md) build on them and do not re-litigate
them.

**1. Author API (resolved).** `Context.set_target` grows exactly one keyword-only, defaulted
parameter — source-compatible with every existing strategy file:

```python
@dataclass(frozen=True)
class ExitRules:  # beside Bar in src/noctis/strategies/base.py
    stop_pct: float | None = None  # exit if adverse move ≥ this fraction of entry
    take_profit_pct: float | None = None
    trail_pct: float | None = None  # exit if drawdown from best-since-entry ≥ this


def set_target(self, target: int, exits: ExitRules | None = None) -> None: ...
```

`TargetContext` captures `exits` alongside `target`. Rules are **re-declared every bar** with
the target — stateless from the strategy's side — and the engine associates them with the
*position*, not the bar. Exit percentages are ordinary `float` params a strategy forwards from
its `Params`, so the research agent tunes `stop_pct`/`take_profit_pct`/`trail_pct` as normal
`ParamSpec`s with no framework change.

**2. Execution semantics — the conservative intrabar policy (resolved).** Per bar *t+1*, in an
order chosen so no step can see a later step's information:

1. **Open** — the pending target from bar *t* executes at the open, exactly as today. If the
   fill opens or flips a position, exit tracking (re)anchors — `entry_price = fill price`,
   `best = entry_price` — and it clears on flat.
2. **Intrabar** — if a position is open and rules are armed, evaluate against the bar's
   high/low under the conservative policy:
   - **Gap-through fills at the open.** If the open is already beyond a level, the exit fills
     at the open — never at the untouched level.
   - **Stop beats take-profit.** Both levels touched within one bar ⇒ assume the stop fired;
     the intrabar path is unknowable from OHLC, so ambiguity resolves to the worst case.
   - **Prior-bar ratchet.** The trailing high-water mark ratchets on the *prior* bar's extreme
     while the trigger evaluates the *current* bar's adverse extreme — ratcheting and
     triggering off the same bar would be intrabar lookahead (the high that sets the mark may
     occur after the low that hits it). Structurally, `evaluate` runs before `ratchet` each
     bar, so the mark is the prior bar's extreme by construction.
   - Exit fills route through the **normal slippage/fee models**, adverse to the closing side.
     Short positions mirror long semantics symmetrically: stop above, take-profit below, and
     the trailing mark is the low-water mark.
3. **Close** — `on_bar` runs (sees the full bar, as today) and sets the next target; equity
   marks at the close.

**3. The re-arm latch (resolved).** After an exit fires, the engine latches that symbol **flat
until the strategy's target series *changes value*** (any transition: `+1→0`, `+1→−1`, …). The
first change un-latches; the new value then executes normally. A target *change* is the
strategy affirmatively re-deciding; a *held* target is stale conviction from before the
stop-out — honoring it would re-enter at the next open and turn every stop into a one-bar
speed bump. The latch mirrors the existing session-halt latch, so operators reason about one
latch shape. The consequence, stated loudly because authors must know it: **a strategy holding
`+1` for weeks treats a stop-out as terminal until its signal cycles.** That is correct — it
is what "the thesis was invalidated" means.

**4. The prefilter stays exit-blind (resolved).** With engine-side exits, realized P&L is no
longer a pure function of the target series, and the vectorised prefilter cannot see exit
fills. It keeps its coarse selection-filter role unchanged; the event-driven walk-forward —
authoritative for the `Scorecard` and every promotion gate — prices exits exactly. Stops are
**never approximated vectorially**; that is where lookahead bugs are born. Gate interaction:
exits change *candidate behavior*, not thresholds — the activity floor, gap guard, holdouts,
and consistency gates apply to exit-bearing candidates unchanged.

**Rollout posture.** Exits are **opt-in by declaration, and that is the only switch** — there
is no config kill-switch, because a knob that silently ignores declared stops would make
backtest and live disagree. A strategy that declares no rules runs a byte-identical code path,
so existing champions stay comparable: no staleness rule, no registry migration. The safety
net for every implementation phase is a golden regression — the three seed strategies (which
declare no exits) byte-identical through `noctis backtest` before and after.

**Refused scope** (re-litigated only as a new plan, never inside this one):

- **Limit/stop *entries* and resting-order management** — they redefine "activity," collide
  with the activity-floor/turnover gates, and open an entry-price overfitting axis.
- **Strategy-visible fills or position state** — a strategy stays a pure function of the bars
  it has seen; exits are rules the engine enforces and the strategy never observes.
- **Sub-timeframe stop evaluation in live** — exits evaluate on the strategy's declared
  timeframe bars, identical to what the backtest scored.
- **Absolute-price exit levels** — percentages only, so one tuned param set stays scale-free
  across a panel of symbols.
- **Any "realistic" intrabar path model** — unfalsifiable from OHLC; it can only flatter a
  backtest. The conservative policy is the only policy.
- **A config kill-switch for exits** — opt-in-by-declaration is the switch (see rollout
  posture above).

## The trading day

While the market is open, champions run on live or replayed bars and emit **paper** orders
through a simulated exchange, within risk limits. Live trading assigns each symbol its
best-scoring eligible champion (champions persist the symbols they were fit on; pre-panel
champions keep trading the whole universe).

**One driver, one feed contract, one settle order.** The session driver polls a `BarFeed`
(`src/noctis/live/feed.py`) — the live yfinance adapter (clock-bounded: the session close ends
the day) or a catalog `ReplayBarFeed` (data-bounded: the slice's exhaustion does) — so live and
replay can never diverge on how a day is traded. However the day ran, `TradingDay`
(`src/noctis/engine/trading_day.py`) settles it the same way: attribute forward P&L (derived
evidence, never blocks), persist the account **first**, advance the session high-water mark
**second** — a crash between the two re-trades the session rather than silently skipping it.
The TRADING entry itself sits behind its own seam: `TradingPhase`
(`src/noctis/engine/trading_phase.py`) assembles the account, forward ledger, and day runner,
resolves live vs replay, runs the catch-up loop, and folds every settled session into one
outcome the runtime copies into its report accumulators — the same interface tests drive
directly with fake bars and feeds.

**Catalog replay is a rolling live-holdout.** Each day trades only the newest session(s) past a
persisted high-water mark (the state dir's `trading_sessions.json`) — bars no tuning ever saw — one
risk-managed session per session date, so results accumulate into a genuine forward track
record. A day with no new lake data skips trading and says so in the report instead of
replaying stale bars.

**The account is one continuous paper account** (the state dir's `paper_account.json`): equity *and* open
positions carry across sessions — overnight gaps are real P&L, and the daily loss limit anchors
to that day's carried starting equity. Champion turnover never resets it; a corrupt state file
refuses to trade rather than silently restarting at 100k. Inspect it with `noctis account`
(also one line in `noctis status`); archive and start fresh with `noctis account --reset`.

**Champion turnover and carried positions.** A symbol *reassigned* to a different champion is
inherited: the new assignee starts from the carried position and re-decides at its first bar
(realized/unrealized attribution follows the inheritor). A position whose symbol **no** current
champion is eligible for is an **orphan** — unmanaged by anyone — and each session flattens
orphans at their first tradable bar through the normal risk/broker path (allowed even under the
daily-loss halt: it is risk-reducing). The closing P&L is credited to the champion that opened
the position via the forward ledger's recorded holder, and the flatten is named in the close
report.

## At the close

Noctis writes a report (`<run>/reports/<date>.md` + `.json`), syncs its data catalog
(tail-only), reconciles live-built bars against the authoritative catalog (see
[data.md](data.md)), reorganizes its own memory, and loops back to research.

## Where state lives

One contract: **the operator surface is committed templates/scaffold plus local, gitignored
copies; everything the engine writes lands under `workspace/`** (one knob, `workspace_dir`,
env `NOCTIS_WORKSPACE`) — and inside it, **a run owns its state**. `noctis init` scaffolds the
local copies; `noctis migrate` moves a pre-workspace layout in and adopts pre-run-scoped
workspace state into the reserved `legacy` run; a startup guard refuses to run beside abandoned
pre-workspace data and warns beside un-adopted workspace state.

```
workspace/
  data_lake/                  ← SHARED across runs. Vendor data is expensive and run-neutral.
  runs/
    index.json                ← the derived listing roll-up
    <run_id>/
      run.json  run.lock      ← the record and the liveness lock
      state/                  ← champions.json, paper_account.json, forward_ledger.json,
                                 specs.json, sessions/, experiments/
      strategies/             ← this run's __tmp/ and champions/ tiers
      memory/MEMORY.md        ← seeded from the committed MEMORY.seed.md at run creation
      reports/                ← this run's per-day close reports
      qa/                     ← the --debug tree
```

Two runs in one workspace therefore cannot crown champions onto one board or trade one paper
account, and a run's numbers describe only what that run produced. The four per-run paths
(`state_dir`, `reports_dir`, `qa_dir`, `memory_path`) derive from `run_dir` in
`config/settings.py`; `bootstrap.open_run_store` rebinds `run_dir` to the run it just minted, so
no command body does path arithmetic. `run_dir` defaults to the reserved `runs/legacy/` run —
what an invocation that never opened a run reads (`status`, `champions`, `account`, `report`, a
bare `research`), and the run `noctis migrate` adopts existing state into. The **committed
`strategies/` seeds stay read-only input for every run**: the three-tier discovery contract
(seeds → `__tmp/` → `champions/`) is unchanged; only the two writable tiers moved under the run.

**The operator surface (input — the engine treats all of it as read-only):**

| Path | What | Git |
|---|---|---|
| `config.example.yaml` → `config.yaml` | Every operating knob: committed template → your local copy (optional; defaults apply without it) | template **committed**; local copy ignored |
| `.env.example` → `.env` | Secrets + the `ALLOW_LIVE` gate | template **committed**; local copy ignored |
| `strategies/*.py` | The seed library — `TEMPLATE.py` + three worked examples, one reviewable `.py` per strategy with its research record in the header (see `strategies/README.md`) | **committed** |
| `mandate/` | Operator mandate scaffold — `MANDATE.md.example` (a balanced Sortino swing brief), five shipped profiles, `tune-first`, `references/` (see `mandate/README.md`) | **committed** (scaffold only) |
| `mandate/MANDATE.md` + custom personalities + personal `references/` | The operator's own steering input | ignored |
| `MEMORY.seed.md` | Curated starting lessons — copied into the live memory on first run | **committed** |

**The workspace (output — git never sees inside it):**

| Path | What |
|---|---|
| `workspace/data_lake/` | Parquet catalog + `coverage.db` + `manifest.json` — **shared by every run** |
| `workspace/runs/index.json` | The derived listing roll-up over every run record |
| `<run>/run.json` + `run.lock` | One run's self-describing record and its liveness lock |
| `<run>/state/champions.json` | That run's champion registry |
| `<run>/state/paper_account.json` | That run's continuous paper account |
| `<run>/state/trading_sessions.json` | That run's replay high-water mark |
| `<run>/state/experiments/<strategy>.jsonl` | Per-strategy experiment journals, one line per backtest/sweep trial |
| `<run>/state/specs.json` | LLM-minted `StrategySpec` definitions, re-registered at startup |
| `<run>/reports/YYYY-MM-DD.md` + `.json` | Close-of-day reports, human- and machine-readable |
| `<run>/memory/MEMORY.md` | That run's live long-term memory, seeded from `MEMORY.seed.md` |
| `<run>/strategies/__tmp/` | The research agent's working files (drafts, candidates, rejects) |
| `<run>/strategies/champions/` | Locally-promoted champions (never reach the public repo) |

(`<run>` is `workspace/runs/<run_id>/`.)
