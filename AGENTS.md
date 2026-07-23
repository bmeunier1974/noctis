# AGENTS.md

This file provides guidance to coding agents working in this repository (Claude Code loads it
via the `CLAUDE.md` stub; tools that follow the [agents.md](https://agents.md/) convention read
it directly).

## What this is

**Noctis** is an autonomous, **paper-only** trading system: one long-running process that
**researches strategies while the market is closed**, **trades champions on (live or replayed)
bars while it is open**, and **reports at the close** — looping day after day. With an
`ANTHROPIC_API_KEY`, RESEARCH *is* an agent session: Claude authors one-file Python strategies
into the library's working tier (`workspace/strategies/__tmp/`; the committed `strategies/`
seeds are read-only input) and drives `formulate → match → optimize → decide` through a curated tool
registry. Without a key it falls back to a legacy Optuna/proposer loop over the same library.

`README.md` is a short landing page; the authoritative narrative lives in `docs/`
(architecture, research, configuration, data, cli, safety, development). This file is the
operating contract — read it first, then the relevant `docs/` page for depth.

## The rules that don't bend

These are load-bearing invariants, not preferences. The whole point of this system is to tell a
*real* edge from noise, so its discipline is enforced **structurally** (in gates and seams), not by
prompting. Do not help anyone — the research agent, a backtest, or a human — get around them.

1. **Paper-only is a two-gate invariant.** Real-money orders are reachable *only* when config
   `mode: live` **and** env `ALLOW_LIVE=true` are both set — and even then the live adapter is a
   stub that refuses. `mode: live` without `ALLOW_LIVE` is a hard startup error, never a silent
   downgrade (`src/noctis/config/gate.py`). Never weaken either gate, never make them share a source,
   never wire a real order path into the stub.

2. **A failing strategy is a signal, not a bug to fix.** If a candidate can't pass the promotion
   gates, the answer is a better thesis or an honest `reject_strategy`, **never** a loosened gate.
   Do not raise `max_gap`, lower a holdout bar, or shrink the holdout set to make something
   "pass." That is overfitting with extra steps. The gates in `src/noctis/champions/promotion.py`
   (activity floor → overfit gap guard → forward-holdout → symbol-holdout → consistency → beat the
   weakest) are the arbiter of quality by design.

3. **No lookahead. Ever.** Both backtest stages decide on bar *t* and fill at bar *t+1*'s open;
   walk-forward test windows sit strictly *after* their train window (`src/noctis/backtest/splits.py`).
   Previews and market digests never expose holdout bars. Any change that lets information from the
   future (or from the holdout) reach a decision is a correctness bug, full stop.

4. **Out-of-sample before promotion, on two axes.** Research is *panel* research: candidates are
   tuned on a fit set of symbols and validated on both a **temporal holdout** (the most-recent
   slice the search never touched) and a **symbol holdout** (names never used in tuning/selection).
   Both gates must stay live end-to-end.

5. **The mandate is a search prior, not a permission slip.** An operator mandate (`mandate/`) steers
   *what* to look for (style, risk appetite, symbols) and may bind exactly one knob —
   `promotion.metric`. It never loosens a gate, the exhaustion rule, or the honesty contract.

6. **Secrets live in `.env` only** (gitignored). No credentials, and no vendor market data, ever land
   in git — the lake is reproducible from the coverage registry + manifests.

When a request would cross one of these, say so and propose the honest alternative instead.

## Commands

```bash
uv sync --all-extras       # full env: core + dev + all seams
uv sync                    # minimal: core + dev (test suite / paper mode); reproducible from uv.lock
# commands below run inside the synced env — prefix with `uv run`, or activate `.venv`
pytest                     # full suite (config: -q, testpaths=tests)
pytest tests/test_champions.py::test_name -q     # a single test
ruff check . && ruff format --check .   # lint + format (line-length 100; rules E,F,I,W,UP,B)
mypy                       # type-check src/noctis (config in pyproject.toml)
pre-commit run --all-files # every quality gate (install once: pre-commit install)

python -m noctis setup [--check]   # guided first-run wizard: files, extras, keys, LLM verify
python -m noctis init              # scaffold local config/.env/mandate + workspace (idempotent)
python -m noctis migrate [--dry-run]   # move a pre-workspace layout into workspace/
python -m noctis run -v            # the day/night loop (stops at time_limit_hours)
python -m noctis research -v       # ONE observable agent research session (needs ANTHROPIC_API_KEY)
python -m noctis status            # resolved mode, market state, next transition, champions
python -m noctis backtest <name>   # replay a library strategy on its shipped Params defaults
python -m noctis champions [--reset]   # list champions; --reset re-fills slots under current gates
python -m noctis report [--as-of DATE]
python -m noctis data status|sync|ingest <SYM> --start ... --end ... [--dry-run]
```

CLI verdict tools and research need `ANTHROPIC_API_KEY`; a bare `python -m noctis run` contacts no
external service.

## Architecture (the parts you can't grok from one file)

**The phase loop.** `src/noctis/engine/machine.py` is a market-clock-driven state machine —
RESEARCH ↔ TRADING → CLOSE → RESEARCH (+ STOPPED) — with a global time limit that can `stop` from
any state. `src/noctis/engine/runtime.py` wires the real phase work behind injectable hooks and paces
ticks in wall-clock time; it researches while closed, waits out the weekend via the `BoundedWaiter`
pacing seam (`src/noctis/engine/pacing.py`), and routes SIGINT/SIGTERM + the time limit through one
clean between-phases shutdown.

**Everything heavy is a seam.** `nautilus_trader`, `vectorbt`, `databento`, `optuna`,
`exchange-calendars`, `anthropic` are **optional extras** (`pyproject.toml`). Each hides behind a
swappable seam (`src/noctis/data/seam.py`, `src/noctis/broker/seam.py`, the research loop selector, …) with
an in-house default, so the full test suite and bare paper mode run on the *core* install alone.
When you touch a feature and see "the '<pkg>' package is required … continuing without it," install
the extra named in the warning — don't add it to core deps.

**A strategy is a file, and the file is the whole artifact.** `strategies/*.py` — one
`TraderStrategy` subclass each (`src/noctis/strategies/base.py`): thesis + provenance in the docstring
header (`status`/`style`/`symbols`/`tuned`), a frozen `Params`, `on_bar` (long/short/flat targets,
O(lookback), no I/O/globals/randomness), `param_space()`, and `scenarios()` (2–8 known-outcome tapes
that are the file's own correctness oracle). `signals()` is an optional vectorised override — the
base class replays `on_bar` so both paths agree by construction. `write_strategy` validates in a
**fresh subprocess** (import + smoke replay + scenario replay + the Tier-1 structural invariants —
warmup honesty, determinism, truncation no-lookahead, price-scale — + signals/on_bar parity) so a
broken file can never land; it is *tolerant-both* — a hand-authored `scenarios()` validates as
written, while the episodic driver's FORMULATE-authored fixed oracle (a structured
`scenario_spec`) is machine-stamped in and re-validated. The library lives in **three tiers**
(`src/noctis/strategies/library.py`,
`LibraryPaths`), discovered lowest-precedence first: committed **seeds** in `strategies/`
(`TEMPLATE.py` + the three worked examples, the *only* files in the public repo — read-only input)
→ the `workspace/strategies/__tmp/` working area (drafts/candidates/rejects, local-only) → the
`workspace/strategies/champions/` folder (locally-promoted). A later tier
overrides an earlier one, so a champion beats a seed of the same name and committed seeds are never
mutated in place (a seed rewrite is redirected into `__tmp/`). `write_strategy` authors into `__tmp/`;
on promotion the file is **moved** into `champions/`, tuned params are written back as defaults, and
the header is re-stamped `status: champion`, so `noctis backtest <name>` replays exactly what shipped.
A champion file is immutable — improving one means a new name. Full contract: `strategies/README.md`.

**Two research paths, one contract.** `src/noctis/research/agent.py` (Claude + `src/noctis/research/tools.py`
`ResearchToolbox`) and the legacy proposer/Optuna loop return the *same* `ResearchSummary`, so the
runtime calls either behind one seam. The agent's discipline is entirely structural: the exhaustion
gate refuses a verdict until ≥ `research.min_trials` distinct param sets are journaled to
`workspace/state/experiments/<name>.jsonl`, backtests return aggregate scorecards only, previews never cross
into holdout bars, and data spend sits behind a budget preflight. `run_sweep`'s execution engine —
the seeded sampler, the fork pool, the OOM/stall guard — is its own `SweepRunner` seam
(`src/noctis/research/sweep.py`); the toolbox keeps the accounting (budget, journal, ranking). The legacy `StrategySpec` engine
(`src/noctis/strategies/spec/`) is strategy-as-data: a JSON graph that compiles to a family, persisted
to the state dir's `specs.json` and re-registered at startup.

**Backtest → Scorecard → promotion.** `src/noctis/backtest/pipeline.py` runs a vectorbt-style pre-filter
→ walk-forward validation → a `Scorecard` (panel means across fit symbols + temporal/symbol holdout
metrics + activity/turnover). `src/noctis/champions/promotion.py` is a **pure** decision function over
scorecards (see rule 2 for the gate order); `src/noctis/champions/registry.py` persists the board to
the state dir's `champions.json`. Promotion compares on a scale-free footing and treats a champion scored
under a *different* metric as "stale" (displaceable), because cross-metric numbers aren't comparable.

**Config + mandate overlay.** `config.yaml` + `.env` → typed `src/noctis/config/settings.py` (env vars
override YAML; `NOCTIS_CONFIG` points at an alternate file). The active mandate's front-matter
`config:` block may overlay **only** `promotion.metric`; a `--metric` CLI flag wins over the overlay,
and `--mandate`/`--directive` (mutually exclusive) override the config selector for one session.
That precedence chain, and the collaborator builders the entrypoints share (lake, memory, console,
the strategy-family registry, the agent research session), live in one composition root —
`src/noctis/bootstrap.py`. Assemble sessions there, not by hand in a command body.

## Conventions and gotchas

- **Where state lives — the input/output contract:** committed files are input the engine treats
  as read-only (`strategies/` seeds + `TEMPLATE.py` + `strategies/README.md`, the `mandate/`
  scaffold, `config.example.yaml`, `MEMORY.seed.md`); **everything the engine writes lands under
  the gitignored `workspace/`** (state, data lake, reports, agent memory, and the strategy
  `__tmp`/`champions` tiers), one knob (`workspace_dir`, env `NOCTIS_WORKSPACE`) all derived paths
  follow. `noctis init` scaffolds the local input copies; `noctis migrate` moves a pre-workspace
  layout in; a startup guard refuses to run beside un-migrated legacy data (`status` only warns).
  A public clone ships templates/seeds/scaffold only, and no operator's champions or rejects
  reach git.
  The `mandate/` folder is the same shape: only the scaffold is committed (`MANDATE.md.example`,
  the five shipped `profiles/`, `tune-first.md`, README, one reference example); the operator's own
  `MANDATE.md`, custom personalities (`mandate/<name>.md`, `profiles/<name>.md`), and personal
  `references/` are gitignored (allowlist in `.gitignore`), so steering never pollutes the repo.
- **Test isolation** (`tests/conftest.py`): safety-gate/secret env vars are cleared around every
  test — tests see defaults, not your `.env`. Strategy families need no isolation: there is no
  global registry — every session gets its own `FamilyRegistry` (`noctis.bootstrap.build_families`),
  and tests build throwaway instances.
  A `.pyc`-staleness gotcha bit `_load_module` before; prefer fresh-subprocess
  validation over trusting import caches.
- **Agent memory follows the seed pattern**: the committed `MEMORY.seed.md` carries curated
  starting lessons every fresh install begins with; the live, agent-maintained file is
  `workspace/memory/MEMORY.md` (gitignored), seeded from the seed on first run by
  `bootstrap.build_memory` — the copy happens *before* `MemoryStore` constructs, because its
  `load()` auto-creates a blank template for a missing file. Skim the live file when something
  behaves unexpectedly; promote a lesson into the seed only when it should ship to every user.
- **Ruff** ignores `B008` on purpose (Typer uses calls in argument defaults). Target py311.
