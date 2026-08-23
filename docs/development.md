# Development

Full installation, the optional extras, and the quality gates. For workflow, standards, and
governance see [CONTRIBUTING.md](../.github/CONTRIBUTING.md); for the strategy-file contract see
`strategies/README.md`.

## Installation

Everything — dependencies, groups, extras, and tool configuration — lives in `pyproject.toml`,
resolved to exact versions in `uv.lock`. [`uv`](https://docs.astral.sh/uv/) is the project
standard; a `.python-version` pins the interpreter (3.11). Requires Python ≥ 3.11.

**Running Noctis? Install everything** — this is the standard operator install, and what the
README's getting-started uses:

```bash
uv sync --all-extras               # every seam filled, reproducible from uv.lock
uv run python -m noctis setup      # then the guided first-run wizard
```

`uv sync` reads `uv.lock` and installs the exact locked versions into `.venv`. Run any command
in that environment with `uv run` (e.g. `uv run pytest`) or activate `.venv` first. Use
`uv sync --locked` to fail loudly if `pyproject.toml` and `uv.lock` have drifted (this is what
CI runs).

**Developing the core?** A minimal install exists as an engineering property, not a usage
recommendation: the heavy stacks live behind swappable seams, so a bare `uv sync` (core + dev
tooling) imports, runs bare paper mode, and passes the *entire* test suite with no optional
stack installed. That is what CI runs, and it is what keeps every seam honest — but it is not
a machine that can research or ingest data. Individual extras exist for working on one seam:

```bash
uv sync                                    # core + dev only (CI / seam work)
uv sync --extra data --extra llm           # databento + yfinance + the LLM seam
uv sync --extra engine --extra research    # nautilus / vectorbt / optuna
```

> [!NOTE]
> On a partial install, a feature whose seam is empty says so and continues — e.g.
> `The 'databento' package is required … continuing without it`. Add the extra named in the
> warning, or just `uv sync --all-extras`. `noctis setup --check` lists what's missing.

## Dependency groups

`dev` is a PEP 735 **dependency-group** — a bare `uv sync` installs it by default. The runtime
seams are **optional extras** (`uv sync --extra <name>`); the core install runs paper-only with
none of them.

| Group / extra | Kind | Packages |
|---|---|---|
| **core** | base | `pydantic`, `typer`, `numpy`, `pandas`, `pyarrow` |
| **dev** | group (default) | `pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit` |
| **engine** | extra | `nautilus_trader` |
| **research** | extra | `vectorbt`, `optuna`, `quantstats-lumi` |
| **data** | extra | `databento`, `yfinance`, `exchange-calendars`, `transitions`, `apscheduler` |
| **llm** | extra | `anthropic`, `litellm` |
| **hardware** | extra | `psutil` — richer per-segment machine facts (CPU model, physical cores, RAM) in the run record; without it the block degrades to the stdlib subset and names `hardware` in its `degraded_seams` |

## Quality gates

The source lives in a `src/` layout (`src/noctis/`); tests live in `tests/`. Ruff handles
linting and formatting, mypy handles static type-checking, and pre-commit runs all of it on
every commit — each tool is configured in `pyproject.toml`.

```bash
uv run pre-commit install         # one-time: lint/format/type-check on every commit

uv run pytest                     # full test suite
uv run ruff check .               # lint
uv run ruff format .              # format
uv run mypy                       # type-check src/noctis
uv run pre-commit run --all-files # all quality gates at once

uv run python scripts/engine_fingerprint.py          # the engine fingerprint ratchet (below)
uv run python scripts/engine_fingerprint.py --write  # regenerate engine_fingerprint.json
                                                     # (refuses an undeclared arbiter move)

uv run python scripts/prompt_fingerprint.py          # the prompt fingerprint ratchet (below)
uv run python scripts/prompt_fingerprint.py --write  # regenerate prompt_fingerprint.json
                                                     # (refuses an undeclared prompt change)
```

## The engine fingerprint ratchet

`engine_fingerprint.json` (repo root) is the committed statement of what this engine *is*: the
declared `ENGINE_VERSION` plus one digest per behavioural component — and, under each component,
a digest per allowlisted file, so a drift report names the file that moved rather than every file
the component covers. The digests come from `src/noctis/observability/engine_id.py`. The **rule** —
the tier split below, and the `ENGINE_VERSION` agreement it needs — lives in
`src/noctis/observability/engine_ratchet.py`; the mechanics every ratchet shares — the record, the
check, `--write` and the report — live once in `src/noctis/observability/ratchet.py`, so a fix to
them cannot land in one ratchet and be missed in the other. The check runs in **CI** and in
**pre-commit**.

Why it exists: `ENGINE_VERSION` is the key two runs' numbers are compared on
(`noctis engine` prints it), and a declared version nobody remembers to bump is worse than none —
it asserts a comparability that does not hold. The check is split on the arbiter/searcher line
(see [architecture.md](architecture.md)):

| Component drift, no `ENGINE_VERSION` bump | Result |
|---|---|
| `gates`, `backtest` — the **arbiter**: what passes, and what a number means | **Fail.** Naming the component and the files that moved. This is the change that invalidates every stored champion comparison, so it can never land silently — and `--write` **refuses to regenerate** it (see below) |
| `research`, `prompts`, `profiles`, `seeds`, `memory_seed`, `schema` — the **searcher** | **Warn and pass.** Naming the component and the files. Improving the searcher must not invalidate an experiment whose arbiter held still, and a ratchet that fires on a docstring edit gets disabled |

The **same line governs resuming a run** (`src/noctis/observability/engine_change.py`): arbiter
drift between the engine a run froze at creation and this checkout **refuses the resume** unless
`--allow-engine-upgrade` accepts it on the record, searcher drift **warns, records and proceeds**,
and no drift is silent (see [cli.md → Engine change](cli.md#engine-change-resuming-after-the-code-moved)).
Both enforcers classify through the one `tier_of` function over the one `ARBITER_COMPONENTS`
constant in `engine_id.py`, and a test binds them to each other component by component — two copies
of that set would eventually disagree, and the disagreement would be silent.

The rule in full, in evaluation order: a missing or unreadable record fails; a record declaring
another `ENGINE_VERSION` than the tree fails as **stale**; arbiter drift fails (with the version
unchanged that is undeclared drift, with it bumped the record simply was not regenerated); drift
confined to the searcher tier warns and exits zero. Staleness is always reported and always names
the regeneration command — which tier moved decides only whether it *blocks*.

A component **new to the map** is drift too, even when this checkout cannot identify it (an
optional input, a file not landed yet): what moved is one rule, `engine_id.compare` — present on
one side only moved, two nulls did not — so a fingerprint surface *appearing* is never silent. It
prints as `null -> null` with no file lines under it, because the name is the news, and the tier
then decides as usual: a searcher name warns, an arbiter name fails.

So, when you move a component:

```bash
# 1. arbiter component (gates/backtest)? bump ENGINE_VERSION in
#    src/noctis/observability/engine_id.py — searcher-only changes need no bump
# 2. regenerate the record, and commit it in the SAME PR so the diff shows what moved
uv run python scripts/engine_fingerprint.py --write
```

**Step 1 is not optional, and `--write` enforces that.** Regenerating rewrites *every* component
at once, so a PR that moves a searcher component (the common case) is told to run it — and if that
also quietly absorbed a moved `gates` digest, the ratchet would hold only for contributors who read
the failure before typing the command it printed. So `--write` runs the check first and **refuses to
regenerate** on arbiter drift while the recorded and computed `ENGINE_VERSION` agree: it writes
nothing, exits 1, and prints the bump-or-restore guidance plus its refusal.

```text
$ uv run python scripts/engine_fingerprint.py --write
FAIL  engine fingerprint ratchet (engine_fingerprint.json)
  arbiter drift with no ENGINE_VERSION bump: gates. A change here invalidates every stored
  champion comparison — bump ENGINE_VERSION in src/noctis/observability/engine_id.py in this PR,
  or restore the behaviour
  gates (arbiter): 4a9c1e0f8b21d735 -> 0d45608deb971291
      src/noctis/champions/promotion.py
  refusing to regenerate: --write cannot be the way an undeclared arbiter move gets recorded
```

An arbiter move must therefore arrive **declared** — bump, then regenerate — and the two-step
sequence above is the only one that lands it. Everything else stays a single command that leaves
the tree checkable: searcher-only drift, an arbiter move whose bump *is* already in the tree (the
record had simply not caught up), no drift at all, and a missing or unreadable record — there is
nothing to compare against, and that is how the baseline is created in the first place.

Files outside the allowlist in `COMPONENT_PATHS` — docs, tests, the README, an operator's
gitignored mandate — move no digest and never fire the check.

## The prompt fingerprint ratchet

`prompt_fingerprint.json` (repo root) is the same idea for what the model is *told*: one content
hash per LLM call site, plus a digest per allowlisted file under it. The hashes come from
`src/noctis/observability/prompt_id.py` (`site_digest(site)` is the pure read a future benchmark
record's key uses); the **rule** — the declared-change rule below, and the changelog reader it
needs — lives in `src/noctis/observability/prompt_ratchet.py`, on the same shared mechanics
(`src/noctis/observability/ratchet.py`) the engine ratchet runs on. It runs in **CI** and in
**pre-commit**, exactly like the engine one.

It is a **separate artifact on a separate clock**, and deliberately so: prompts and arbiter
behaviour drift independently, so a prompt rewrite must not read as "the judge moved" and a
threshold change must not read as "the model was told something new".

| Site | Assets |
|---|---|
| `author` — the coder site's brief and the contract sheet it must satisfy | `research/author.py`, `research/contract_sheet.py`, `research/digests.py` |
| `briefings` — the rendered briefings that are the episodic stages' user turns | `research/briefings.py`, `research/digests.py` |
| `conversation` — the conversation loop's system prompt | `research/prompt.py`, `research/digests.py` |
| `distill` — the memory distiller's summarization prompt | `research/distill.py` |
| `episodic` — the driver's per-stage system texts and emit contracts | `research/driver.py`, `research/digests.py` |
| `ideation` — the seeded-idea prompt | `research/ideation.py` |

`research/digests.py` renders facts four of those prompts embed, so it is listed under each of
them and its edit moves all four hashes. Over-partitioning is the accepted direction; silence is
the failure this ratchet exists to end. There is **no tier here** — every site is the same kind of
thing, so every drift is the same kind of event, and nothing warns-and-passes. A site **new to
the map** whose assets this checkout cannot identify is drift here as well — the same one rule,
printed `null -> null` with no file lines — and with no entry naming it, it is an undeclared
change like any other: it fails.

**The declared-change rule** has two halves and needs both: the newest entry in
[`docs/prompt-changelog.md`](prompt-changelog.md) must *name the drifted site* on its heading line
(`## 2026-08-01 — sites: author, ideation`), **and** that entry must have arrived after the
committed record was written (the record stores the digest of the entry it was regenerated
against). A nameless entry declares nothing, and yesterday's entry is a standing permission rather
than a declaration.

So, when you change a prompt:

```bash
# 1. add a dated entry at the top of docs/prompt-changelog.md naming the site(s) that moved,
#    and what changed — the hash has to read back to a sentence
# 2. regenerate the record, and commit it in the SAME PR
uv run python scripts/prompt_fingerprint.py --write
```

**Step 1 is not optional, and `--write` enforces that**: regenerating rewrites every site at once,
so it **refuses to regenerate** undeclared drift — it writes nothing, exits 1, and prints the
declare-or-restore guidance plus its refusal.

```text
$ uv run python scripts/prompt_fingerprint.py --write
FAIL  prompt fingerprint ratchet (prompt_fingerprint.json)
  undeclared prompt drift: author. A prompt change must arrive with its explanation — add a dated
  entry to the top of docs/prompt-changelog.md whose heading names the site(s), e.g.
  "## 2026-08-01 — sites: author" — or restore the wording
  author (UNDECLARED): 0a9c1e0f8b21d735 -> 7d45608deb971291
      src/noctis/research/author.py
  newest docs/prompt-changelog.md entry: 2026-07-14 — sites: distill
  refusing to regenerate: --write cannot be the way an undeclared prompt change gets recorded
```

That last line is there because "I wrote an entry and it still fails" is the question this tool
gets asked: it names the entry the check actually read, so a heading inside a code fence, a
misspelled site or a `sites:` marker left off is visible rather than mystifying. It is printed for
a missing or unreadable record too — the one verdict no policy judges — so the report always names
the entry the check read.

Everything else stays a single command: drift the changelog declares (the record had simply not
caught up), no drift at all, and a missing or unreadable record — that is how the baseline is
created. A changelog edit on its own never fails the check: nothing the model is told has moved,
and failing there would push you to regenerate, consuming the entry you had just written.

## The eval boundary and its import guard

The third guard in this family enforces a *direction* rather than a hash: **the eval layer
(`src/noctis/eval/`) imports the engine, and the engine never imports the eval layer.** The eval
layer is benchmark infrastructure — one `AgentSite` declaration per LLM judgment site, the
`HarnessSpec` that names the prompt-composition ablations, the per-site knob sets — and a
benchmark exists to measure production, so production must not be able to notice that it exists.

Why it is a rule and not a preference: the moment an engine module can import `noctis.eval`, a
bench-only ablation ("run FORMULATE with the contract sheet off, to see what it is worth") becomes
reachable from a real research session, and every run afterwards is a run whose prompt composition
nobody can state from the record alone. That is the platform's invariant *production behaviour never
depends on benchmark infrastructure*, and it is held structurally: `HarnessSpec` is not a settings
field and no mandate overlay path can bind it, because production config has no word for it.

Enforcement is a static, stdlib-only scan (`src/noctis/eval/guard.py`) that returns every module
*outside* the package which imports it — `import noctis.eval`, `from noctis import eval`, a
submodule import, or the relative spelling. `tests/test_eval_boundary.py` runs it against
`src/noctis` on every CI run and **fails hard**, naming the offending module, file and line:

```bash
uv run pytest tests/test_eval_boundary.py -q   # the eval import-isolation guard
```

```text
noctis.research.agent imports noctis.eval.sites (noctis/research/agent.py:12)
```

If you are on the wrong side of it, the fix is never an import. Either the thing belongs in the
engine — move it there and have the eval layer import *it* — or the engine does not need it.

**One exemption, whose shape is checked too.** `noctis.cli` may name `noctis.eval.cli` — the bodies
behind the operator's `bench` verb group — and only from *inside a function body*
(`guard.DEFERRED_EXEMPTIONS`). The rule protects production *behaviour*, and a CLI verb group is an
operator typing a word, not behaviour a research session can reach: deferred, nothing is imported
until somebody types `bench`, so `noctis run` still loads no benchmark code at all. A module-level
import in `cli.py`, a deferred one in any other engine module, and a deferred one naming any other
eval module (including `from noctis.eval import cli`, which reaches the *layer* first) are all still
violations — the suite asserts each of them.

**The registry is five sites, pinned.** `src/noctis/eval/registry.py` declares `coder`,
`formulate`, `decide`, `discover` and `distill` as plain module-level constants (no runtime
registration: a registry whose contents depend on import order is a benchmark nobody can
reproduce), and `tests/test_eval_closure.py` pins that id set and resolves each declaration against
the production objects it binds — the episodic driver's own emit contracts *by identity*, an
importable renderer, a `SiteKnobs` subclass. A renamed briefing builder or a copied contract
therefore breaks the build, not the first benchmark run.

Two LLM call sites are **deliberately undeclared**, and the registry's docstring says so: the
**conversation loop**, whose input is an accumulated transcript rather than a function of disk (it
is measured end-to-end by the parity harness instead — see [parity.md](parity.md)), and
**onboarding-verify**, which is a liveness check rather than an agent judgment. A site's identity
for a benchmark record — its hand-bumped `version` plus its prompt-asset hash — comes from
`src/noctis/eval/identity.py`, the one bridge between the registry and the prompt ratchet above.

**The layer has its own composition root.** `src/noctis/bootstrap.py` may never import the eval
layer, so the eval layer assembles its own sessions in `src/noctis/eval/bootstrap.py` — the bench
area and cases root, the execution seam a `--workers` count selects, the site-input adapter table
(`SITE_ASKS`), and the live attempt callable that asks the configured model through the engine's
own LLM seam. Assemble a bench there, not in a verb body — the same rule `bootstrap.py` holds the
engine's entrypoints to, on the other side of the line.

## The research-toolbox surface and its reach-through guard

The fourth guard in this family protects a *seam* rather than a direction or a hash: an agent
research session carries exactly one object — the `ResearchToolbox` — and every reader of it holds
the declared surface `src/noctis/research/surface.py`, never the collaborators behind it. Two tiers:
`ResearchFacts`, the derived facts a *renderer* reads (the briefings, the system prompt, an eval site
rebuilding a past ask), and `Toolbox` on top of it, which adds the tools, the capture seams and the
frozen counters snapshot a *driver* needs.

Why it is a rule and not a preference: before the surface existed, each reader reached *through* the
toolbox for whichever collaborator happened to hold the answer (`toolbox.journal`,
`toolbox.registry.capacity`, `toolbox.lake.preflight.budget_usd`), usually behind a
`getattr(…, default)` probe — so a rename changed what the model was told without changing a
renderer, and a probe that missed stated a number nobody configured as though it had been measured.
Every fact on the surface is answerable instead: a lake with no cost preflight answers `None`, an
unreadable coverage registry answers an empty inventory.

`tests/test_toolbox_boundary.py` is the enforcement, in two halves. A static scan over `src/noctis`
names every module outside `noctis.research.tools` — the module that *owns* the collaborators — that
reads one off a toolbox or probes it with `getattr`, naming module, file and line; and four objects
are measured against the Protocol they claim: the production toolbox, the episodic driver's fake,
and the bench's neutral session and case toolbox.

```bash
uv run pytest tests/test_toolbox_boundary.py -q   # the toolbox reach-through guard
```

```text
noctis.research.briefings reaches toolbox.journal (noctis/research/briefings.py:365)
```

The scan tokenizes each file and drops every comment and string literal before matching, so an
explanation may quote the reach it replaced — the surface module's own docstring does — without
tripping the guard; and it lives in the test rather than in the package because nothing in
production reads its verdict. Its twin is `tests/test_prompt_goldens.py`, which pins the three
briefings and the system prompt byte-for-byte: the boundary says who may read a fact, the goldens
say that re-pointing a reader changed nothing the model is told.

## The run tree's layering and its import guard

The fifth guard protects a *direction inside one package*. `src/noctis/reporting/run_tree/` is the
only code that touches `workspace/runs/<run_id>/`, and it is five modules whose dependencies point
one way: **`record ← {address, index, lock, evidence} ← store`**.

Why it is a rule and not a preference: the whole value of splitting one 2300-line module into a
package is that the narrow modules stay narrow. Resolving `@label` needs `read_record` and nothing else;
deciding whether a lock may be taken needs not even that. One import added by accident — an index
reaching for the store, an address reaching for the lock — puts the collectors, the engine
fingerprint and the whole lifecycle back behind a ten-line function, and nothing visibly breaks
while it happens.

The second clause points outwards: the six collectors need the research journals, the champion
registry, the broker's ledgers, the data types, the settings model and pandas — and **only**
`evidence.py` may name one of them, at any nesting level, deferred or not. That is what makes "a
record write stays cheap" a shape rather than a comment: the subprocess probe in
`tests/test_run_tree_store.py` measures what importing the package actually costs, and this says
where a heavy import may ever be *written*.

Enforcement is a stdlib-only `ast` walk over every import statement of each module — module level
and function level, absolute and relative, including a reach around through the package's own
`__init__` — and it lives in the test rather than in the package, because nothing in production
reads its verdict:

```bash
uv run pytest tests/test_run_tree_boundary.py -q   # the run-tree layering guard
```

```text
index.py imports ['store'] from noctis.reporting.run_tree
```

The layering is also what lets a test of a narrow module stay narrow: `tests/_run_tree_helpers.py`
writes the files those modules read (`write_run` for a `run.json`, `hold_lock` for a `run.lock`), so
a test of the four address forms never opens a store. What each module owns is in the glossary entry
**Run tree** (`CONTEXT.md`); the record it all exists to write is [run-record.md](run-record.md).

## Dev scripts

`scripts/` holds dev tools that are deliberately *not* CLI subcommands. One is the ratchet
entrypoint above (`scripts/engine_fingerprint.py`, a thin wrapper over the tested module). The
other is the **parity harness** (`scripts/parity_harness.py`): it runs both research loops — conversation and
episodic — on one fixed lake fixture and prints a side-by-side metrics comparison, the evidence gate
for preferring episodic on small-context backends. It runs *paid* model sessions, so it refuses
without an API key and prints its spend first; the deterministic metric math it prints is covered by
`tests/test_parity.py`. See [parity.md](parity.md).

## Reading a QA report

`--debug` (on both `noctis run` and `noctis research`, see
[cli.md](cli.md#qa-report---debug)) records everything a session did to a per-run report tree
under `qa_dir` — default `<run_dir>/qa/<run-id>/`, so it follows the run (and `workspace_dir` /
`NOCTIS_WORKSPACE` relocation) and, like the rest of the workspace, never enters git. The run id
(`20260720T144233Z-a3f9c1`) is a sortable UTC-stamped, greppable name, so a plain `ls` of the QA
area is already chronological. Retention is prune-on-start: the newest `qa.keep_last_runs` runs
survive (default `20`; see [configuration.md](configuration.md)).

```text
workspace/runs/<run_id>/qa/20260720T144233Z-a3f9c1/
├── run.json        # the manifest: argv, mode, config digest, versions, started/stopped/duration
├── summary.md      # cumulative whole-run rollup (funnel + per-strategy fates + phase timing)
├── h00/            # elapsed-hour segment 0 — the first hour since the run started
│   ├── counts.md       # that hour's funnel table + per-strategy fates + phase timing
│   ├── counts.json     # the same numbers, machine-readable (funnel: null on a legacy loop)
│   ├── events.jsonl    # every event that arrived this hour, one JSON object per line
│   └── errors.md       # every failed tool call, full untruncated text
└── h03/            # hours are ELAPSED and lazy: an idle hour writes no folder, so 00→03 can jump
    └── …
```

**Start at the manifest.** `run.json` stamps the `run_id`, the CLI `argv`, the resolved `mode`, a
`config_digest` (a 12-char SHA over the resolved settings with API keys excluded — reproducible,
never leaks a secret), the `noctis`/`python` versions, and `started` / `stopped` / `duration_s`. A
`"stopped": null` means the process was killed before the recorder's `close()` — the run is
truncated, though every already-finalized segment and the manifest itself are still on disk (writes
are synchronous, so at worst the current hour's unflushed tail is lost).

**Cumulative summary vs. hour segments.** `summary.md` is the whole-run rollup; each `hNN/` is one
**elapsed** hour since start. Segments are lazy — an idle hour writes nothing, so the folders can
skip (`h00` then `h03`). Per-hour counters reset each segment while the summary holds running
totals, so read `summary.md` for "what did the whole run do" and a segment for "what happened in
that hour".

**Counts vs. detail documents.** Within a run, the *counts* documents are the aggregate and the
*detail* documents are the raw material:

- **counts** (`summary.md`, `counts.md`, `counts.json`) — the funnel (how many distinct strategies
  reached each stage: write attempts → written → backtested → swept → compared → champion /
  rejected, plus the *rejected pre-sweep* early-kill count), a per-strategy fate row for each
  candidate, and phase-time accounting. `counts.json` is the same numbers machine-readable.
- **detail** (`events.jsonl`, `errors.md`) — `events.jsonl` is the raw event stream (below);
  `errors.md` reproduces every failed tool call's text **verbatim and untruncated** inside a fenced
  block, because a debug run's whole value is the full traceback.

**Phase timing always adds up.** The four buckets — `research`, `trading`, `close`, `idle-wait` —
sum to the segment window by construction. `idle-wait` is the honest catch-all: the gap before the
first phase frame and any weekend-wait / stopped tail, so a reader never has to guess at an
unlabelled gap.

### The events.jsonl line

Each line of `events.jsonl` is one arrival-stamped JSON object:

```json
{"t":"2026-07-20T14:05:00.000Z","el":300.0,"phase":"RESEARCH","kind":"tool","tool":"run_sweep","ok":true,"text":"run_sweep(...) -> ok","meta":{"ok":true,"n_trials":40,"n_failed":3,"best":0.9,"tool":"run_sweep","args":{"name":"alpha_reversion"}}}
```

| Key | Meaning |
|---|---|
| `t` | Arrival timestamp — UTC ISO-8601, millisecond precision, trailing `Z` |
| `el` | Elapsed seconds since the run started (one decimal) |
| `phase` | The phase in force when the event arrived (`null` before the first phase frame) |
| `kind` | Event kind: `phase`, `tool`, `say`, `think`, `usage`, `feed`, … |
| `tool`, `ok` | Present **only on tool events** — the tool name and its success flag, lifted from `meta` |
| `text` | The human-readable one-line rendering |
| `meta` | The structured payload. For a tool event: `ok`, the result brief (e.g. `n_trials`/`n_failed` on `run_sweep`), `tool`, and `args` — the call arguments, including `args.name`, the strategy the call worked on |

### jq one-liners over events.jsonl

Run these from a run folder; the `h*/events.jsonl` glob feeds every hour segment in name (time)
order. All five are verified against a real recorded stream.

```bash
# Every failed tool call, with its full error text (the errors.md content, greppable)
jq -c 'select(.ok == false) | {t, tool, text}' h*/events.jsonl

# Sweeps that burned budget — trials that errored (nothing learned, spend gone)
jq -c 'select(.tool == "run_sweep" and (.meta.n_failed // 0) > 0)
       | {name: .meta.args.name, n_trials: .meta.n_trials, n_failed: .meta.n_failed}' h*/events.jsonl

# Per-kind event counts across the whole run (-s slurps the stream into one array)
jq -s 'group_by(.kind) | map({kind: .[0].kind, count: length})' h*/events.jsonl

# Everything one strategy did — its whole funnel trail, in order
jq -c 'select(.meta.args.name == "alpha_reversion") | {el, tool, ok}' h*/events.jsonl

# The phase timeline (when each RESEARCH → TRADING → CLOSE transition landed)
jq -c 'select(.kind == "phase") | {t, phase: .meta.phase}' h*/events.jsonl
```

### Two honesty lines to recognize

A QA report tells the truth even when it has less to report than a table would imply. Two lines
exist precisely so you never misread an absence as an emptiness.

**1. The legacy loop is not funnel-instrumented.** Only the agent research loop emits the funnel
events; the legacy proposer/Optuna path does not. So when a `run --debug` falls back to the legacy
loop (no LLM configured or reachable), a zero-filled funnel would read as "nothing happened" when
the truth is "we did not measure." Instead, `counts.md` / `summary.md` print this line where the
funnel table would go (and the stop echo shows `legacy research loop — funnel not instrumented`):

> research loop: legacy (proposer/Optuna) — funnel not instrumented; counts below cover phase timing only

When you see it, read the phase timing (which *is* measured either way) and treat the funnel as
absent, not zero.

**2. A tripped fail-safe latch names where coverage stopped.** Recording is strictly secondary and
must never crash a run, so on its **first** internal write failure the recorder latches off for
good (one warning, then silent) and stamps a best-effort note into `summary.md` naming the hour
coverage stopped:

> recorder self-disabled after an internal failure during hour h03; coverage stops here — this report is truncated, not a complete run.

(The wording names the open segment — `during hour hNN` — or `before the first event was recorded`
if it tripped before any segment opened; the stop echo shows `recording disabled after an internal
failure — funnel unavailable`.) A truncated report is **not** a complete one: everything after that
point is missing, so do not read its funnel or counts as final.

## AI-assisted development

Noctis is developed in close collaboration with AI coding agents (primarily
[Claude Code](https://claude.com/claude-code)). That is a statement of process, not of
standards: every change — agent-written or human-written — passes the same gates before it
lands: the full test suite, mypy, ruff, and CI on the minimal install. It is the same
philosophy the system applies to its own strategies: provenance does not earn trust,
surviving the gates does.
