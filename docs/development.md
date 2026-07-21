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
```

## Reading a QA report

`--debug` (on both `noctis run` and `noctis research`, see
[cli.md](cli.md#qa-report---debug)) records everything a session did to a per-run report tree
under `qa_dir` — default `workspace/qa/<run-id>/`, so it follows `workspace_dir` /
`NOCTIS_WORKSPACE` relocation and, like the rest of the workspace, never enters git. The run id
(`20260720T144233Z-a3f9c1`) is a sortable UTC-stamped, greppable name, so a plain `ls` of the QA
area is already chronological. Retention is prune-on-start: the newest `qa.keep_last_runs` runs
survive (default `20`; see [configuration.md](configuration.md)).

```text
workspace/qa/20260720T144233Z-a3f9c1/
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
