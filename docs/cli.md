# CLI reference

All commands run as `python -m noctis <command>` (or `noctis <command>` with the venv active).
A bare `noctis run` contacts no external service; `research` and the verdict tools need a
configured LLM (see [research.md](research.md)).

## Setup

```bash
python -m noctis setup                # guided first-run wizard: files, components, keys, LLM
python -m noctis setup --check        # read-only install audit; exit 1 on gaps
python -m noctis init                 # scaffold local config.yaml/.env/mandate + the workspace
python -m noctis migrate [--dry-run]  # move a pre-workspace layout into workspace/
```

`setup` is the one command a fresh machine needs after `uv sync --all-extras`: it scaffolds
the local files, offers to install any missing optional components, prompts for the DataBento
key (written into `.env`), connects an LLM — paste a hosted API key, or it detects a local
Ollama/noctis-ollama server and writes the `config.yaml` wiring itself — and then **verifies
the model actually answers** with one real completion before pointing you at `noctis run`.
It is idempotent and edit-preserving: existing files are kept, and config/env edits are
surgical (comments and unrelated lines survive), so re-running is always safe. Unattended
use: `--yes` takes every default and never prompts; `--databento-key` / `--model` /
`--api-key` pre-answer individual prompts; `--check` audits without changing anything (the
scriptable "is this install healthy?").

`init` is the non-interactive core of `setup`: it copies each committed template
(`config.example.yaml`, `.env.example`,
`mandate/MANDATE.md.example`) to its local, gitignored name and creates `workspace/` — the one
output root everything the engine writes lands under. It is idempotent and never overwrites:
re-running after edits is always safe. `migrate` moves the six legacy artifacts (`state/`,
`data_lake/`, `reports/`, root `MEMORY.md`, `strategies/__tmp|champions`) into the workspace;
it refuses with a list when a legacy artifact and its workspace counterpart both exist, notes
knobs explicitly pinned at legacy paths, and never touches `config.yaml`. Until a legacy
layout is migrated, every state-touching command refuses and names `migrate`; `status` warns
but still prints.

## The loop

```bash
python -m noctis run -v                    # start the day/night loop (stops at time_limit_hours)
python -m noctis run -vv --show-reasoning  # narrate each research session's reasoning inline
python -m noctis run --time-limit-hours 8  # override the time limit
python -m noctis run --mandate aggressive  # one-session mandate override
python -m noctis run --directive "..."     # one-session inline directive (excludes --mandate)
python -m noctis run --debug               # also record an hour-segmented QA report under workspace/qa/
```

`run` loads config + memory, resolves the safety gate, enters the correct phase for the current
market clock (RESEARCH while closed, TRADING while open), and loops RESEARCH → TRADING → CLOSE.
It needs catalog data first — ingest history (or let auto-backfill run), then run.
`SIGINT`/`SIGTERM` and the time limit all route through one clean shutdown that stops between
phases and flushes state.

### Verbosity

`run` and `research` share one ladder. A bare command is silent; `-v` streams phase banners and
the research tool feed; `-vv` adds the model's reasoning + narration + per-round token usage and
drops stdlib logging to DEBUG. `--show-reasoning` opens the reasoning/narration streams at `-v`
without the full DEBUG firehose (only providers that return chain-of-thought over the API show
reasoning; narration always shows). Purely observability — it never changes what the system
decides. In `run` the research feed is narrated per session and each RESEARCH → TRADING → CLOSE
transition announces itself inline.

### QA report (`--debug`)

`run` and `research` both accept `--debug`, which records an hour-segmented QA report of the whole
run under `qa_dir` (default `workspace/qa/<run-id>/`): a stamped manifest, funnel counts,
per-strategy fates, phase timing, and a raw `events.jsonl`. It is a diagnostic, not a verbosity
level — it **records silently** and never turns on the `-v` console feed, so `-v --debug` prints
byte-for-byte what `-v` alone prints (just the additive `QA …` framing lines on top). Off by
default; a bare run is byte-identical to today.

At start `--debug` echoes the minted run id and the report path; at stop it echoes the report path
again plus a one-line funnel (`written=… backtested=… swept=… compared=… champions=… rejected=…`),
or, if the recorder self-disabled mid-run, a note that says so instead of a comforting all-zeros
line. Retention is prune-on-start: the newest `qa.keep_last_runs` runs survive (default `20`), and
everything QA lands under the gitignored `workspace/` — nothing reaches git. To read the tree it
produces — the manifest, the cumulative summary, the hour segments, counts vs. detail documents,
plus `jq` one-liners over `events.jsonl` — see
[development.md → Reading a QA report](development.md#reading-a-qa-report).

## Observability

```bash
python -m noctis status                    # resolved mode, market state, next transition, champions
python -m noctis mandate <name>            # preflight a mandate: provenance + the effective settings diff
python -m noctis report [--as-of DATE]     # generate / print the close-of-day report
python -m noctis account [--reset]         # the continuous paper account; --reset archives + starts fresh
python -m noctis champions [--reset]       # the champion board; --reset re-fills slots under current gates
```

### What `status` reports about steering

`status` resolves the **whole** session the way `run` does (gate → mandate → overlay), so every
value it prints is the **post-overlay** one a run would actually use, not what `config.yaml` said
before the mandate touched it. It closes with the provenance: which mandate steered this
configuration and every `k=v` override that applied.

```
research_budget:   17 min
research model:    ollama_chat/noctis-qwen3:14b
data provider:     databento (budget $3.5)
trading driver:    replay (execution=auto)
mandate:           profile:homelab
overrides:
  data.history_days=45
  promotion.metric=sortino
```

With no mandate the last two lines read `mandate: none (unconstrained)` and
`overrides: none` — "unconstrained" is a configuration too, so it is said out loud. The override
lines are read back off the validated settings, so they show the value the run will use rather
than the value the file asked for.

A **mandate the overlay refuses** is reported, not raised: `status` is the command you run
*because* something is wrong, so it still exits 0 and prints the refusal verbatim under

```
mandate:           UNUSABLE — the values above are pre-overlay; `run` would refuse to start:
```

`run` and `research` still exit 1 on that same mandate, which is where a fatal configuration
error belongs. A `SafetyGateError` is *not* degraded this way — an unresolvable mode is not a
configuration `status` can narrate, so it still exits 1.

### The kickoff echo

`run` and `research` both echo the resolved mandate and every applied override before any work
starts, so the assembled configuration lands in the log an operator already reads:

```
Mandate: profile:homelab
  mandate profile:homelab overrides:
    data.history_days=45
    promotion.metric=sortino
```

Same lines, same order as `status`, so the two surfaces can never disagree. No mandate — or a
mandate whose overlay applied nothing — prints nothing extra.

### Preflighting a mandate

```bash
python -m noctis mandate homelab           # what would this mandate actually do?
```

`mandate <name>` is the dry run before committing a machine to a multi-day loop. `<name>` is a
selector in the same vocabulary `--mandate` takes (a profile name, `MANDATE`, or `auto`). It
resolves through the same composition root a real session does — so what it prints is what a run
would get, not a second reading of the precedence chain — and then **starts nothing**: no
research session, no LLM client, no orders.

```
mandate:           profile:homelab
summary:           A small-context homelab personality — local coder, tight spend.
symbols:           SMR, CCJ, LEU
references:
  references/watchlist.md (49 bytes)
overrides:
  data.budget_usd                125.0 → 3.5
  data.history_days              365 → 45
  promotion.metric               sharpe → sortino
  research.agent.context_window  None → 32768
  research.model                 openai/gpt-5.4 → ollama_chat/noctis-qwen3:14b
  research_time_budget_minutes   60 → 17
```

The **effective settings diff** is the part `status` cannot show: every path the overlay binds,
with the value config resolved for it *and* the value the run would use. A mandate with no
`config:` block still prints its provenance, and reads `overrides: none (this mandate binds no
settings)`. `auto` binds nothing by contract (the agent picks its profile mid-session, long after
settings are assembled), so its diff is empty too.

It **exits non-zero** on any refusal, wrong-direction clamp, invalid value, or unresolvable
selector, printing exactly what startup would print — every problem at once, each with the reason
for that path — so a cron job or a shell script can gate on it:

```
$ python -m noctis mandate sneaky; echo $?
MANDATE: mandate profile:sneaky — 3 config overrides refused:
  - promotion.max_gap: promotion gates + metric robustness — …
  - research.metrik: not a setting — check the spelling and the dotted path against config.example.yaml
  - research.min_trials: may only be raised by an overlay — 2 is below the configured 8
1
```

## Research & strategies

```bash
python -m noctis research -v               # one observable agent research session (needs a configured LLM)
python -m noctis research --metric total_return   # override the promotion metric for this session
python -m noctis research --debug          # record this session's QA report under workspace/qa/
python -m noctis strategies                # the strategy library: status / style / thesis / tuned
python -m noctis backtest <name>           # replay a library strategy on its shipped Params defaults
```

`research` accepts the same `--mandate` / `--directive` one-session overrides as `run`, and the
same `--debug` QA recorder (see [QA report (`--debug`)](#qa-report---debug) above). A `research`
session only records when the agent loop is actually buildable — it never opens a report tree for a
legacy session that would immediately exit. `--metric` is applied **after** the mandate overlay, so
it wins over a mandate's `promotion.metric` (as `--time-limit-hours` does over its knob on `run`);
everything else a mandate binds comes from the mandate file —
[configuration.md](configuration.md#the-mandate-overlay).

## Data

```bash
python -m noctis data status               # tracked series in the coverage registry
python -m noctis data ingest AAPL --start 2024-01-01 --end 2024-12-31 [--dry-run]
python -m noctis data sync                 # tail-only incremental catalog sync
```

`--dry-run` prices an ingest without spending; every ingest is coverage-diffed and
budget-gated — see [data.md](data.md).
