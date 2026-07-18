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

## Observability

```bash
python -m noctis status                    # resolved mode, market state, next transition, champions
python -m noctis report [--as-of DATE]     # generate / print the close-of-day report
python -m noctis account [--reset]         # the continuous paper account; --reset archives + starts fresh
python -m noctis champions [--reset]       # the champion board; --reset re-fills slots under current gates
```

## Research & strategies

```bash
python -m noctis research -v               # one observable agent research session (needs a configured LLM)
python -m noctis research --metric total_return   # override the promotion metric for this session
python -m noctis strategies                # the strategy library: status / style / thesis / tuned
python -m noctis backtest <name>           # replay a library strategy on its shipped Params defaults
```

`research` accepts the same `--mandate` / `--directive` one-session overrides as `run`.

## Data

```bash
python -m noctis data status               # tracked series in the coverage registry
python -m noctis data ingest AAPL --start 2024-01-01 --end 2024-12-31 [--dry-run]
python -m noctis data sync                 # tail-only incremental catalog sync
```

`--dry-run` prices an ingest without spending; every ingest is coverage-diffed and
budget-gated — see [data.md](data.md).
