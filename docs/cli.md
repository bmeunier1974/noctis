# CLI reference

All commands run as `python -m noctis <command>` (or `noctis <command>` with the venv active).
A bare `noctis run` contacts no external service; `research` and the verdict tools need a
configured LLM (see [research.md](research.md)).

## Setup

```bash
python -m noctis setup                # guided first-run wizard: files, components, keys, LLM
python -m noctis setup --check        # read-only install audit; exit 1 on gaps
python -m noctis init                 # scaffold local config.yaml/.env/mandate + the workspace
python -m noctis migrate [--dry-run]  # move a legacy layout in; adopt state into run `legacy`
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
re-running after edits is always safe.

`migrate` handles both legacy generations in one pass. It moves the six pre-workspace artifacts
(`state/`, `data_lake/`, `reports/`, root `MEMORY.md`, `strategies/__tmp|champions`) to where the
engine reads them now, and **adopts** a pre-run-scoped `workspace/state|reports|memory|qa|
strategies` into the reserved `legacy` run — so an existing operator's champions, account and
reports survive and become their first resumable run, with its own `run.json` listed by
`noctis runs`. It refuses with a list (and moves nothing) when a destination already exists or
two legacy copies claim one, notes knobs explicitly pinned at legacy paths, never touches
`config.yaml`, and is a no-op when run twice. `--dry-run` prints the whole plan, adoption
included, without moving a byte. Until a **pre-workspace** layout is migrated every
state-touching command refuses and names `migrate` (`status` warns but still prints); un-adopted
workspace state only warns, on every command.

## The loop

```bash
python -m noctis run -v                    # start the day/night loop (stops at time_limit_hours)
python -m noctis run -vv --show-reasoning  # narrate each research session's reasoning inline
python -m noctis run --time-limit-hours 8  # override the time limit
python -m noctis run --resume <run_id>     # continue an existing run under its frozen config
python -m noctis run --mandate aggressive  # one-session mandate override
python -m noctis run --directive "..."     # one-session inline directive (excludes --mandate)
python -m noctis run --debug               # also record an hour-segmented QA report under the run's qa/
```

`run` loads config + memory, resolves the safety gate, enters the correct phase for the current
market clock (RESEARCH while closed, TRADING while open), and loops RESEARCH → TRADING → CLOSE.
It needs catalog data first — ingest history (or let auto-backfill run), then run.
`SIGINT`/`SIGTERM` and the time limit all route through one clean shutdown that stops between
phases and flushes state.

Every invocation **mints a new run** and echoes its id and record path at start
(`Run: 20260727T142233Z-a1b2c3`, `Run record: workspace/runs/<run_id>/run.json`). Identity is
minted, never derived from the configuration: two byte-identical configs are two runs. The run
owns a tree under `runs_dir` (default `workspace/runs/<run_id>/`) holding one self-describing
`run.json` — identity, lifecycle, one entry in `segments[]` per process invocation, the engine
identity that produced it, and the run's events — plus a `run.lock` while the engine holds it.
The record is rewritten at each CLOSE and at stop, atomically; a writer failure logs one warning
and leaves the record marked incomplete rather than taking the run down. A second engine refuses
to open a run another engine already holds; a stale lock (a dead pid on this host, or a
week-cold heartbeat) is stolen with a warning and a recorded event. A run killed mid-segment is
marked `interrupted` the next time it is opened.

### Resuming a run — `--resume <run_id>`

```bash
python -m noctis run                       # night 1: mints a run, echoes its id
python -m noctis run --resume 20260727T142233Z-a1b2c3   # every night after
```

`--resume` continues an existing run instead of minting one: same id, same tree, one more entry in
`segments[]`, and the same record keeps accumulating research hours, trials, champions and P&L. The
**run**, not the process, is the unit progress is tracked on — stop the engine each morning, resume
it each night, and a multi-week experiment survives every restart. `noctis runs` lists the ids;
the kickoff echoes `Resumed run: <run_id>`.

Every cumulative number in the record is **derived, never incremented**: recomputed at each write
from the durable artifacts plus the append-only `segments[]` list, so three one-hour segments total
exactly what one three-hour segment would, and a crash mid-write cannot double-count. A run killed
between phases is marked `interrupted` on the next open and resumes from there; its unclosed
segment contributes no runtime, because an unclosed segment has no honest duration.

A resumed run reads its **own** state — champions, paper account, memory, strategy tiers, reports
— out of its own tree, exactly as its earlier segments did, and shares the workspace-level data
lake with every other run.

Three things refuse a resume, all before any work starts:

| refusal | why |
|---|---|
| the id names no run | an address an operator typed must not silently become a *new* run |
| the run is `completed` | terminal by design: a published result can never quietly gain segments |
| the resolved mode differs from the run's | a paper run's results may not acquire live segments (see below) |

**Config is frozen at run creation** (the full contract:
[configuration.md → Config freezing](configuration.md#config-freezing-what-a-resumed-run-reads)).
Editing `config.yaml` or a mandate profile between segments does not change what a running
experiment was told to do; the frozen keys and the run's frozen digest stay put. The current files
still supply the *live* tier — secrets, paths, and the per-process budgets like
`--time-limit-hours`. Because the mandate is frozen, `--mandate`/`--directive`/`--metric` are
**refused** with a reason on a resume rather than silently ignored: start a new run to research
something else.

The safety gate is never rehydrated. It re-resolves from `config.yaml` + `ALLOW_LIVE` at every
process start, so `mode: live` without `ALLOW_LIVE` is the same hard startup error on a resume as
on a first start, and a run whose frozen mode disagrees with the freshly resolved one refuses to
continue.

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
run under `qa_dir` (default `workspace/runs/<run_id>/qa/<run-id>/`): a stamped manifest, funnel counts,
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
python -m noctis runs [--all]              # the run board: id, label, status, segments, headline numbers
python -m noctis run-record <run_id>       # print one run's whole record (pipe it into jq)
python -m noctis status                    # resolved mode, market state, next transition, champions
python -m noctis mandate <name>            # preflight a mandate: provenance + the effective settings diff
python -m noctis engine                    # engine identity: version, component fingerprint, comparable key
python -m noctis report [--as-of DATE]     # generate / print the close-of-day report
python -m noctis account [--reset]         # the continuous paper account; --reset archives + starts fresh
python -m noctis champions [--reset]       # the champion board; --reset re-fills slots under current gates
```

### The run board — `runs` and `run-record`

```bash
python -m noctis runs                      # the experiments worth comparing
python -m noctis runs --all                # …plus the noise
python -m noctis run-record <run_id> | jq .run
```

`runs` lists this workspace's runs newest first, so an experiment can be found and compared
without opening a file:

```
run                      label              status      segments   runtime  comparable key
20260730T025536Z-bc14eb  sector-specialist  stopped            1     1d01h  1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe
20260727T142233Z-7a8f9d  nightly-momo       running            4     2d12h  1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe
20260101T000000Z-brokn0  -                  unreadable         -         -  an unreadable run.json (JSONDecodeError)

1 short run(s) hidden; pass --all to list them.
```

The last column is the **comparable key** (see [Engine identity](#engine-identity--are-these-two-runs-comparable)):
runs may only be pooled or ranked against each other within one key, so the board is partitioned
structurally rather than from memory. A run whose record is missing or unreadable is *listed as
such*, with the reason where its key would be — a broken record is evidence, and hiding it would
be the one thing a listing must never do.

**The default listing hides noise**: a run that finished with under 60 seconds of cumulative
runtime produced nothing to compare (a startup failure, a mistyped command, a config typo). The
count of what was hidden is always printed, and `--all` shows everything. Two kinds are never
hidden whatever their runtime — a run that is still `running`, and a run whose record could not
be read.

`run-record <run_id>` prints that run's whole `run.json` on stdout. The record has **no
sidecars**: one file holds everything about the run, which is exactly what a website `fetch()`es
and what `jq` reads here. It exits non-zero when no run answers the id, or when that one run's
record cannot be read (the listing tolerates a broken record because it has others to show; a
command asked for exactly one does not).

Beside the run trees, `workspace/runs/index.json` is a **derived** roll-up of the same entries —
one `fetch()` for a listing page. Derived, never authoritative: the engine refreshes it after
every record write, `noctis runs` regenerates it from the records on disk, and a test pins that a
rebuild reproduces the incrementally-maintained file byte for byte. Delete it whenever you like.

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

### Engine identity — "are these two runs comparable?"

```bash
python -m noctis engine                    # the declared version, what moved, and the key
```

Two runs' numbers are comparable only if the same engine produced them. A promotion threshold
moved, a prompt reworded, a shipped profile edited, a seed strategy changed: each shifts results
without a single config key differing. `engine` prints that identity in one screen.

```
engine version:    1
components:
  gates        f63d47b7b9604ab1  (arbiter — binds comparability)
  backtest     3ba3e0bf1c97134f  (arbiter — binds comparability)
  research     4baf9dea0c82c8cc
  prompts      14eb169506a6b5aa
  profiles     6803b9d26c63d6ae
  seeds        4826fe7224641eb4
  memory_seed  3337fa2cbf896932
  schema       null
               missing input(s): src/noctis/reporting/schema.py
election metric:   sharpe
comparable key:    1|f63d47b7b9604ab1|3ba3e0bf1c97134f|sharpe
```

`engine version` is a plain incrementing integer versioning the **behavioural contract**,
deliberately decoupled from the package version: a release that changes no behaviour must not
fragment comparison buckets, and a one-line gate change that ships in no release must. The
fingerprint beneath it is **per component**, not one opaque hash, because "a prompt was reworded"
and "a gate moved" are not the same news: two runs with different `prompts` digests but identical
`gates` and `backtest` digests still have comparable scorecards.

The **comparable key** — `(engine_version, gates_digest, backtest_digest, election_metric)` — is
the tuple two runs must match on before their champion and scorecard numbers may be pooled, ranked
or plotted together. The two arbiter digests carry that guarantee rather than the declared version,
because a digest cannot be forgotten in review; the election metric rides along for the reason
promotion already treats a differently scored champion as *stale* (numbers in different units were
never comparable), and it is the **post-overlay** metric, so a mandate that binds `promotion.metric`
is reflected here.

Digests cover committed files only — the shipped `mandate/` scaffold, the seed strategies,
`MEMORY.seed.md` and the engine modules that decide what passes, what a number means, how
candidates are found and what the model is told. An operator's gitignored `mandate/MANDATE.md`,
custom profiles and personal references are deliberately **out**, so the same checkout fingerprints
identically on every machine. Content is hashed raw (LF-normalized), *not* stripped of comments:
prompt text is indistinguishable from a comment to any safe automated rule, so a docstring edit
moving a component is the accepted cost of never silently pooling incomparable runs. A missing
input — like the run-record `schema` module before it lands — reads `null` with a note, never a
crash.

## Research & strategies

```bash
python -m noctis research -v               # one observable agent research session (needs a configured LLM)
python -m noctis research --metric total_return   # override the promotion metric for this session
python -m noctis research --debug          # record this session's QA report under the run's qa/
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
