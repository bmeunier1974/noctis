# CLI reference

All commands run as `python -m noctis <command>` (or `noctis <command>` with the venv active).
A bare `noctis run` contacts no external service; `research` and the verdict tools need a
configured LLM (see [research.md](research.md)). What the verbs below write into `run.json` is
documented field by field in [run-record.md](run-record.md).

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
python -m noctis run --time-limit-hours 8  # bound THIS process (tonight); the run stays resumable
python -m noctis run --run-limit-hours 100 # bound the whole RUN; at the cap it is `completed`
python -m noctis run --embed-all-sources   # archive EVERY candidate's source in the record
python -m noctis run --label nightly-momo  # attach a human alias to this run
python -m noctis run --resume <address>    # continue a run (id | latest | run.json path | @label)
python -m noctis run --resume latest --finish   # seal a run as completed; runs no segment
python -m noctis run --resume latest --show-config-drift   # what would adopting the files change?
python -m noctis run --resume latest --rebase-config       # adopt them, on the record
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

### Resuming a run — `--resume <address>`

```bash
python -m noctis run --label nightly-momo                # night 1: mints a run, echoes its id
python -m noctis run --resume 20260727T142233Z-a1b2c3    # by id — the identity, always
python -m noctis run --resume latest                     # the most recently active resumable run
python -m noctis run --resume workspace/runs/20260727T142233Z-a1b2c3/run.json   # by record path
python -m noctis run --resume @nightly-momo              # by label
```

`--resume` continues an existing run instead of minting one: same id, same tree, one more entry in
`segments[]`, and the same record keeps accumulating research hours, trials, champions and P&L. The
**run**, not the process, is the unit progress is tracked on — stop the engine each morning, resume
it each night, and a multi-week experiment survives every restart. `noctis runs` lists the ids;
the kickoff echoes `Resumed run: <run_id>` — the **id** of the run reached, whatever address got
you there.

`noctis research --resume <address>` continues the same run with a **research-only** segment, using
the same lock, the same frozen config and the same run-scoped state — see
[A research session belongs to a run](#a-research-session-belongs-to-a-run--research---resume-address).

#### The four address forms, and how they are told apart

An address is resolved in one place (`reporting/run_tree.resolve_run_dir`, shared with
`run-record`), in this **fixed order**, so one string always names one run whatever a workspace
happens to contain:

| # | form | matches when | resolves to |
|---|---|---|---|
| 1 | **path** | it contains `/` or `\`, or is `run.json` / `.` / `..` | that `run.json`'s directory, or that run directory — honoured wherever it points, including outside `runs_dir` |
| 2 | **`@label`** | it starts with `@` | the one run carrying that label; if none does, the same name read as an **id** |
| 3 | **`latest`** | it is exactly `latest` | the most recently active run that is not `completed` |
| 4 | **run id** | anything else | `runs_dir/<id>` |

Three rules follow, and they are the whole disambiguation story:

- **A bare address is always the id.** A run *labelled* `nightly-momo` is not addressed by
  `nightly-momo` — the refusal says so and names `@nightly-momo`. So a label that looks exactly
  like another run's id can never shadow that run.
- **`@` means "label first".** `@<name>` looks `name` up as a label and only then as an id, so an
  id typed (or pasted) with a leading `@` still resolves, and a run labelled with another run's id
  is still reachable — as `@<that-id>`.
- **`latest` is a reserved word, not a lookup.** It means the same thing in every workspace, so a
  run that happens to be *named* `latest` never captures it (address that one by its path) and one
  *labelled* `latest` never does either (address it as `@latest`).

**"Most recently active"** is read off the record — `run.last_active_utc`, falling back to
`created_utc` — and never off a filesystem mtime, which lies after a copy or a migration; ties
break on the run id, so the answer is deterministic rather than dependent on directory order.
`latest` skips `completed` runs (they refuse resume anyway) and runs whose record cannot be read
(they carry no stamp to be "most recent" by — address those by id). It does **not** skip a
`running` run: that is usually the one you mean, and if another engine really holds it the
liveness lock refuses a moment later. With nothing resumable left, `--resume latest` fails and
says what it found instead.

#### Labels — `--label`

`--label nightly-momo` attaches a human alias. It is stored in the **record** (the source of
truth) and reaches `index.json` only by derivation, so `noctis runs` shows it and a rebuilt index
still carries it. `--label` is also accepted **with `--resume`**, where it renames the run it
addressed: a label decides nothing, so fixing a typo'd nickname must not cost a run — unlike the
frozen config, which a resume refuses to move.

A label is **convenience only: the id is the identity.** A label may be reassigned to a second
run; when it is, both runs keep their own ids, records and history — and `@nightly-momo` then
answers with a **refusal naming both candidate ids** rather than picking one, because an alias
that silently chose between two runs would eventually append a night's work to the wrong record.
Address the one you mean by its id (or relabel the other on its next resume).

Every cumulative number in the record is **derived, never incremented**: recomputed at each write
from the durable artifacts plus the append-only `segments[]` list, so three one-hour segments total
exactly what one three-hour segment would, and a crash mid-write cannot double-count. A run killed
between phases is marked `interrupted` on the next open and resumes from there; its unclosed
segment contributes no runtime, because an unclosed segment has no honest duration.

A resumed run reads its **own** state — champions, paper account, memory, strategy tiers, reports
— out of its own tree, exactly as its earlier segments did, and shares the workspace-level data
lake with every other run.

Four things refuse a resume, all before any work starts:

| refusal | why |
|---|---|
| the address names no run | an address an operator typed must not silently become a *new* run |
| the address names more than one | a reassigned label has no honest single answer; the refusal lists the candidate ids |
| the run is `completed` | terminal by design: a published result can never quietly gain segments |
| the resolved mode differs from the run's | a paper run's results may not acquire live segments (see below) |

**Config is frozen at run creation** (the full contract:
[configuration.md → Config freezing](configuration.md#config-freezing--what-a-resumed-run-reads)).
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

#### Bounding a run — `--run-limit-hours` and `--finish`

```bash
python -m noctis run --run-limit-hours 100          # "spend 100 hours on this, then stop"
python -m noctis run --resume latest                # …resume as usual; the cap keeps counting
python -m noctis run --resume @nightly-momo --finish  # publish it: completed, and terminal
```

**Two ceilings, one shutdown.** `--time-limit-hours` bounds *this process* — how long tonight
lasts — and leaves the run resumable, exactly as it always has. `--run-limit-hours` bounds the
*run*: the hours of cumulative runtime it may accumulate **across every stop and resume**. Once the
run's total crosses the cap the loop stops cleanly between phases (the same shutdown path the time
limit and `SIGINT` use — there is no second route), the segment closes with
`stopped_reason: run_limit`, and the run becomes `completed`.

That is how *"run this mandate for 100 research hours, then stop"* is expressed, and it is what
lets two runs be compared **on equal compute**: a mandate given 100 hours and one given 30 are not
the same experiment, however similar their configs.

- **Frozen at creation.** The cap is accepted when a run is minted (flag, `config.yaml`, or a
  mandate's `config:` block) and pinned into the record. Editing `config.yaml` afterwards does not
  move it, and `--run-limit-hours` **with** `--resume` is refused with a reason — a cap that could
  be raised each morning would bound nothing.
- **Derived, never counted in memory.** `run.cumulative_runtime_s` (and the
  `cumulative_research_s` / `cumulative_trading_s` beside it, summed from each segment's
  `phase_seconds`) are recomputed from `segments[]` at every write, so the breach is a property of
  the record rather than of whichever process happened to notice it.
- **`completed` is terminal.** Every later resume is refused, naming the cap and the runtime spent.
  A segment that ends *below* the cap leaves the run `stopped` and resumable as normal.

`--finish` is the deliberate twin: it marks a run `completed` and exits. It opens **no segment** and
starts no engine — it only reads the liveness lock far enough to refuse a run another process is
working — so it is the command for "this result is published". On a run that is already
`completed` it is a **documented no-op**: it says so and leaves the original seal stamp alone.

`noctis runs` shows the state of the budget in the runtime column (`2h00m/100h`), and every record
and index entry carries `run_limit_hours` so a leaderboard can group like-for-like.

#### Archiving a run whole — `--embed-all-sources`

```bash
python -m noctis run --embed-all-sources          # keep every candidate's code in the record
```

The record's `strategies[]` lists **every** candidate the run considered — champions, rejections
and the drafts that never reached a verdict — each with the structured gate results behind its
outcome, so *"47 of 66 candidates died at the symbol-holdout gate"* is a number a reader computes
rather than a claim the page makes. Sources follow one rule: a **champion's** file is embedded in
full, and every other candidate is named by `source_path` (relative to the run directory) plus
`source_sha256`. That is what keeps a fortnight's record in the hundreds of kilobytes instead of
megabytes, and it is why the record stays a single `fetch()` a website can render.

The cost of that rule is stated rather than hidden: a rejected candidate's *code* is readable only
while the run's own `strategies/__tmp/` tier survives — which is as long as the run does, since
[retention](#reclaiming-disk--run-prune-address) refuses to prune anything resumable and the
record's `run.state_pruned` says when it finally went. `--embed-all-sources` is the deliberate
alternative for an experiment worth keeping after its workspace is not: every candidate's source
lands *inside* the record, which then survives a prune, a copy, or a move to another machine.

Like `--run-limit-hours`, it is **accepted at run creation only and frozen into the record**
(`embed_all_sources` in `config.yaml` does the same for the next run you mint). Passing it with
`--resume` is refused with a reason: the record is rewritten whole at every write, so a flag
supplied on some nights and not others would make what the record *contains* depend on how it was
last invoked — and an earlier night's embedded sources would silently vanish.

#### Config drift: seeing it, and adopting it

```bash
python -m noctis run --resume latest --show-config-drift   # look: what would I be adopting?
python -m noctis run --resume latest --rebase-config       # decide: adopt it, on the record
```

Drift is normal and costs nothing — a resume keeps using the frozen values whatever the files say.
These two flags are the *see it* and the *adopt it deliberately*, and they are deliberately
separate commands to type (passing both is refused): looking first must never be a decision.

`--show-config-drift` prints the diff and exits. It is an **inspection**: no segment is opened, no
lock is taken, and not one byte of the record is rewritten.

```
Config drift for run 20260727T142233Z-a1b2c3 (config_epoch 1, frozen at 2026-07-27T14:22:33.418Z):

settings (frozen tier — a resume ignores the current files for these):
  champion_count    5 → 1
  promotion.metric  'sortino' → 'total_return'
mandate (frozen as resolved text, not as a selector):
  source       profile:aggressive → profile:aggressive
  text_sha256  44502474589f → 4842800e26f7
  frozen text  Trade the most volatile names. Risk appetite: high.
  current text Buy and hold index funds. Risk appetite: low.
```

**What counts as drift** is exactly what freezing covers, and nothing else:

| compared | not compared |
|---|---|
| every **frozen** key (`promotion.*`, `research.*`, `universe`, …) | the **live** tier — paths, secrets, per-process budgets. It is this process's *by design*, so it is not a difference to adopt |
| the **resolved mandate text** (and its sha256) — so rewriting `mandate/profiles/aggressive.md` behind an unchanged selector shows up here | the mandate **selector**. The same bytes reached through a renamed profile changed nothing about what the run was told |
| | `mode` / `allow_live` — never recorded, never restored, never rebasable |

`--rebase-config` adopts the current files for the rest of the run: it re-freezes them onto the
record, bumps `inputs.config_epoch`, and appends a before/after entry to `inputs.config_changes`
naming the **segment** it happened in. A mid-run config change is never silent — a record whose
config changed has to say so *and say where*, or every comparison built on it is false. From then
on the new values are the run's own: the next resume restores those.

**With no drift it is a documented no-op.** The epoch does not move, no entry is written, and the
resume proceeds normally — a cosmetic bump would mark the run as mixed-config forever and make
every consumer that renders `config_epoch > 1` as "this run changed mid-flight" a liar.

`mode` and `allow_live` are **never rebasable under any flag**. `mode` is not even in the frozen
settings — the record keeps only the gate's verdict — so the concrete attempt (edit `mode`, open
`ALLOW_LIVE`, ask for the current files) is refused by the mode check that runs before any rebase,
with a message saying no flag lifts it. The live-money double gate re-resolves from two independent
sources at every process start (AGENTS.md rule 1); a record is neither of them.

#### Engine change: resuming after the code moved

The configuration is one half of "what these numbers mean"; the **engine** is the other. A run
freezes its engine identity at creation — the declared `ENGINE_VERSION` plus one digest per
behavioural component (see [Engine identity](#engine-identity--are-these-two-runs-comparable)) —
and every resume compares that against this checkout, component by component. The policy splits on
**who changed: the judge, or the searcher**, and it is deliberately the same line the CI ratchet
draws (see [development.md → Engine fingerprint ratchet](development.md#the-engine-fingerprint-ratchet)).

| what moved | on resume |
|---|---|
| `gates`, `backtest` — the **arbiter**: what passes, and what a number means | **Refused**, naming the component and both digests, before a segment is opened or a lock is taken. Champions crowned under two sets of gates must not accumulate inside one experiment — inside a *single* run that is worse than across two |
| `research`, `prompts`, `profiles`, `seeds`, `memory_seed`, `schema` — the **searcher** | **Warn, record, proceed.** An event lands on the record naming the component, both digests and the files to look at. Improving how candidates are *found* must not invalidate a run whose arbiter held still |
| nothing | Silence. Nothing printed, nothing recorded |

```bash
python -m noctis run --resume latest --allow-engine-upgrade   # accept an arbiter change, on the record
```

`--allow-engine-upgrade` is the escape hatch, and it is **never invisible**: it bumps
`engine.engine_epoch`, appends an `engine.engine_changes` entry naming every component that moved
(with both digests and the **segment** it happened in), re-freezes the run onto the new engine — so
its comparable key honestly follows it into the new bucket — and flags the run `mixed_engine` for
good, which `noctis runs` shows beside the key. With no arbiter drift it is a documented **no-op**:
the epoch never moves for nothing, exactly like `--rebase-config`.

The alternatives the refusal names are the honest ones: restore the engine the run was created
under (its digests are in the record), or start a new run under the current engine — identity is
minted, never derived, so a fresh run under the same configuration is one command away.

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
python -m noctis run-record <address> [--validate]   # print one run's whole record, or schema-check it
python -m noctis run-prune <address> [--dry-run]   # reclaim a completed run's heavy directories
python -m noctis status                    # resolved mode, market state, next transition, champions
python -m noctis mandate <name>            # preflight a mandate: provenance + the effective settings diff
python -m noctis engine                    # engine identity: version, component fingerprint, comparable key
python -m noctis report [<address>] [--as-of DATE]   # generate / print one run's close-of-day report
python -m noctis account [--reset]         # the continuous paper account; --reset archives + starts fresh
python -m noctis champions [--reset]       # the champion board; --reset re-fills slots under current gates
```

### Reading the run you just ran

Three commands read a finished run back — the board, the record, the report. They are the same
three the README's try-it example ends on, verbatim:

```bash
uv run python -m noctis runs                                  # id, label, status, segments
uv run python -m noctis run-record latest | jq .performance   # the paper account's numbers
uv run python -m noctis report latest                         # the close-of-day report
```

Each step narrows. [`runs`](#the-run-board--runs-and-run-record) finds the run — its id, whether
it is still `running`, and the [comparable key](#engine-identity--are-these-two-runs-comparable)
that says what it may be ranked against. `run-record` prints that one run's whole `run.json`,
which is why piping into `jq` works at all: the record has no sidecars, so `.performance` sits in
the same document as the config that run froze, its engine identity and every candidate's gate
evidence. It reads `null` until the run has actually traded — a night of pure research reports no
performance rather than a flat zero. [`report`](#the-close-of-day-report--report-address) prints
that run's close-of-day report, which is a file under the run's own tree, at
`workspace/runs/<run_id>/reports/YYYY-MM-DD.md`.

`latest` is one of the four
[address forms](#the-four-address-forms-and-how-they-are-told-apart), so the same three commands
take an id, a `run.json` path or `@label` when the run wanted is not the newest. Dropping the
address is not the same question: a bare `report` reads the reserved `legacy` run, not the run
that just finished — see [the close-of-day report](#the-close-of-day-report--report-address) for
what an address does and does not change.

### The run board — `runs` and `run-record`

```bash
python -m noctis runs                      # the experiments worth comparing
python -m noctis runs --all                # …plus the noise
python -m noctis run-record @nightly-momo | jq .run
```

`runs` lists this workspace's runs newest first, so an experiment can be found and compared
without opening a file:

```
run                      label              status      segments      runtime  comparable key
20260730T025536Z-bc14eb  sector-specialist  stopped            1        1d01h  2|0fb6148041c95608|3ba3e0bf1c97134f|sharpe
20260727T142233Z-7a8f9d  nightly-momo       running            4  2d12h/100h  2|0fb6148041c95608|3ba3e0bf1c97134f|sharpe
20260714T031102Z-4d9c1a  gate-rework        completed          6    5d04h/8h  3|8c1de5f0a2b34c77|3ba3e0bf1c97134f|sharpe  (mixed engine)
20260101T000000Z-brokn0  -                  unreadable         -            -  an unreadable run.json (JSONDecodeError)

1 short run(s) hidden; pass --all to list them.
```

A run given a compute cap shows it beside the runtime it has spent (`2d12h/100h`), because two
runs given different compute are different experiments; `completed` is the terminal status a
spent cap or a `--finish` leaves behind ([above](#bounding-a-run----run-limit-hours-and---finish)).

The last column is the **comparable key** (see [Engine identity](#engine-identity--are-these-two-runs-comparable)):
runs may only be pooled or ranked against each other within one key, so the board is partitioned
structurally rather than from memory. A run whose record is missing or unreadable is *listed as
such*, with the reason where its key would be — a broken record is evidence, and hiding it would
be the one thing a listing must never do. **`(mixed engine)`** marks a run that ran more than one
engine — an accepted `--allow-engine-upgrade`, or a segment whose digests differ from the ones the
run was created under. Its key names the bucket its *latest* engine puts it in, while some of its
numbers were produced by another one, so the marker is the part a leaderboard must not be told
twice.

**The default listing hides noise**: a run that finished with under 60 seconds of cumulative
runtime produced nothing to compare (a startup failure, a mistyped command, a config typo). The
count of what was hidden is always printed, and `--all` shows everything. Two kinds are never
hidden whatever their runtime — a run that is still `running`, and a run whose record could not
be read.

`run-record <address>` prints that run's whole `run.json` on stdout, and takes the same four
address forms as [`--resume`](#the-four-address-forms-and-how-they-are-told-apart) — an id,
`latest`, a `run.json` path, `@label` — resolved by the same rules, because an address form
invented twice would eventually resolve two different runs from one string. The record has **no
sidecars**: one file holds everything about the run, which is exactly what a website `fetch()`es
and what `jq` reads here. It exits non-zero when no run answers the address (or when more than
one does), or when that one run's record cannot be read (the listing tolerates a broken record
because it has others to show; a command asked for exactly one does not).

`run-record <address> --validate` **schema-checks** the record instead of printing it, so an
artifact can be verified before it is published:

```bash
python -m noctis run-record latest --validate
# workspace/runs/20260714T031102Z-4d9c1a/run.json: valid against run-record schema version 1 …
```

It checks the whole contract in `reporting/schema.py` — documented field by field in
[run-record.md](run-record.md) — every section present, every absent value
an explicit `null` rather than a dropped key, every timestamp UTC ISO-8601 with a `Z`, every
dimensioned number naming its unit the one canonical way (`_usd`, `_pct`, `_bps`, `_s`, `_bytes`,
`_bars`), every cost field calling itself an estimate, and neither live-money gate anywhere in the
document. It prints **every** problem at once (an operator asking "is this record readable?" wants
the whole list, not the first line of it) and exits non-zero when there is one. Unknown keys are
never a problem: the schema is additive-only, so a record written by a newer Noctis validates here
and a record written by an older one is upgraded in place — with an event saying so — the next time
the run gains a segment.

Beside the run trees, `workspace/runs/index.json` is a **derived** roll-up of the same entries —
one `fetch()` for a listing page. Derived, never authoritative: the engine refreshes it after
every record write, `noctis runs` regenerates it from the records on disk, and a test pins that a
rebuild reproduces the incrementally-maintained file byte for byte. Delete it whenever you like.

### Reclaiming disk — `run-prune <address>`

```bash
python -m noctis run-prune 20260714T031102Z-4d9c1a --dry-run   # what would go, and how much?
python -m noctis run-prune 20260714T031102Z-4d9c1a             # …do it
```

A run's `state/`, `strategies/` and `reports/` directories are the megabytes; its `run.json` is
kilobytes. `run-prune` deletes the first three and **never** the record — nor `index.json` — so a
pruned run still lists in `noctis runs`, still prints in full through `run-record`, and still
carries every number it accumulated. Retention takes the same four
[address forms](#the-four-address-forms-and-how-they-are-told-apart) as everything else that names
a run.

Three properties are the whole design, and each is enforced rather than documented:

- **Opt-in, one run at a time.** Nothing prunes on a schedule, at startup, or from a config
  setting: the only way a byte is removed is an operator typing this verb at one address. The
  default keeps everything, forever.
- **`completed` runs only.** The pruned directories are exactly what a resume reads back, so
  pruning a `stopped`, `interrupted` or `running` run would silently destroy its resumability — the
  one thing the run record promises. All three are refused, with that reason. Seal a run you really
  have finished with (`noctis run --resume <address> --finish`) and prune it then; `completed` is
  terminal, so nothing that could still be continued is ever at risk. A run another engine is live
  on is refused too, whatever its record says.
- **`--dry-run` measures and removes nothing** — the same directories, the same byte count, one
  step short of the removal.

Afterwards the record's `run.state_pruned` reads `true`. That is the flag a reader checks before
following a path into the run tree: **everything the record carries in it survives** — pruning
removes three directories and rewrites one flag — while any reference *out* of the record into
those directories (a path plus a hash, rather than the content itself) no longer resolves. It is
also why the strategy sources that matter are meant to be embedded in the record rather than
pointed at: an embedded champion is readable a year after its run's tree was reclaimed.

### The close-of-day report — `report [address]`

```bash
python -m noctis report                              # the reserved `legacy` run, as always
python -m noctis report latest                       # the run that just finished
python -m noctis report @nightly-momo --as-of 2026-07-27   # a named run, on a named day
```

A run owns its `reports/` ([run-scoped state](configuration.md#the-workspace)), so the report of
the run `noctis run` just minted lives under *that* run's tree. `report <address>` reads it,
taking the same four [address forms](#the-four-address-forms-and-how-they-are-told-apart) as
`--resume`, `run-record` and `run-prune`, resolved by the same rules — an address form invented
twice would eventually resolve two different runs from one string. It exits non-zero when no run
answers the address, or when more than one does.

**With no address nothing changes**: it reads the reserved `legacy` run, which is what an
invocation that never opened a run should read. `--as-of` and `--sweep-stale` compose with an
address exactly as they always have, on the addressed run's own reports.

An addressed run's tree is **authoritative**, and that is the whole rule:

- Its `reports/` are what is printed, its `state/` is what a missing one is assembled from, and
  its `reports/` is where that one is written — no report is ever assembled from, or written
  into, another run's tree.
- The un-migrated-layout guard ([the workspace](configuration.md#the-workspace)) does not run:
  that guard answers *"which tree does an unaddressed command read?"*, and an address answers it.
  A bare `report` beside a pre-workspace layout still refuses and names `noctis migrate`.
- A run whose state [`run-prune`](#reclaiming-disk--run-prune-address) removed is **refused**
  rather than reported on. Its `reports/` and `state/` were deleted on purpose and the record
  says so (`run.state_pruned`), so the only report left to give would be one assembled from
  nothing — an empty champion board attributed to a run that had one. Everything that run
  accumulated is still in `noctis run-record <address>`, which is the refusal's own advice.

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
engine version:    2
components:
  gates        0fb6148041c95608  (arbiter — binds comparability)
  backtest     3ba3e0bf1c97134f  (arbiter — binds comparability)
  research     67528310408eeefc
  prompts      14eb169506a6b5aa
  profiles     6b0e047bd1dafb6d
  seeds        4826fe7224641eb4
  memory_seed  3337fa2cbf896932
  schema       8ff21a89b4298a42
election metric:   sharpe
comparable key:    2|0fb6148041c95608|3ba3e0bf1c97134f|sharpe
```

(Digests are whatever *this* checkout hashes to; run the command for yours.)

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
input — a checkout without the file a component maps to — reads `null` with a note naming what was
missing, never a crash. The full component table, and where the key lands on a record:
[run-record.md → engine](run-record.md#engine--identity-components-comparable-key).

## Research & strategies

```bash
python -m noctis research -v               # one observable agent research session (needs a configured LLM)
python -m noctis research --resume latest  # …append it to an existing run, under that run's config
python -m noctis research --metric total_return   # override the promotion metric for this session
python -m noctis research --debug          # record this session's QA report under the run's qa/
python -m noctis strategies                # the strategy library: status / style / thesis / tuned
python -m noctis backtest <name>           # replay a library strategy on its shipped Params defaults
```

`research` loads config + memory and resolves the safety gate exactly as `run` does, and accepts
the same `--mandate` / `--directive` one-session overrides and the same `--debug` QA recorder (see
[QA report (`--debug`)](#qa-report---debug) above). A `research` session only records when the
agent loop is actually buildable — it never opens a report tree for a legacy session that would
immediately exit. `--metric` is applied **after** the mandate overlay, so
it wins over a mandate's `promotion.metric` (as `--time-limit-hours` does over its knob on `run`);
everything else a mandate binds comes from the mandate file —
[configuration.md](configuration.md#the-mandate-overlay).

### A research session belongs to a run — `research --resume <address>`

**A session is always part of a run.** A bare `noctis research` *mints* one, exactly as `noctis run`
does — identity is minted, never derived, so two sessions under one config are two runs — and
`--resume <address>` appends a **research-only segment** to an existing one instead:

```bash
python -m noctis research --resume 20260727T142233Z-a1b2c3   # by id
python -m noctis research --resume latest                    # …or @label, or a run.json path
```

Everything about the resume is the resume `run` performs, from the same code: the
[four address forms](#the-four-address-forms-and-how-they-are-told-apart) resolve identically, the
run's **frozen config** is rehydrated (so `--mandate` / `--directive` / `--metric` are refused with
a reason rather than silently ignored), the run's own state, strategy tiers, per-run memory and
reports are what the session reads and writes, and the liveness lock refuses a run another engine
is working — while a `completed` run refuses resume outright, because terminal is terminal.

The segment records what a `run` segment records, for a session instead of a night: its own
start/stop stamps and duration, `command: "research"`, `stopped_reason` (the session's own —
`agent_done`, `budget_exhausted`, …), its argv, and its counters
(`sessions` / `research_iterations` / `research_promotions`). **"Research-only" is derived, not a
second flag:** `research` is a verb that cannot trade, so the command each segment was invoked as
is the whole answer. Its measured RESEARCH seconds roll into the run's `cumulative_research_s`, and
the trials it journals into the run's `state/experiments/` roll into `cumulative_trials` — both
re-derived at every write, so a run's research hours and trials accumulate the same whether the
night came from `noctis run` or from a standalone session.

**The segment also carries the gate's verdict.** `research` resolves the safety gate exactly as
`run` does, so a run a session mints freezes `inputs.execution_mode: "paper"` and reports
`assumptions.paper_only: true` — not `null`, which now means only an adopted history that froze no
verdict at all. The session places no order itself, but the run it mints may trade on a later
`run --resume`, and the record has to say what the gate *decided* rather than "nobody measured".
For the same reason no verb is a silent downgrade: `noctis research` under `mode: live` without
`ALLOW_LIVE` refuses at startup with `SAFETY GATE: …`, exactly as `run` does
([safety.md](safety.md)).

**A run that only ever researched is a first-class shape.** It reports `traded: false` and
`performance: null` — never zeros — so a consumer renders "researching" rather than a flat 0% equity
curve it was handed as if it were a result. The pairing is enforced by the record's schema
validator, and it is one of the two cases `performance` is `null` in
([run-record.md](run-record.md#performance)).

One asymmetry worth knowing: `research --resume` has no `--allow-engine-upgrade`. Searcher-tier
engine drift warns and is recorded here exactly as on `run`, but a resume whose **arbiter** moved is
refused, and accepting that deliberately is done once with
`noctis run --resume <address> --allow-engine-upgrade` (see
[Engine change](#engine-change-resuming-after-the-code-moved)).

## Data

```bash
python -m noctis data status               # tracked series in the coverage registry
python -m noctis data ingest AAPL --start 2024-01-01 --end 2024-12-31 [--dry-run]
python -m noctis data sync                 # tail-only incremental catalog sync
```

`--dry-run` prices an ingest without spending; every ingest is coverage-diffed and
budget-gated — see [data.md](data.md).

## Bench

```bash
python -m noctis bench run --site decide --dry-run    # preflight only: print the plan, spend nothing
python -m noctis bench run --site decide [--split tuning|holdout|all] [--reps N] [--workers N] \
                           [--model <provider/model>] [--label <name>]
python -m noctis bench run --site coder --tier smoke  # the declared 12-case tier, six at a time
python -m noctis bench report <bench-id>   # one bench record's reading, straight to stdout
python -m noctis bench corpus --site coder # validate one site's corpus; print its stats and balance
```

The bench area is workspace-level (`<workspace>/bench/<bench_id>/bench.json`), run-neutral like the
data lake, so a bench is addressed by the id it was minted with and never through a run. A corpus is
run-neutral for the same reason — a population, not one run's trajectory — and it is read from **two
tiers**, the same split every committed input has: the repo's `cases/<site>/` (the curated buckets a
review shipped, read-only input, `cases_dir`) and `<workspace>/cases/<site>/` (mined, harvested,
the only one anything ever writes). Both are read, the workspace wins a shared case id — the
strategy library's rule, applied to the population — and `bench corpus` names every tier it read
from, so "20 cases validated" always comes with "from where". Nothing is copied between them on
purpose: a copied corpus goes stale in silence, and the digest is the only thing that would have
said so. A site no tier holds is refused naming both paths rather than read as an empty corpus.

`bench run` **always preflights**. It resolves the site through the registry, loads that site's
corpus from the cases root, counts the jobs (cases × reps × configurations), prices their ceiling
under the current pricing table, and prints that plan before anything is asked. A live run then
hands the very same plan back as the acknowledgement the runner refuses to start without, so a
spend nobody stated is unreachable through this verb, and `--dry-run` simply stops after the
printing: no model client is built (no key and no `[llm]` extra are needed), no directory is made,
nothing is spent. `--workers N` above 1 works the jobs on a pool; the printed `workers:` line states
the width either way.

`--tier <name>` measures a **declared subset** of the corpus instead of all of it. A tier is data,
not logic: `src/noctis/eval/bootstrap.py`'s `SITE_TIERS` names the cases by id, so what a tier
selects is read off the table and reviewed in a diff. The coder ships one — `smoke`, **twelve
cases**: every canary the corpus holds (briefs so plain that a red one indicts the harness) plus the
six edge cases that complete the coverage, so the twelve together touch every level of all seven
difficulty axes. An unknown name is refused listing the tiers that site declares; a site with no
tiers says so rather than accepting a word it cannot honour; and a tier that names a case the corpus
no longer holds is refused naming it, because a partial tier would publish its name over another
population.

**A tier and a `--split` both name what to measure, so stating both is refused** rather than
intersected: filtering a declared twelve-case tier down to its holdout is a handful of cases wearing
the word `smoke`, and the number would then be compared against a tier nobody ran. The whole-corpus
word (`--split all`, and the absence a bare invocation leaves) filters nothing, so it composes with a
tier freely. Selection happens *after* the corpus is dealt, so every case keeps the tuning/holdout
half the whole corpus gave it — a tier never re-deals a split.

**A tiered run sizes its own pool.** With no `--workers`, an untiered run stays sequential (what a
bench has always done — widening it spends against somebody's rate limit and stays an explicit act),
while a tiered one opens `min(cases, 6)` workers: the smoke tier's twelve independent jobs are two
waves of six, so a five-minute smoke target asks that one authoring job finish in about two and a
half minutes — the budget a job already has from its per-attempt timeout and retry budget. The cap is
not the job count on purpose: twelve concurrent completions is a rate-limit decision an operator
takes deliberately, and a stated `--workers` always wins.

The verb names no site. How a case becomes its site's renderer input is a lookup in the eval
layer's ask table (`src/noctis/eval/bootstrap.py`) — DECIDE's frozen cases are reconstructed into
the production briefing's own inputs, a site that declares no adapter is asked with the payload its
cases carry — so every site runs down one code path. A live ask goes through the engine's own LLM
seam and the site's declared emit contract; a site whose live ask is not declared is **refused**
rather than asked under an invented system prompt.

`--site coder` is the one site whose answer no emit contract types, and the ask table says so: its
row declares its own live maker instead of a schema, because the coder's ask is a whole **authoring
job** and its judge is the fresh-subprocess write gate. One runner attempt is one such job — the
engine's private validator retries (and, when the escalation cap allows it, one fallback to the paid
coder) happen *inside* it — and every internal attempt is recorded, so the job-level pass rate the
record carries and the finer first-attempt reading a reader wants are both derivable from what was
retained. Each job builds a throwaway three-tier strategy library inside its own working directory:
the committed `strategies/` seeds are read in place (read-only input — a job authoring a seed's own
name lands its file in its own `__tmp/` tier, exactly as a research session does), and the two
writable tiers are fresh per job, so nothing one case writes is visible to the next and no run's
library, board or state is touched at all. Beside the authored file the job keeps the exact prompt
it sent and every rejected attempt's source and gate error, in the same `failed/` shape
`noctis.eval.failed_attempts` already reads.

Once every job has answered, the run **scores itself through the site's declared scorers** and folds
what they publish into the record's `harness.dials`. There is no flag: a scorer is part of a site's
declaration, so a site that declares one publishes its reading and a site that declares none writes
exactly the record it wrote before. For DECIDE that reading is the same block the retrospective
miner publishes — the co-primary approval pair, the deferral figures, one row per case and the
per-axis strata — computed from the verdicts this bench just earned against the labels the promotion
gates recorded, and marked `answers: fresh` beside the baseline's `answers: recorded`. Three honesty
rules hold it up: a case is the equal-weight unit, so every rep of one case folds into the single
outcome it contributes (by strict majority of the verdicts that parsed); a reply the emit contract
does not admit is a failed attempt and an `unreadable` exclusion, never a row in agreement's
denominator; and a case whose reps hold no majority is counted `unsettled` rather than given a
verdict nobody emitted.

For the coder that reading is the **co-primary pass pair** — the first-attempt rate and the job-level
one, published together or not at all, the job rate always under the words *pass@k with feedback*
because its retries saw the gate's rejection — with the effort, escalation and spend that explain the
gap between them, a **failure-taxonomy table** naming every declared class with the knob its share
points at (a class stays on the screen at zero, because a class that stopped happening and one that
stopped matching look identical the moment a row disappears), and the same pair broken down **per
difficulty axis**, level by level, since a job pass rate that is one thing on `bars_only` briefs and
another on `exits` ones is two findings rather than one.

Refusals come from the machinery and are rendered as one line: an undeclared `--site` names every
site that is declared, an absent corpus names the directory it looked in, a `--split` word that
names no half lists the three it accepts, and a live run with no buildable client names the model
and why. A finished run prints the bench id, the directory it landed in, and whether it completed —
address it with `bench report` from there.

`bench report` prints the record's **own** reading: the identity block, the population it measured
(n, reps per case, and the corpus cases it never scored), whatever the record's `harness.dials`
publish, and the metrics block every bench record carries. The verb knows nothing about the site
that was benched — a record whose dials carry a co-primary approval pair prints that pair
(agreement is never printed without the approval rate it was bought at, and a block carrying one
without the other is *refused* rather than half-printed), and a record from another site with
entirely different strata prints those, through the same code path.

Refusal-first, like everything else that publishes a number here: an unmeasured figure prints `n/a`
and never `0`, a record the schema validator has problems with is refused with **every** problem
rather than half-rendered, and an id nothing answers is refused naming the bench root it looked in.
Reading a bench writes nothing at all.

`bench corpus` asks about the **input** rather than about a run over it: what is in a site's corpus,
whether all of it still loads, and how it is divided. Validation is the first half and it is total —
every case file goes through that site's own reader (the coder's bucket-walking provider and coder
schema; the generic flat provider for everybody else, chosen by a lookup in
`src/noctis/eval/bootstrap.py`'s `SITE_CORPORA`, so a second corpus reports down the same code
path) — and a file that no longer parses, a case labelled with an axis nobody declares or a bucket
directory nobody named is a refusal naming the file and the defect, with nothing counted.

What prints is the population: the case count and the corpus digest, then the tuning/holdout counts
and shares overall, per bucket (where the site has buckets — one that does not says so rather than
printing an empty table) and per difficulty-axis level. A bucket or level the site *declares* but no
case represents is named as exactly that, because an absence dropped from a table is an absence
nobody can act on; a share over an empty group prints `n/a` and never `0`; and a case whose file
declares no `split:` is counted **unstamped** — it was dealt in memory at load time and can still
move when the corpus grows, which is the curator's cue to freeze it. Reading a corpus writes
nothing: no split is stamped, no file repaired, no index touched.
