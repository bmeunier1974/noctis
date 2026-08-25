# The run record

Every `noctis run` and every `noctis research` **mints a run** — an addressable entity that
outlives the process that started it — and gives it one tree under the workspace:

```text
workspace/runs/
  index.json                 # the DERIVED listing roll-up over every record
  <run_id>/
    run.json                 # THE record — everything below is in this one file
    run.lock                 # liveness, while an engine holds the run
    state/  strategies/  memory/  reports/  qa/
```

`run.json` is the artifact this page is the contract for. It has **no sidecars**: one `fetch()` of
one URL returns everything a run page needs, so a website needs no server-side logic, and `jq`
answers any question about a run without opening a second file. `index.json` beside it serves the
*listing* page in one more fetch, and is derived from the records — delete it whenever you like.

Where the shape is decided, in the code, in one place each:

| Module | What it owns |
|---|---|
| `src/noctis/reporting/schema.py` | The version, the required sections, every section's key tuple, the caps, and a pure `validate()` |
| `src/noctis/reporting/run_record.py` | `build(artifacts) -> dict` — the **pure** builder; every derived number is computed here |
| `src/noctis/reporting/assumptions.py` | The `assumptions` block (pure) |
| `src/noctis/reporting/metrics.py` | The `performance` block's arithmetic (pure) |
| `src/noctis/reporting/run_tree/` | The only package that touches the run tree — five modules over one narrow read plus the store that holds them, layered `record ← {address, index, lock, evidence} ← store`: `record` (the tree's names, `read_record`, one atomic `write`), `address` (the four address forms), `index` (the derived listing roll-up), `lock` (the liveness protocol), `evidence` (the six collectors, and every heavy import in the package), `store` (the lifecycle verbs, `read_artifacts` and `RunStore`) |
| `src/noctis/config/rehydrate.py` | Freezing and rehydration — which settings a resume restores |
| `src/noctis/observability/engine_id.py` | Engine identity, the component map, the comparable key |

---

## The load-bearing rules

These are the promises a consumer may build on. Each is enforced structurally, not by convention.

1. **Evidence, never a gate.** Nothing in the record is read by a promotion gate, a research
   budget or the exhaustion floor. A test imports the whole promotion path in a fresh subprocess
   and asserts that nothing under `noctis.reporting` is reachable from it. The record describes
   the judging; it never participates in it (AGENTS.md rule 2).
2. **The safety gate is never rehydrated.** `mode` and `allow_live` are never written to a record
   and never restored from one — `validate()` refuses either as a **key** in the three places one
   could arrive (`assumptions`, `assumptions.live_gate`, `inputs.settings.resolved`). The record
   carries the gate's **verdict** (`inputs.execution_mode`, `assumptions.live_gate`), which is a
   measurement, not a second source; the pair itself re-resolves from `config.yaml` + `ALLOW_LIVE`
   at every process start (AGENTS.md rule 1, [safety.md](safety.md)). `inputs.settings.refused_keys`
   *names* the two — naming is the point of that list — and never values them.
3. **The writer is never fatal — except lock contention, which is a hard refusal.** The first
   internal failure logs exactly one warning, latches the store off, and every later call is a
   no-op; a reporting artifact must never take down a multi-week run. A **live lock** is different
   in kind: two engines writing one run's record, champion board and paper account is corruption,
   so it is refused outright.
4. **Cumulative fields are derived, never incremented.** Every total is recomputed at each write
   from the durable artifacts plus the append-only `segments[]`. Three one-hour segments therefore
   total exactly what one three-hour segment does, and a crash mid-write cannot double-count.
5. **Numbers are never pooled across `comparable_key`.** Two runs' champion and scorecard numbers
   may be compared only when `engine.comparable_key` matches — and a run flagged `mixed_engine`
   ran more than one engine, so its key describes only its latest.
6. **Retention never breaks resumability.** Only a `completed` run may be pruned, and `completed`
   is exactly the status that refuses a resume: prunable and resumable are two sides of one
   constant, so no window exists in which a run is both.
7. **Output is workspace-only.** Records live under the gitignored `workspace/`; no operator's
   run, champion or rejection ever reaches git.
8. **The schema is additive, and additive means declared.** New fields may appear at any time and
   an existing field never changes meaning or type; a *reader* ignores keys it does not know, so a
   newer record is still readable. `validate()` is the stricter question — does this record conform
   to the schema this engine ships? — and it refuses a key no section declared.

Two more rules are about honesty rather than safety, and a consumer must respect both:

- **A truncated record never passes for a complete one.** Every cap that bites writes a note with
  kept/total counts into `run.truncated`, and `run.complete` is `false` whenever a segment is open,
  a process was killed, or the writer latched off.
- **An absent value is an explicit `null`, never a dropped key.** That is what lets a consumer tell
  "not applicable" from "this schema version did not have it".
- **An undeclared key is refused, never quietly published.** A section's keys are a closed
  vocabulary, so `validate()` names a key no section declared (`run.cumulative_trails: undeclared
  key`) exactly as it names a missing one. The failure this catches is an emitter typo: a field
  spelled one way in the writer and another in every reader passes every presence check and is
  indexed by nobody. Growing a section stays one edit — declare the key in `schema.py`.

---

## Reading one

```bash
python -m noctis runs                          # the board: id, label, status, segments, key
python -m noctis run-record latest             # print one whole record on stdout
python -m noctis run-record @nightly | jq .run # …and pipe it anywhere
python -m noctis run-record latest --validate  # schema-check it instead of printing it
python -m noctis engine                        # this checkout's identity and comparable key
```

`run-record` takes the same four address forms as `--resume` (an id, `latest`, a `run.json` path,
`@label`), resolved by the same code — see [cli.md → the run board](cli.md#the-run-board--runs-and-run-record).

---

## Conventions

Four, and they are part of the contract rather than house style. `validate()` enforces all four
structurally, over the whole document, so a section added tomorrow inherits them.

| Convention | Rule |
|---|---|
| **Self-describing** | `kind` is always `"noctis.run"`, `schema_version` is always an integer — so a consumer can tell a record from any other JSON before parsing it |
| **Units in the name** | A dimensioned number spells its unit exactly one way: `_usd`, `_pct`, `_bps`, `_s`, `_bytes`, `_bars`, `_hours`. `_seconds`, `_ms`, `_bp`, `_kb` are schema violations |
| **UTC with a `Z`** | Every key named `*_utc`, or named `t` / `at` / `ts`, is ISO-8601 UTC ending in `Z` (`2026-07-27T14:22:33.418Z`). Session and trade *dates* are calendar days (`as_of`, `date`, `peak_date`), which is why they are not `_utc` |
| **Explicit `null`** | Every key of every section is present whatever its value. "Not applicable" is `null`; a dropped key is a violation |

Two subtrees are exempt from the unit rule, by name, because they are **foreign documents quoted
verbatim**: `inputs.settings.resolved` (the operator's own configuration as the run froze it —
`research_time_budget_minutes` is a config key, not a record field) and `inputs.mandate` (the
mandate's front-matter as written). Renaming a key inside either would make the record disagree
with the file it claims to quote.

Every dollar figure additionally names itself an **estimate** (`llm_usd_estimate`,
`usd_per_champion_estimate`). These prices come from a versioned list-price table, not from an
invoice, and a key that said `usd` alone would read as a receipt. `validate()` enforces that too.

---

## The versioning promise

`SCHEMA_VERSION` is **1**.

- **Additive-only.** New fields may be added at any time. An existing field never changes meaning
  or type, and the **read path never rejects a document** — `read_artifacts()` and `upgrade()`
  carry a record forward whatever it carries, and a consumer ignores keys it does not know. That is
  what lets a record written tonight still be read by the Noctis that resumes the run in a month.
- **Additive means declared.** `validate()` is a *conformance* check against the schema this engine
  ships, not the read path, and its keys walker runs both ways: a section carries the keys the
  schema declares for it and no others, so an **undeclared key is refused** rather than published
  under a spelling no reader indexes. The record's *sections* stay open — `REQUIRED_SECTIONS` is a
  floor, never a ceiling — which is where a later story's whole new section lands; a new key inside
  an existing section is one line in `schema.py`.
- **A breaking change bumps the version**, and `schema.upgrade()` rewrites the record **in place**
  on the next open: it walks a record up one version at a time through a reviewable registry
  (`schema.UPGRADES` — empty at version 1, because there is no earlier version to come from),
  restamps it, and the run files an `info` event saying what happened to it. A version is never
  silently repurposed and a key is never quietly reinterpreted.
- **A record from the future is never touched.** Additive-only means a newer document is readable
  by ignoring what this engine does not know; rewriting its version *down* would destroy exactly
  the information a later reader needs.
- The upgrade runs at `read_artifacts()`, so the upgraded document lands on disk with the next
  ordinary write. A record already at this version is returned untouched and produces no event — a
  run that is simply resumed does not accumulate a note per night.

`index.json` carries the same `schema_version`, read off the same constant.

---

## The record, section by section

### Top level

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | `1` today. See above |
| `kind` | string | Always `"noctis.run"` |
| `run` | object | Identity, lifecycle, the cumulative totals |
| `segments[]` | list | One entry per process invocation. **Never capped** — it is the run's spine |
| `environment_latest` | object \| null | The newest machine any segment recorded. Derived |
| `engine` | object | What judged this run |
| `inputs` | object \| null | The frozen configuration. `null` for a run that never froze one |
| `strategies[]` | list | Every candidate considered. Empty, never `null` |
| `spend` | object \| null | What the run cost. `null` for a run with no research evidence at all |
| `sessions[]` | list | Every session the paper account closed. Empty, never `null`; never capped |
| `performance` | object \| null | The realised paper-account record. `null` in two cases (below) |
| `assumptions` | object | The arena. Never `null` |
| `events[]`, `errors[]` | list | The run's own timeline. Capped at `EVENT_CAP` each |

`REQUIRED_SECTIONS` in `schema.py` is the authority; additive-only means that tuple may grow,
never shrink.

### `run`

| Field | Type | `null` when |
|---|---|---|
| `run_id` | string | never — `20260727T142233Z-a1b2c3`: UTC compact stamp + 6 hex. **Minted, never derived** from the config, so two byte-identical configs are two runs |
| `label` | string \| null | the run was never given a `--label`. Convenience only: the id is the identity |
| `status` | string | never — one of `running` / `stopped` / `interrupted` / `completed` (below) |
| `created_utc` | string \| null | the record was built without one (an adopted history that no process ever opened) |
| `last_active_utc` | string \| null | as above. This — never a filesystem mtime — is what `latest` is resolved by |
| `completed_utc` | string \| null | the run is not `completed`. Set by `--finish`, or **derived** at the segment close that crossed a run-level cap |
| `run_limit_hours` | number \| null | the run is uncapped, or froze no configuration. Read back from `inputs.settings.resolved`, never stored twice |
| `traded` | bool | never. `true` if any segment counted a trade **or** any session journaled a fill |
| `cumulative_runtime_s` | number | never — `0.0` for a run with no closed segment. Summed from `segments[].duration_s` |
| `cumulative_research_s` | number \| null | **no** segment measured any phase timing at all. A run that measured and never traded reports an honest `0.0` |
| `cumulative_trading_s` | number \| null | as above |
| `cumulative_trials` | int \| null | the run journaled nothing. Counted off `state/experiments/*.jsonl` — the very lines the exhaustion gate counts |
| `state_pruned` | bool | never. `true` once retention removed `state/`, `strategies/` and `reports/` |
| `complete` | bool | never. `false` while a segment is open, after a kill, or after the writer latched off |
| `truncated` | object | never — `{}` when nothing was capped. Otherwise `{"events"\|"errors"\|"strategies"\|"trades": {"kept": N, "total": M}}` |

### `segments[]`

One segment = one process invocation. The stop-each-morning / resume-each-night pattern makes one
segment a night. Indexed from 0 with no gaps; append-only; never capped.

| Field | Type | `null` when |
|---|---|---|
| `index` | int | never. Equals the entry's position |
| `started_utc` | string | never |
| `stopped_utc` | string \| null | the segment is still open, or was killed |
| `duration_s` | number \| null | as above — an unclosed segment has no honest duration, and contributes none |
| `stopped_reason` | string \| null | as above. Seen in practice: `time_limit`, `run_limit`, `stop_requested`, `max_cycles`, `guard`, `startup`, `no_data`, plus a research session's own (`agent_done`, `time_budget`, `max_iterations`, `author_budget_exhausted`, `api_error`, `no_client`, `no_session`, `prose_stall`, …). Free-form by design — the enumerated field beside it is `status` |
| `status` | string | never — `running` / `stopped` / `interrupted` |
| `argv[]` | list of string | never. The invocation, minus the program name |
| `command` | string | never — `run` or `research`. **"Research-only" is derived from this**, not marked by a second flag: `research` is a verb that cannot trade |
| `resumed` | bool | never. `true` when the run already had segments |
| `counters` | object | never — `{}` when none. `cycles` / `research_iterations` / `trades` from the loop; `sessions` / `research_iterations` / `research_promotions` from a research segment |
| `phase_seconds` | object \| null | this segment measured none. `{"RESEARCH": …, "TRADING": …, "CLOSE": …}` — seconds spent **working**; waiting out a weekend belongs to `duration_s` and to no phase |
| `environment` | object \| null | this segment measured none (see below) |
| `engine_version` | int \| null | this segment recorded no engine |
| `engine_fingerprint` | object \| null | as above. The engine that actually produced *this* segment, which is not necessarily the run's |

#### `segments[].environment`

Per segment, **never per run**: a run may migrate machines mid-experiment, and research throughput
is CPU-bound, so one night's trials-per-hour is only comparable to another's when the hardware
behind each is on the record.

| Field | Type | Notes |
|---|---|---|
| `hostname_hash` | string \| null | `sha256(hostname)[:12]` — two segments on one machine are provably the same host, without publishing a name |
| `os` | object \| null | `system`, `release`, `arch` |
| `container` | bool \| null | whether the process ran in a container |
| `cpu` | object \| null | `model`, `cores_physical`, `cores_logical`, `freq_max_mhz` |
| `memory_total_bytes` | int \| null | |
| `disk_free_bytes` | int \| null | |
| `python` | string \| null | |
| `noctis_version` | string \| null | |
| `git` | object \| null | `commit`, `branch`, `dirty`, `describe`; `null` outside a checkout |
| `lockfile_digest` | string \| null | of `uv.lock`; `null` outside a checkout |
| `extras_present` | object \| null | `{extra name: version or null}` over `llm` / `data` / `research` / `engine` / `hardware` |
| `degraded_seams[]` | list \| null | the capabilities that were missing — the remedy is `uv sync --extra <name>` |

**Degradation is the ordinary case**, and it is explicit: `psutil` is an optional extra
(`hardware`), never a core dependency, so a bare core install produces a schema-valid block whose
facts are `null` and whose `degraded_seams` names what was missing. A block that dropped the keys
it could not answer would be indistinguishable from an older schema version.

### `environment_latest`

The newest environment any segment recorded, or `null` when none did. **Derived**, so a consumer
rendering "the machine this run is on" reads one key that cannot disagree with `segments[]`. A
segment that measured nothing is skipped rather than blanking the answer.

### `engine` — identity, components, comparable key

Two runs' numbers are comparable only if the same engine produced them. A promotion threshold
moved, a prompt reworded, a shipped mandate profile edited, a seed strategy changed: each shifts
results without a single config key differing. So the engine carries an identity, and it is two
things at once — a **declared** version and a **computed** per-component fingerprint.

| Field | Type | Notes |
|---|---|---|
| `engine_version` | int | The declared behavioural contract version — **3** today. A plain incrementing integer, deliberately decoupled from the package version: a release that changes no behaviour must not fragment comparison buckets, and a one-line gate change that ships in no release must |
| `engine_epoch` | int | `1` on every run; moved only by a deliberate `--allow-engine-upgrade` |
| `noctis_version` | string | The package version the run was created under |
| `fingerprint` | object | `{component: digest or null}` over the eight components below |
| `comparable_key` | string | `"<engine_version>\|<gates>\|<backtest>\|<election_metric>"` |
| `mixed_engine` | bool | **Derived** from two independent facts that mean the same thing: a segment whose digests differ from the run's, or an accepted engine change on the record |
| `engine_changes[]` | list | Empty (never absent) on the runs that never upgraded |

The record's `engine` block is what the run was **frozen at creation** under — the side a resume is
compared against — while each `segments[].engine_fingerprint` is what actually produced that
segment.

**The component map** (`engine_id.COMPONENT_PATHS`) is an explicit allowlist of committed files,
never a directory walk, so an operator's gitignored mandate can never leak in and make a
fingerprint machine-local. Content is hashed raw and LF-normalized, deliberately *not* stripped of
comments: prompt text is indistinguishable from a comment to any safe automated rule, so a
docstring edit moving a component is the accepted cost of never silently pooling incomparable runs.
A missing input yields a `null` component with a note, never a crash.

| Component | Tier | Decides | Covers |
|---|---|---|---|
| `gates` | **arbiter** | what passes | `champions/promotion.py`, `backtest/scorecard.py`, `backtest/splits.py` |
| `backtest` | **arbiter** | what a number *means* | `backtest/pipeline.py`, `validate.py`, `candidate.py`, `prefilter.py`, `broker/seam.py`, `broker/paper.py` |
| `research` | searcher | how candidates are found | `research/agent.py`, `driver.py`, `tools.py`, `episode.py`, `sweep.py` |
| `prompts` | searcher | what the model is told | `research/prompt.py`, `briefings.py`, `contract_sheet.py`, `digests.py`, `ideation.py` |
| `profiles` | searcher | the shipped steering personalities | the committed `mandate/` scaffold (never the operator's own files) |
| `seeds` | searcher | the read-only library every run starts from | `strategies/TEMPLATE.py` + the three worked examples |
| `memory_seed` | searcher | the starting condition of every run's memory | `MEMORY.seed.md` |
| `schema` | searcher | what is recorded | `reporting/schema.py` |

**The arbiter/searcher split is declared exactly once** (`ARBITER_COMPONENTS`) and read by both
enforcers, so the CI ratchet (a change may not land) and the resume policy (a run may not continue)
can never answer it differently. `gates` and `backtest` decide what passes and what a number means,
so they bind comparability; everything else describes how candidates are *found*. Two runs with
different `prompts` digests but identical `gates` and `backtest` digests still have comparable
scorecards.

**`comparable_key` = `(engine_version, gates_digest, backtest_digest, election_metric)`** — the
tuple two runs must match on before their champion and scorecard numbers may be pooled, ranked or
plotted together. The two arbiter digests carry the strict guarantee rather than the declared
version, because a digest cannot be forgotten in review. The election metric rides along for the
reason promotion already treats a differently scored champion as *stale*: numbers under different
metrics are in different units and were never comparable. A `null` digest renders as the literal
`null` inside the key (`2|null|3ba3e0bf1c97134f|sharpe`).

Each `engine_changes[]` entry carries `at`, `segment`, `from_epoch`, `to_epoch`,
`from_engine_version`, `to_engine_version`, `components[]` (each `{component, tier, from, to}`) and
`accepted_by` (`"--allow-engine-upgrade"`). A run whose engine changed mid-flight says so *and says
where*. See [cli.md → Engine change](cli.md#engine-change-resuming-after-the-code-moved).

### `inputs` — the frozen configuration

A run outlives its process, so it carries its own configuration: **frozen at creation**, restored
on every later segment, so an edit made on Tuesday cannot retroactively change what Monday's
results meant. `null` for a run that never froze one — an adopted history (`noctis migrate`), which
resumes against the current files instead.

| Field | Type | Notes |
|---|---|---|
| `config_epoch` | int | `1` until a deliberate `--rebase-config` |
| `config_changes[]` | list | Empty (never absent). Each entry: `at`, `segment`, `from_epoch`, `to_epoch`, `digest_before`, `digest_after`, `settings[]` (`{path, from, to}`), `mandate` (`{from, to}` or `null`) |
| `frozen_at_utc` | string | When this block was frozen (or last rebased) |
| `execution_mode` | string \| null | The **safety gate's verdict** for this run (`"paper"`). Never the `mode` setting. `null` for a run that froze none |
| `mandate` | object \| null | `source`, `summary`, `text`, `text_sha256`, `symbols[]`, `config_overrides`, `overrides_applied[]`, `references[]`. Frozen as **resolved text**, not as a selector |
| `models` | object | `research`, `coder`, `coder_fallback`, `ideation`, `research_loop`, `context_window`, `cost_profile` — a derived, resolved view. `null` per key when unset |
| `data` | object | `provider`, `dataset`, `lake_dir` |
| `settings.digest` | string | `sha256[:12]` over the frozen tier only — the "these two runs mean the same thing" label |
| `settings.resolved` | object | The settings dump **minus** the secrets, the refused pair and the run's own paths. Quoted verbatim (exempt from the unit rule) |
| `settings.frozen_keys[]` | list | The dotted paths a resume restores from the record |
| `settings.live_keys[]` | list | The dotted paths a resume takes from the current process |
| `settings.refused_keys[]` | list | `["allow_live", "mode"]` — named, never valued |

#### The three freezing tiers

Every leaf setting belongs to exactly one tier, classified in `config/rehydrate.py`. Today:
**75 frozen, 18 live, 2 refused**. The record publishes the three lists, so a consumer never has to
guess which is which.

| Tier | Count | What | On a resume |
|---|---|---|---|
| **Frozen** | 75 | Everything that decides what the accumulated results *mean*: `research.*`, `promotion.*`, `backtest.*`, `trading.*`, `risk.*`, `ideation.*`, `session.*`, `universe`, `champion_count`, `data.provider`/`dataset`/`history_days`/`auto_backfill`, `research_time_budget_minutes`, `run_limit_hours`, `embed_all_sources`, `live_feed.*` — **plus the whole mandate**, as resolved text | from the record |
| **Live** | 17 | The three API keys; every path knob (`workspace_dir`, `runs_dir`, `run_dir`, `state_dir`, `reports_dir`, `memory_path`, `qa_dir`, `strategies_dir`, `mandate_dir`, `data.lake_dir`); the per-process budgets `time_limit_hours`, `data.budget_usd`, `qa.keep_last_runs`, `observability.heartbeat_polls` | from the current process |
| **Refused** | 2 | `mode`, `allow_live` | from **neither** — never recorded, never restored |

Three consequences worth stating plainly:

- **Live is not a gap, it is the design.** Secrets are excluded from the record entirely, so a
  record is shareable and resuming it needs *your own* keys from `.env`. Paths are live so a run
  can resume on a machine whose absolute paths differ. The per-process budgets bound one *night*,
  not one experiment.
- **The two wall-clock ceilings sit in different tiers on purpose.** `time_limit_hours` is live (it
  bounds tonight); `run_limit_hours` is frozen (it bounds the experiment — a cap that could be
  raised each morning would bound nothing).
- **Two of the three tiers are derived, not re-listed.** The refused pair is exactly the mandate
  overlay's live-money refusals, the path knobs are exactly its state/IO refusals, the secrets are
  the one set `Settings` names — and frozen is the *complement*, so a knob added tomorrow freezes
  by default. That is the safe direction: it keeps meaning attached to results.

The whole contract, with the drift and rebase flags:
[configuration.md → Config freezing](configuration.md#config-freezing--what-a-resumed-run-reads).

### `strategies[]` — the funnel, not the trophy shelf

One entry per candidate the run considered — champions, rejections and the drafts that never
reached a verdict. *"47 of 66 candidates died at the symbol-holdout gate"* is the sentence that
makes a results page credible where an equity curve does not, and it is computable only when the
failures are on the record in the same shape as the promotions. Champions are ordered first, then
alphabetically, so two writes of one run order identically and the cap can never drop a champion.

| Field | Type | `null` when |
|---|---|---|
| `name` | string | never |
| `outcome` | string | never — `promoted` / `rejected` / `undecided`. `undecided` is a real and common state: authored, perhaps tuned, never carried to a verdict |
| `tier` | string \| null | the run no longer has the file (pruned, or a swept draft). Otherwise `champions` or `__tmp` — the committed `strategies/` seeds are read-only input and belong to no run |
| `decided_utc` | string \| null | no verdict was journaled for this name |
| `trials` | int \| null | this candidate has no experiment journal |
| `gates[]` | list | never — `[]` for an undecided candidate |
| `rationale` | string \| null | no verdict, or the verdict carried no prose |
| `source_path` | string \| null | the run has no file for this name. **Relative to the run directory**, so it stays portable |
| `source_sha256` | string \| null | as above |
| `source` | string \| null | the candidate is not a champion — by tier *or* by verdict — and the run was not created with `--embed-all-sources`; or the run has no file for it at all. An embedded source always carries its hash |

#### `strategies[].gates[]`

The structured evidence `champions/promotion.py` produced, in gate order:

| Field | Type | Notes |
|---|---|---|
| `gate` | string | one of, and appended in this order: `validated`, `activity_floor`, `overfit_gap`, `reverse_gap`, `magnitude_cap`, `forward_holdout`, `symbol_holdout`, `symbol_consistency`, `family_slot`, then **either** `minimum_bar` (a free or stale slot was open) **or** `beat_weakest` (a full board) |
| `passed` | bool | |
| `observed` | number \| null | `null` where the gate had nothing numeric to compare (`validated`), or the scorecard carried no such metric |
| `threshold` | number \| null | as above |
| `note` | string \| null | why a gate could not bite (`inert: switched off by a max_test_metric of 0`), or which slot a promotion took |

**A rejection short-circuits, so the list is the gates *reached*, plus the one that failed.** An
absent gate means "not reached", never "passed" — a consumer counting deaths per gate must read it
that way, and the funnel's death counts come off the one entry whose `passed` is `false`.

**The size policy, and its cost, stated rather than hidden.** A champion's file is embedded in
full; everyone else is a path plus a content hash. That is what holds a fortnight's record to
hundreds of kilobytes instead of megabytes — and it means a rejected candidate's *code* is readable
only while the run's own `strategies/__tmp/` tier survives. `run.state_pruned` says when it no
longer does. `noctis run --embed-all-sources` is the deliberate alternative for an experiment worth
archiving whole, and is frozen at creation.

### `spend`

What the run cost and what that bought. `null` for a run with **no research evidence at all** — no
journaled ledger *and* no journaled trials, which is the shape of a run with no LLM key: it must
report an unknown bill rather than a free one. Once either kind of evidence exists the block is
present, with an explicit `null` wherever a number is genuinely unknown.

| Field | Type | `null` when |
|---|---|---|
| `tokens` | object \| null | the run journaled no session ledger. Carries the four billed fields plus `total_tokens` |
| `by_model` | object \| null | as above. `{model: bucket}` |
| `by_stage` | object \| null | as above. `{stage: bucket}` — the stages the driver actually journaled, spelled as it spells them (`formulate`, `match`, `discover`, `author`, `optimize`, `decide`); the record invents no taxonomy of its own |
| `by_segment` | list \| null | as above. One `{index, …bucket}` per segment, joined on the entry's stamp |
| `llm_usd_estimate` | number \| null | **any** entry could not be priced — a partial sum presented as a total understates the bill while looking complete |
| `pricing_table_version` | string \| null | no ledger. `2026-07.1`, or `2026-07.1+custom.<digest>` when an operator overrode prices. A `<month>[.<revision>]` label: a revision is the same month's prices with corrected coverage, and a record keeps whichever label was in force when it was written |
| `efficiency` | object | never |

A **bucket** is the four fields (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
`cache_read_input_tokens`) plus `total_tokens` and `usd_estimate`. The total is always known — the
ledger journals it per episode — while the four fields are `null` for the whole bucket if *any*
contributing entry never journaled its split (a ledger written before the split existed): a sum
over only the entries that did would silently report less spend than happened.

`efficiency` carries `usd_per_champion_estimate`, `usd_per_trial_estimate`, `trials_per_hour` and
`research_hours_per_champion`. Every one is a ratio, and every ratio is `null` when its denominator
is zero or unknown — a run that has crowned no champion has no cost per champion, and that is the
normal state of a young run rather than an error.

An entry whose stamp falls in no segment window is attributed to **no** segment but still counted
in the run's totals: a number may lose its attribution, never its existence.

### `sessions[]`

One entry per session the paper account closed — the realised evidence `performance` is derived
from. Uncapped, like the segments: they are the run's realised spine, and losing one would make
every derived total a lie. The *trades* inside them are bounded across the whole run by
`TRADE_CAP`, and a bounded log writes `run.truncated.trades`.

| Field | Type | Notes |
|---|---|---|
| `as_of` | string | The session date (`YYYY-MM-DD`), not a timestamp |
| `equity` | number | The **account's** mark at that close — the point on the curve |
| `start_equity`, `end_equity` | number | The session's own bounds, which differ from `equity` whenever positions were carried in |
| `realized_pnl` | number | |
| `orders_submitted` | int | |
| `positions_end` | object | `{symbol: quantity}` at the close |
| `trades[]` | list | The session's fills |

Each fill: `ts`, `symbol`, `side`, `quantity`, `price`, `fees_usd`, `slippage_bps`, `champion`,
`rationale` — every key present, `null` where the fill did not carry the value. `champion` is what
makes the trade log readable per champion instead of as one blended blob. (The per-day report under
`<run>/reports/` uses the opposite convention — an absent enrichment field is *omitted* there — so
an operator's existing reports stay byte-identical.)

### `performance`

The paper account's realised record. **`null` in two cases**, which are the same case seen from two
sides:

1. **The run never traded** (`run.traded` is `false`) — a research-only run is a first-class shape,
   and a website must render "researching" rather than a flat 0% equity curve it was handed as if it
   were a result. The schema *enforces* this pairing: `traded: false` with a populated
   `performance` is a violation.
2. **The run traded but its account journaled no daily mark** — what an adopted history looks like.
   A block of nulls would read as a measurement that came out empty rather than one nobody took.

The block names itself `source: "paper_account"`, checked rather than assumed: **backtest numbers
are never blended in**. A consumer that renders both can tell them apart structurally.

| Field | Type | Notes |
|---|---|---|
| `source` | string | Always `"paper_account"` |
| `account` | object | `opened`, `start_equity`, `end_equity`, `cumulative_pnl_usd`, `sessions` |
| `equity_curve[]` | list | `{date, equity}` per session, **re-derived from the ledger at every write** — never carried in memory, which is why three short nights publish exactly the curve one long night would |
| `returns` | object | `total_return_pct`, `cagr_pct`, `annual_volatility_pct`, `best_day_pct`, `worst_day_pct` |
| `risk_adjusted` | object | see below |
| `drawdown` | object | `max_drawdown_pct`, `max_drawdown_days`, `peak_date`, `trough_date`, `recovered`, `recovery_factor` |
| `trades` | object | `count`, `win_rate`, `loss_rate`, `profit_factor`, `expectancy_usd`, `avg_win_usd`, `avg_loss_usd`, `payoff_ratio`, `gross_profit_usd`, `gross_loss_usd`, `total_fees_usd`, `by_champion`, `open_at_close`, `exposure`, `turnover` |
| `benchmark` | object | see below |
| `monthly_returns_pct` | object | `{"YYYY-MM": pct}` |

`risk_adjusted` carries `sharpe`, `sortino`, `calmar`, `psr`, `deflated_sharpe`, `n_trials_used`,
`deflation_basis`, `skew`, `excess_kurtosis`, `annualization_basis` — each `null` when its inputs do
not exist yet.

- **The Deflated Sharpe Ratio publishes the count that deflated it.** `n_trials_used` is the run's
  own `cumulative_trials` — the very lines the exhaustion gate reads off the experiment journals —
  and the schema refuses a `deflated_sharpe` without it: a deflation nobody can audit is a number
  nobody should trust. `deflation_basis` names the variance assumption behind it.
- **Trade statistics are over *closed* round trips** (flat → position → flat, per symbol). A
  position still open at the last mark is not a trade yet — its P&L is unrealised and already sits
  in the equity curve — so it is counted in `open_at_close` instead.
- `trades.by_champion` is `{champion: {count, pnl_usd, fees_usd, win_rate}}`; fills that carried no
  attribution group under `unattributed`.

`benchmark` is `equal_weight_universe_bh` — named so nobody mistakes it for an index. It is
equal-weight buy-and-hold over the symbols the run **actually traded**, priced from bars *already
in the shared lake* over the run's own session window: no vendor call and no new spend. Fields:
`name`, `method`, `symbols[]`, `total_return_pct`, `sharpe`, `alpha_pct`, `beta`,
`information_ratio`, `tracking_error_pct`, `correlation`, `note`. A run whose symbols the lake does
not hold is **not benchmarked** rather than benchmarked wrongly — the numbers are `null` and `note`
says why. Only statistics reach the record; the benchmark's price series never does.

### `assumptions` — the arena, as data

Mandatory on every record and never `null`. A results page that does not state its assumptions is
not taken seriously, and one that states them in prose cannot be diffed between two runs. The
engine-fixed half is true of any run; the configured half is an explicit `null` key by key on a run
that froze no configuration.

| Field | Type | Notes |
|---|---|---|
| `paper_only` | bool \| null | **Measured**, never asserted: read off the gate's own frozen verdict. `null` = nobody measured, which is not "paper" |
| `live_gate` | object | `execution_mode`, `real_orders_reachable`, `re_resolved_each_segment` (always `true`), `note`. The two settings behind the verdict are never recorded |
| `fill_model` | string | `next_bar_open` — a decision on bar *t* is filled at bar *t+1*'s open |
| `lookahead` | string | The no-lookahead statement, in one sentence |
| `fee_bps`, `slippage_bps` | number \| null | What this run was charged. `null` when it froze no configuration |
| `round_trip_cost_bps` | number \| null | `2 × (fee + slippage)`. `null` if either side is unknown |
| `costs_charged` | string | `per side — a round trip pays both legs`. Stated because the commonest way to misread a cost assumption is to halve it |
| `walk_forward` | object | `sizing` (`auto`), `min_train_bars` 40, `max_train_bars` 120, `min_test_bars` 20, `max_test_bars` 40, `step_bars` (`null` = one test window, so windows never overlap), `embargo_bars` 0, `test_after_train` `true` |
| `forward_holdout` | object | `min_bars`, `max_bars`, `reserved`, `note` |
| `symbol_holdout` | object | `size`, `fit_set_size`, and `symbols` — **always `null`**: which names were held out is sampled per research session, not fixed for the run |
| `min_trials` | int \| null | The exhaustion gate's floor: the distinct param sets a verdict may not be reached without |
| `promotion_thresholds` | object \| null | The **whole** `promotion` subtree verbatim, plus `champion_count` — so a threshold added tomorrow appears here with no edit and can never quietly go unpublished |
| `benchmark` | object | `name`, `method`, `rebalancing` (`none` — weights are set at the first priced session mark and drift thereafter) |
| `state_scope` | string | `run` — champions, paper account and journals live inside the run's own tree |

The walk-forward geometry is stated as a **rule with bounds** rather than one number, because it is
sized per candidate from the panel's shortest series. Three values here are engine constants rather
than settings — the fill model, the sizing bounds and the rebalancing convention — and each is held
to the code that implements it by a test.

### `events[]` and `errors[]`

One flat timeline for the whole run — deliberately small: the record is evidence a website renders,
not a log. Rich per-event detail stays in the `--debug` QA tree.

| Field | Type | `null` when |
|---|---|---|
| `t` | string \| null | the observation was made while the run was being *opened*, before this process's segment existed (a schema upgrade, an unreadable prior record) |
| `segment` | int \| null | as above |
| `kind` | string | never — `info` / `warn` / `error` |
| `text` | string | never |

`errors[]` is the same shape; `note(..., kind="error")` files there. Both are capped at
`EVENT_CAP`; the **earliest** entries are kept, because they are the ones explaining how a run
reached its state.

---

## Caps, truncation and the size budget

| Constant | Value | Applies to |
|---|---|---|
| `TRADE_CAP` | 5 000 | fills, **across the whole run** (not per session — a fortnight of quiet days followed by one frantic one should publish the frantic one) |
| `EVENT_CAP` | 2 000 | `events[]` and `errors[]`, each |
| `STRATEGY_CAP` | 500 | `strategies[]`. Champions are ordered first, so a cap never drops one |
| `RECORD_SIZE_BUDGET_BYTES` | 384 KiB | the whole record — **measured, not enforced** |

`segments[]` and `sessions[]` are deliberately **uncapped**: they are the run's spine, and losing
one would make every derived total a lie.

**Silent truncation is forbidden.** Every cap that bites writes `run.truncated.<list> = {kept,
total}`, so a reader always knows what is not there.

The size budget is a *measurement a test defends*, not a ceiling that truncates: a synthetic
two-week run (14 segments, 66 candidates, 3 champions embedded in full, ~3 000 trials, 14 traded
sessions at 30 fills each) is held under it, so a change that quietly makes the record ten times
heavier is a red test rather than a slow website. It has moved twice for exactly that reason —
per-candidate gate evidence took the worked fortnight to ~172 KB (budget 256 KiB), and the realised
equity curve and trade log took it to ~286 KB (budget 384 KiB). The caps above are a different
instrument: they bound the *pathological* run.

---

## The resume model

### Mint, or resume

A bare `noctis run` (or `noctis research`) **mints** a run: a fresh id, a fresh tree, a fresh
record. Identity is minted, never derived from the configuration — two byte-identical configs are
two runs. `--resume <address>` appends a segment to an existing run instead, and the same record
keeps accumulating: the **run**, not the process, is the unit progress is tracked on.

Four address forms resolve in one place (`run_tree.resolve_run_dir`, in `run_tree/address.py` —
which reads records and nothing else, so naming a run takes no lock and runs no collector; shared by
`run`, `research`, `run-record` and `run-prune`) in a fixed order — a path, `@label`, the reserved
word `latest`, a run id — and **a bare address is always the id**. Details:
[cli.md → the four address forms](cli.md#the-four-address-forms-and-how-they-are-told-apart).

### The four statuses

`run.status` is **derived** from the segments rather than carried beside them, so the two can never
disagree.

| Status | Means | Resume? | Prune? |
|---|---|---|---|
| `running` | the last segment is open | **yes** — it is also the shape a *crash* leaves behind, and telling a crash from a live engine is the lock's job, not a status field's | no |
| `stopped` | the last segment closed cleanly, or the run has no segments | yes | no |
| `interrupted` | the last segment was killed mid-flight | yes | no |
| `completed` | sealed by `--finish`, or by crossing `run_limit_hours` | **no — terminal** | **yes** |

**`completed` is the only status that refuses a resume, and the only one that may be pruned.**
`TERMINAL_STATUSES` and `PRUNABLE_STATUSES` are the *same tuple*, not two that happen to agree
today, which is what makes "a pruned run that later resumed" unreachable rather than merely
unlikely.

### `interrupted` is observed, never guessed

A segment carrying a start stamp and no stop stamp belongs to a process that is no longer here — and
that observation can only honestly be made when the run is **next opened**. A writer that guessed at
write time would have to guess wrong at least once: at the moment of the crash, when nothing is
there to write. So an unclosed segment is marked `interrupted` on the next open, contributes no
runtime (it has no honest duration), and the run resumes from there.

### Locking

`run.lock` beside the record carries the pid, the hashed hostname, the start stamp and a heartbeat
touched at each CLOSE.

- **A live lock is a hard refusal.** Two engines on one run is corruption, not degradation.
- **A stale lock is stolen, loudly** — one warning plus a recorded event, because that is the one
  moment a run's history could be attributed to the wrong process. Stale means a dead pid *on this
  host* (a pid on another host tells you nothing), or a heartbeat colder than `STALE_HEARTBEAT_S`
  (**7 days** — deliberately generous: a live engine can sit in RESEARCH right through a long
  weekend, and the same-host dead-pid check catches the common crash promptly).
- `--finish` reads the lock only far enough to refuse a run another process is working, and opens
  no segment. `--show-config-drift` takes **no** lock at all and writes nothing — a command an
  operator runs in order to decide must not itself be a decision.

### What a resume refuses, and what it does not

| Refusal | Why |
|---|---|
| the address names no run | an address an operator typed must not silently become a *new* run |
| the address names more than one | a reassigned label has no honest single answer; the refusal lists the candidate ids |
| the run is `completed` | terminal: a published result may never quietly gain segments |
| the resolved mode differs from the frozen verdict | a paper run's results may not acquire live segments |
| **arbiter** drift (`gates`, `backtest`) | champions crowned under two sets of gates must not accumulate inside one experiment — lift it deliberately with `--allow-engine-upgrade` |
| `--mandate` / `--directive` / `--metric` | the steering is frozen; start a new run to research something else |
| `--run-limit-hours` / `--embed-all-sources` | frozen at creation; a cap raisable each morning would bound nothing, and a record whose contents depended on the last invocation would lose an earlier night's sources |

**Searcher** drift warns, records an event and proceeds. **No** drift is silent — a policy that says
something every time is one operators learn to skip.

Config drift is not a refusal at all: frozen wins, silently. `--show-config-drift` shows what
adopting the current files *would* change; `--rebase-config` adopts them and says so on the record.

### Retention

`noctis run-prune <address> [--dry-run]` is a **verb, not a flag**, and it is the one code path in
Noctis that deletes a run's files. It removes exactly three directories — `state/`, `strategies/`,
`reports/` — and never `run.json` or `index.json`: they *are* the long-term progress history, and
they are kilobytes against the megabytes it reclaims. Nothing prunes on a schedule, at startup or
from a config setting; the default keeps everything, forever.

Afterwards `run.state_pruned` reads `true`. That is the flag a reader checks before following a
`source_path` into the run tree: **everything the record carries survives** — an embedded champion
is readable a year after its run's tree was reclaimed — while any reference *out* of the record no
longer resolves.

---

## The listing roll-up: `index.json`

`workspace/runs/index.json` is `{schema_version, kind: "noctis.run-index", runs: [...]}`, newest run
id first, one entry per run directory. It is **derived, never authoritative**: the engine refreshes
it after every record write, `noctis runs` regenerates it from the records on disk, and a test pins
that a rebuild reproduces the incrementally-maintained file byte for byte. It carries no generation
stamp, precisely so those two paths can be compared.

Each entry: `run_id`, `label`, `status`, `created_utc`, `last_active_utc`, `segments` (a count),
`cumulative_runtime_s`, `run_limit_hours`, `complete`, `engine_version`, `comparable_key`,
`mixed_engine`, `readable`, `note`. A run whose record is missing or unreadable is listed as exactly
that — `readable: false`, every fact `null`, and the reason in `note` — because a broken record is
evidence and hiding it is the one thing a listing must never do.

`noctis runs` hides one kind of noise by default: a *finished* run with under 60 seconds of
cumulative runtime (a startup failure, a mistyped command). The count hidden is always printed, and
`--all` shows everything. Three kinds are never hidden — a `running` run, an unreadable one, and one
with zero segments (the adopted-history shape).

---

## What the record does not know

Stated here because a consumer that assumes otherwise would be quietly wrong.

- **The conversation research loop writes no session ledger.** Spend is read off
  `<run>/state/sessions/*.jsonl`, which the *episodic* driver writes. A run researched entirely
  through the conversation loop therefore reports `spend.tokens` and the buckets as `null` — an
  unknown bill, not a free one — even though the session itself accounted for its tokens in memory.
- **Coder-authoring completions are not token-metered.** An `AUTHOR` ledger line is a *marker* (an
  escalation to the paid fallback, or a refined-brief exit), journaled with zero tokens. The coder
  model's own usage is not on any ledger, so `spend` understates the true bill by whatever authoring
  cost. The stage appears in `by_stage` with honest zeros rather than being hidden.
- **There is no `strategies[].scorecard`.** A candidate's backtest numbers reach the record only as
  the `observed` values inside `gates[]`, beside the `threshold` each was measured against. (Some
  code comments describe a `scorecard` sub-object; it does not exist in schema version 1. The rule
  those comments are protecting does hold: `performance` is the paper account's realised record and
  never carries backtest numbers.)
- **`assumptions.symbol_holdout.symbols` is always `null`.** The held-out names are sampled per
  research session, so no run-level value would be true.
- **A benchmark the shared lake cannot price is absent, not approximated** — `null` statistics with
  a `note`.
- **`run.status` is not a liveness check.** `running` is equally the shape of a live engine and of a
  crash; `run.lock` is what distinguishes them.
- **Nothing in the record is a credential.** No secret's value *or name* survives anywhere, once
  `inputs.settings`' tier lists (where naming one is the point) are removed. Nor is any bar
  reconstructable from it: the record carries holdout *metrics* and holdout *geometry*, never
  holdout bars.

---

## Building a run page from this file

A website needs two fetches and no server logic.

1. **The listing page**: fetch `runs/index.json`. Group by `comparable_key` — never pool or rank
   across two keys — and mark any entry with `mixed_engine: true`, whose key describes only its
   latest engine. Show `run_limit_hours` beside `cumulative_runtime_s`: two runs given different
   compute are different experiments. Render `readable: false` entries as broken, with their `note`.
2. **The run page**: fetch `runs/<run_id>/run.json`. Then:
   - **Header** — `run.run_id`, `run.label`, `run.status`, `run.created_utc`, the segment count, and
     `engine.comparable_key`. If `run.complete` is `false`, say so: the record is a snapshot of a
     run whose last write did not land cleanly.
   - **Equity** — `performance.equity_curve` against `performance.benchmark`. If `performance` is
     `null`, render **"researching"**, never a flat 0% line: check `run.traded` to tell which of the
     two `null` cases you are in.
   - **The funnel** — group `strategies[]` by `outcome`, then by the `gates[]` entry whose `passed`
     is `false`. That is where *"47 of 66 died at the symbol-holdout gate"* comes from. Remember
     that an absent gate means "not reached".
   - **The honesty table** — render `assumptions` verbatim as a table. Two runs' blocks subtract,
     which is the point of it being data.
   - **Cost** — `spend.efficiency` for the headline ratios; `spend.by_model` / `by_stage` /
     `by_segment` for the breakdown. Every dollar figure is an **estimate** from a versioned price
     table (`spend.pricing_table_version`); label it as one.
   - **Provenance** — `inputs.mandate.text` (what the agent was told, as frozen), `inputs.models`,
     `environment_latest`, and `inputs.config_changes` / `engine.engine_changes` if either is
     non-empty: a run that changed mid-flight must be shown as one.
   - **Truncation** — if `run.truncated` is non-empty, say which list was bounded and by how much.
3. **Unknown keys**: ignore them. That is the versioning promise, and a consumer that fails on one
   breaks the moment the schema grows.

---

## A worked example

[`examples/run_record.json`](../examples/run_record.json) is a complete, schema-valid record built
by the real builder: a two-night run with a research-only third segment, one promoted champion, one
rejection that died at the symbol-holdout gate, one undecided draft, ten traded sessions and a
priced spend block. `tests/test_docs_run_record.py` validates it against `schema.validate()` and
checks that this page names every key the schema names, so neither can drift from the code without
a red test.

Its skeleton — the shape of every record:

```json
{
  "schema_version": 1,
  "kind": "noctis.run",
  "run": {
    "run_id": "20260727T142233Z-a1b2c3",
    "label": "nuclear-nights",
    "status": "stopped",
    "created_utc": "2026-07-27T14:22:33.418Z",
    "last_active_utc": "2026-08-08T00:10:00.000Z",
    "completed_utc": null,
    "run_limit_hours": 100.0,
    "traded": true,
    "cumulative_runtime_s": 63000.0,
    "cumulative_research_s": 54300.0,
    "cumulative_trading_s": 8000.0,
    "cumulative_trials": 111,
    "state_pruned": false,
    "complete": true,
    "truncated": {}
  },
  "segments": [{ "index": 0, "command": "run", "resumed": false, "...": "…" }],
  "environment_latest": { "hostname_hash": "9f2c1d3e4b5a", "...": "…" },
  "engine": {
    "engine_version": 2,
    "engine_epoch": 1,
    "noctis_version": "0.1.0",
    "fingerprint": { "gates": "0fb6148041c95608", "backtest": "3ba3e0bf1c97134f", "...": "…" },
    "comparable_key": "2|0fb6148041c95608|3ba3e0bf1c97134f|sharpe",
    "mixed_engine": false,
    "engine_changes": []
  },
  "inputs": { "config_epoch": 1, "execution_mode": "paper", "...": "…" },
  "strategies": [{ "name": "uranium_momo", "outcome": "promoted", "...": "…" }],
  "spend": { "llm_usd_estimate": 0.2466, "pricing_table_version": "2026-07.1", "...": "…" },
  "sessions": [{ "as_of": "2026-07-27", "equity": 100500.0, "...": "…" }],
  "performance": { "source": "paper_account", "...": "…" },
  "assumptions": { "paper_only": true, "fill_model": "next_bar_open", "...": "…" },
  "events": [],
  "errors": []
}
```

(The `"...": "…"` entries stand in for elided keys; the real record carries every key of every
section, which is what makes the elision safe to read.)

A few excerpts worth seeing in full. **A rejection, with the gate that stopped it** — note that the
list ends at the failure, because a rejection short-circuits:

```json
{
  "name": "fuel_cycle_meanrev",
  "outcome": "rejected",
  "tier": "__tmp",
  "decided_utc": "2026-07-28T19:02:55.120Z",
  "trials": 41,
  "gates": [
    { "gate": "validated", "passed": true, "observed": null, "threshold": null, "note": null },
    { "gate": "activity_floor", "passed": true, "observed": 0.08, "threshold": 0.02, "note": null },
    { "gate": "overfit_gap", "passed": true, "observed": 0.44, "threshold": 1.0, "note": null },
    { "gate": "reverse_gap", "passed": true, "observed": -0.44, "threshold": 1.0, "note": null },
    {
      "gate": "magnitude_cap",
      "passed": true,
      "observed": 0.61,
      "threshold": 0.0,
      "note": "inert: switched off by a max_test_metric of 0"
    },
    { "gate": "forward_holdout", "passed": true, "observed": 0.21, "threshold": 0.0, "note": null },
    { "gate": "symbol_holdout", "passed": false, "observed": -0.37, "threshold": 0.0, "note": null }
  ],
  "rationale": "rejected: symbol-holdout metric -0.3700 below bar 0.0000 (symbol-holdout gate)",
  "source_path": "strategies/__tmp/fuel_cycle_meanrev.py",
  "source_sha256": "1c9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f70819293a4b5c6",
  "source": null
}
```

The three gates it never reached — `symbol_consistency`, `family_slot`, and then whichever of
`minimum_bar` / `beat_weakest` the board would have applied — are simply **absent**, which is what "not reached"
looks like. `magnitude_cap` is present with a note because a gate that *could not bite* still has
to be on the record, or a funnel's denominators stop being honest.

**A research-only segment** — `command` is the whole answer to "which nights were research-only?":

```json
{
  "index": 2,
  "started_utc": "2026-08-07T22:40:00.000Z",
  "stopped_utc": "2026-08-08T00:10:00.000Z",
  "duration_s": 5400.0,
  "stopped_reason": "agent_done",
  "status": "stopped",
  "argv": ["research", "--resume", "@nuclear-nights"],
  "command": "research",
  "resumed": true,
  "counters": { "sessions": 1, "research_iterations": 6, "research_promotions": 0 },
  "phase_seconds": { "RESEARCH": 5400.0 },
  "environment": { "hostname_hash": "9f2c1d3e4b5a", "...": "…" },
  "engine_version": 2,
  "engine_fingerprint": { "gates": "0fb6148041c95608", "...": "…" }
}
```

**The deflated Sharpe, beside the count that deflated it** — this project can compute the
multiple-testing correction honestly because the trial count is journaled rather than estimated:

```json
{
  "sharpe": 4.503660030772,
  "sortino": 8.581640357518,
  "calmar": 107.196211234448,
  "psr": 0.779550808377,
  "deflated_sharpe": 0.036213259062,
  "n_trials_used": 111,
  "deflation_basis": "sharpe_standard_error_under_the_zero_sharpe_null",
  "skew": -0.269760459321,
  "excess_kurtosis": -1.623097691138,
  "annualization_basis": 252
}
```

---

## See also

- [cli.md](cli.md) — every verb and flag that mints, resumes, reads, seals or prunes a run
- [configuration.md](configuration.md#config-freezing--what-a-resumed-run-reads) — the freezing
  tiers in full, and the mandate overlay
- [architecture.md](architecture.md#where-state-lives) — the run tree, the shared lake, per-run
  memory, and the read-only seeds
- [validation.md](validation.md) — the promotion gate order the `gates[]` evidence describes
- [safety.md](safety.md) — the two live-money gates the record is forbidden to carry
